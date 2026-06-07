"""
Integration tests for the three-step BBC News inference pipeline.

Test tiers
----------
Tier 1 — data fetch  (~1 min, no GPU needed)
    test_07_fetch_inference_data    BQ partition → GCS Parquet

Tier 2 — inference + write  (marked `slow`, ~5-10 min, requires model on GCS)
    test_08_run_batch_inference     GCS Parquet + BERT → predictions GCS Parquet
    test_09_write_inference_results GCS Parquet → BigQuery (test table)

Run all tiers:
    INTEGRATION_TESTS=true pytest tests/integration/test_inference_pipeline.py -m integration -v

Run data-fetch only (skip slow):
    INTEGRATION_TESTS=true pytest tests/integration/test_inference_pipeline.py -m "integration and not slow" -v
"""
from __future__ import annotations

import os

import pytest
from google.cloud import bigquery, storage

# ─── Fixed day partition used across all tests ────────────────────────────────
# Use day=0 so the test is deterministic regardless of when it runs.
_TEST_DAY = 0

# Separate BQ table so test rows never mix with real predictions.
_TEST_PREDICTIONS_TABLE = "news_topic_classifier_predictions_integration_test"


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def inference_artifacts():
    """Mutable dict passed through all tests — each step stores its output here."""
    return {}


@pytest.fixture(scope="module", autouse=True)
def cleanup_inference_gcs(cfg, test_run_prefix):
    """Delete GCS objects written under the inference test prefix after the module."""
    yield

    if not os.getenv("INTEGRATION_TESTS"):
        return

    client = storage.Client(project=cfg.environment.gcp_project)
    bucket = client.bucket(cfg.environment.gcs_bucket_data)
    blobs  = list(bucket.list_blobs(prefix=f"inference/day={_TEST_DAY}/"))
    for blob in blobs:
        blob.delete()
    if blobs:
        print(f"\n[cleanup] Deleted {len(blobs)} GCS inference objects for day={_TEST_DAY}")


@pytest.fixture(scope="module", autouse=True)
def cleanup_inference_bq(cfg):
    """Delete all rows written to the integration-test BQ predictions table."""
    yield

    if not os.getenv("INTEGRATION_TESTS"):
        return

    bq      = bigquery.Client(project=cfg.environment.gcp_project)
    table   = f"{cfg.environment.gcp_project}.{cfg.environment.bq_dataset}.{_TEST_PREDICTIONS_TABLE}"
    try:
        bq.delete_table(table)
        print(f"\n[cleanup] Deleted BQ table {table}")
    except Exception as e:
        print(f"\n[cleanup] Could not delete BQ table {table}: {e}")


# ─── Tests ────────────────────────────────────────────────────────────────────

@pytest.mark.integration
def test_07_fetch_inference_data(cfg, inference_artifacts):
    """Fetch day=0 BBC News partition from BigQuery and write to GCS Parquet."""
    from pipelines.components.fetch_inference_data import fetch_inference_data_component

    _fn = fetch_inference_data_component.python_func

    gcs_uri = _fn(
        gcp_project=cfg.environment.gcp_project,
        bq_dataset=cfg.environment.bq_dataset,
        gcs_bucket_data=cfg.environment.gcs_bucket_data,
        source_table="bigquery-public-data.bbc_news.fulltext",
        day=_TEST_DAY,
    )

    # Verify shape of the URI
    assert gcs_uri.startswith("gs://")
    assert f"day={_TEST_DAY}" in gcs_uri
    assert gcs_uri.endswith("input.parquet")

    # Verify the file actually exists in GCS
    path      = gcs_uri.replace("gs://", "")
    bucket_name, blob_name = path.split("/", 1)
    client    = storage.Client(project=cfg.environment.gcp_project)
    blob      = client.bucket(bucket_name).blob(blob_name)
    assert blob.exists(), f"Expected GCS file not found: {gcs_uri}"

    inference_artifacts["gcs_input_uri"] = gcs_uri
    print(f"\n[fetch] {gcs_uri}")


@pytest.mark.integration
@pytest.mark.slow
def test_08_run_batch_inference(cfg, inference_artifacts):
    """Load fine-tuned BERT from GCS, run inference on fetched Parquet."""
    if "gcs_input_uri" not in inference_artifacts:
        pytest.skip("test_07_fetch_inference_data did not complete — skipping")

    gcs_model_uri = (
        f"gs://{cfg.environment.gcs_bucket_artifacts}/models/bert-bbc-finetuned/"
    )

    # Check that the model exists before attempting to download it
    client = storage.Client(project=cfg.environment.gcp_project)
    path   = gcs_model_uri.replace("gs://", "")
    bucket_name, prefix = path.split("/", 1)
    blobs = list(client.bucket(bucket_name).list_blobs(prefix=prefix, max_results=1))
    if not blobs:
        pytest.skip(
            f"No fine-tuned model found at {gcs_model_uri}. "
            "Run the training pipeline first."
        )

    from pipelines.components.run_batch_inference import run_batch_inference_component

    _fn = run_batch_inference_component.python_func

    gcs_predictions_uri = _fn(
        gcp_project=cfg.environment.gcp_project,
        gcs_model_uri=gcs_model_uri,
        gcs_input_uri=inference_artifacts["gcs_input_uri"],
        gcs_bucket_data=cfg.environment.gcs_bucket_data,
        day=_TEST_DAY,
        batch_size=cfg.training.batch_size,
        max_seq_length=cfg.model.max_seq_length,
    )

    assert gcs_predictions_uri.startswith("gs://")
    assert f"day={_TEST_DAY}" in gcs_predictions_uri
    assert gcs_predictions_uri.endswith("predictions.parquet")

    # Verify predictions Parquet exists and has the right columns
    import io
    import tempfile
    import pyarrow.parquet as pq

    pred_path   = gcs_predictions_uri.replace("gs://", "")
    bucket_name, blob_name = pred_path.split("/", 1)
    blob        = client.bucket(bucket_name).blob(blob_name)
    assert blob.exists(), f"Predictions Parquet not found: {gcs_predictions_uri}"

    fd, tmp_path = tempfile.mkstemp(suffix=".parquet")
    os.close(fd)
    try:
        blob.download_to_filename(tmp_path)
        with open(tmp_path, "rb") as _fh:
            _buf = io.BytesIO(_fh.read())
    finally:
        os.unlink(tmp_path)
    table = pq.read_table(_buf)

    required_cols = {
        "prediction_date", "run_timestamp", "title", "body",
        "true_label", "predicted_label", "confidence",
        "score_business", "score_entertainment", "score_politics",
        "score_sport", "score_tech", "day_partition",
    }
    assert required_cols.issubset(set(table.schema.names))
    assert table.num_rows > 0

    rows = table.to_pylist()
    for row in rows:
        assert 0.0 <= row["confidence"] <= 1.0
        score_total = sum(
            row[f"score_{l}"]
            for l in ("business", "entertainment", "politics", "sport", "tech")
        )
        assert abs(score_total - 1.0) < 1e-3
        assert row["predicted_label"] in {"business", "entertainment", "politics", "sport", "tech"}
        assert row["day_partition"] == _TEST_DAY

    inference_artifacts["gcs_predictions_uri"] = gcs_predictions_uri
    inference_artifacts["prediction_count"]    = table.num_rows
    print(f"\n[infer] {table.num_rows} predictions → {gcs_predictions_uri}")


@pytest.mark.integration
@pytest.mark.slow
def test_09_write_inference_results(cfg, inference_artifacts):
    """Stream-insert predictions Parquet into BigQuery and verify row count."""
    if "gcs_predictions_uri" not in inference_artifacts:
        pytest.skip("test_08_run_batch_inference did not complete — skipping")

    from pipelines.components.write_inference_results import write_inference_results_component

    _fn = write_inference_results_component.python_func

    rows_written = _fn(
        gcp_project=cfg.environment.gcp_project,
        bq_dataset=cfg.environment.bq_dataset,
        gcs_predictions_uri=inference_artifacts["gcs_predictions_uri"],
        predictions_table=_TEST_PREDICTIONS_TABLE,
    )

    assert rows_written == inference_artifacts["prediction_count"]
    assert rows_written > 0

    # Verify rows actually landed in BigQuery
    import time
    time.sleep(5)  # allow streaming buffer to flush

    bq        = bigquery.Client(project=cfg.environment.gcp_project)
    table_ref = f"{cfg.environment.gcp_project}.{cfg.environment.bq_dataset}.{_TEST_PREDICTIONS_TABLE}"
    result    = list(bq.query(f"SELECT COUNT(*) AS n FROM `{table_ref}`").result())
    bq_count  = result[0]["n"]

    assert bq_count == rows_written, (
        f"Expected {rows_written} rows in BQ but found {bq_count}"
    )
    print(f"\n[write] {rows_written} rows → {table_ref}")
