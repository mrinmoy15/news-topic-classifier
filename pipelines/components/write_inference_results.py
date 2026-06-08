import os

from kfp import dsl

_TRAINER_IMAGE = os.environ.get(
    "TRAINER_IMAGE",
    "us-central1-docker.pkg.dev/cs-cdwp-data-dev2188/news-topic-classifier/trainer:latest",
)


@dsl.component(base_image=_TRAINER_IMAGE)
def write_inference_results_component(
    gcp_project: str,
    bq_dataset: str,
    gcs_predictions_uri: str,
    predictions_table: str = "news_topic_classifier_predictions",
) -> int:
    """
    KFP component — load a predictions Parquet from GCS and stream-insert it into BigQuery.

    The output table preserves all original BBC News columns (title, body, category)
    and adds a single predicted_label column.  Returns the number of rows written.

    Parameters
    ----------
    gcp_project : str
        GCP project ID.
    bq_dataset : str
        BigQuery dataset for the predictions table.
    gcs_predictions_uri : str
        GCS URI of the predictions Parquet from run_batch_inference_component.
    predictions_table : str
        Destination table name.

    Returns
    -------
    int
        Number of rows written to BigQuery.
    """
    import io
    import os
    import tempfile

    import pyarrow.parquet as pq
    from google.cloud import bigquery, storage

    # ── Download predictions Parquet from GCS ─────────────────────────────────
    gcs_path = gcs_predictions_uri.replace("gs://", "")
    bucket_name, blob_name = gcs_path.split("/", 1)

    fd, tmp_path = tempfile.mkstemp(suffix=".parquet")
    os.close(fd)
    try:
        storage.Client(project=gcp_project).bucket(bucket_name).blob(blob_name).download_to_filename(tmp_path)
        with open(tmp_path, "rb") as _fh:
            _buf = io.BytesIO(_fh.read())
    finally:
        os.unlink(tmp_path)
    pred_table = pq.read_table(_buf)

    rows = pred_table.to_pylist()
    print(f"Loaded {len(rows)} predictions from {gcs_predictions_uri}")

    # ── Ensure BigQuery table exists ──────────────────────────────────────────
    bq     = bigquery.Client(project=gcp_project)
    schema = [
        bigquery.SchemaField("title",           "STRING"),
        bigquery.SchemaField("body",            "STRING"),
        bigquery.SchemaField("category",        "STRING"),
        bigquery.SchemaField("predicted_label", "STRING"),
    ]
    full_ref = f"{gcp_project}.{bq_dataset}.{predictions_table}"
    bq_tbl   = bigquery.Table(full_ref, schema=schema)
    bq.create_table(bq_tbl, exists_ok=True)
    print(f"Predictions table ready: {full_ref}")

    # ── Stream insert ─────────────────────────────────────────────────────────
    errors = bq.insert_rows_json(full_ref, rows)
    if errors:
        raise RuntimeError(f"BigQuery insert errors: {errors}")

    print(f"Wrote {len(rows)} rows to {full_ref}")
    return len(rows)
