"""
Integration test fixtures.

Requirements
------------
- Set INTEGRATION_TESTS=true in your environment (or in CI secrets).
- GCP Application Default Credentials with access to cs-cdwp-data-dev2188.
- The base BERT model must be present at:
    gs://cs-cdwp-data-dev2188-model-artifacts/models/base-models/bert-base-uncased/

All tests in this directory are skipped automatically when
INTEGRATION_TESTS is not set, so they never block the normal unit-test suite.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

import pytest
from google.cloud import storage
from omegaconf import OmegaConf


# ─── Skip gate ───────────────────────────────────────────────────────────────

def pytest_runtest_setup(item):
    if "integration" in item.keywords and not os.getenv("INTEGRATION_TESTS"):
        pytest.skip("Set INTEGRATION_TESTS=true to run integration tests")


# ─── Dev environment config ──────────────────────────────────────────────────

@pytest.fixture(scope="module")
def cfg():
    """OmegaConf config pointing at the dev GCP environment.

    Uses seq_length=128 and epochs=1 to keep training fast.
    """
    return OmegaConf.create({
        "project": {"name": "news-topic-classifier"},
        "environment": {
            "name": "dev",
            "gcp_project": "cs-cdwp-data-dev2188",
            "gcp_region": "us-central1",
            "bq_dataset": "DATA_SCNCE_DEV_DATA",
            "bq_source_table": "bigquery-public-data.bbc_news.fulltext",
            "bq_results_table": "news_topic_classifier_results",
            "gcs_bucket_data": "cs-cdwp-data-dev2188-model-data",
            "gcs_bucket_artifacts": "cs-cdwp-data-dev2188-model-artifacts",
            "artifact_registry_repo": "news-topic-classifier",
            "mlflow": {
                # Use Cloud Run endpoint only inside GCP (CI/Vertex AI) where
                # metadata.google.internal is reachable for OIDC auth.
                # Locally (Windows / Mac) fall back to SQLite so the metadata
                # server lookup in _setup_mlflow_tracking is never triggered.
                "tracking_uri": os.getenv(
                    "MLFLOW_TRACKING_URI",
                    "sqlite:///integration_mlflow.db",
                )
            },
        },
        "data": {
            "bq_source_table": "bigquery-public-data.bbc_news.fulltext",
            "bq_text_column": "body",
            "bq_label_column": "category",
            "bq_title_column": "title",
        },
        "model": {
            "bert_base_model": "bert-base-uncased",
            "num_labels": 5,
            "max_seq_length": 128,  # shorter than production 512 for speed
        },
        "training": {
            "epochs": 1,
            "batch_size": 8,
            "lr": 2e-5,
            "warmup_steps": 10,
            "weight_decay": 0.01,
            "val_split": 0.1,
            "test_split": 0.1,
            "early_stopping_patience": 3,
        },
    })


# ─── GCS test namespace + cleanup ────────────────────────────────────────────

@pytest.fixture(scope="module")
def test_run_prefix():
    """Unique GCS path segment for this test run: integration-tests/<timestamp>/"""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    return f"integration-tests/{ts}"


@pytest.fixture(scope="module", autouse=True)
def cleanup_gcs(cfg, test_run_prefix):
    """Delete all GCS objects created under test_run_prefix after the module finishes."""
    yield  # run all tests first

    if not os.getenv("INTEGRATION_TESTS"):
        return

    gcp_project = cfg.environment.gcp_project
    client      = storage.Client(project=gcp_project)

    for bucket_name in (cfg.environment.gcs_bucket_data,
                        cfg.environment.gcs_bucket_artifacts):
        bucket = client.bucket(bucket_name)
        blobs  = list(bucket.list_blobs(prefix=test_run_prefix))
        for blob in blobs:
            blob.delete()
        if blobs:
            print(f"\n[cleanup] Deleted {len(blobs)} objects from gs://{bucket_name}/{test_run_prefix}")


# ─── Shared pipeline state ───────────────────────────────────────────────────

@pytest.fixture(scope="module")
def pipeline_artifacts():
    """Mutable dict passed through all tests — each step stores its output URI here."""
    return {}
