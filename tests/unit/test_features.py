"""Unit tests for news_topic_classifier.features."""
from __future__ import annotations

import re
from unittest.mock import MagicMock, patch

from news_topic_classifier.features import (
    _build_preprocessing_query,
    _gcs_split_output_paths,
    preprocess_and_split,
)


# ─── _build_preprocessing_query ───────────────────────────────────────────────

def test_preprocessing_query_references_source_table():
    sql = _build_preprocessing_query("proj.ds.raw_ext", val_pct=10, test_pct=10)
    assert "proj.ds.raw_ext" in sql


def test_preprocessing_query_assigns_test_split():
    sql = _build_preprocessing_query("proj.ds.raw_ext", val_pct=10, test_pct=10)
    assert "'test'" in sql


def test_preprocessing_query_assigns_val_split():
    sql = _build_preprocessing_query("proj.ds.raw_ext", val_pct=10, test_pct=10)
    assert "'val'" in sql


def test_preprocessing_query_assigns_train_split():
    sql = _build_preprocessing_query("proj.ds.raw_ext", val_pct=10, test_pct=10)
    assert "'train'" in sql


def test_preprocessing_query_embeds_test_pct():
    sql = _build_preprocessing_query("proj.ds.raw_ext", val_pct=10, test_pct=15)
    assert "15" in sql


def test_preprocessing_query_val_boundary_is_test_plus_val():
    # SQL emits the boundary as a literal expression: "10 + 20"
    sql = _build_preprocessing_query("proj.ds.raw_ext", val_pct=20, test_pct=10)
    assert "10 + 20" in sql


def test_preprocessing_query_uses_farm_fingerprint():
    sql = _build_preprocessing_query("proj.ds.raw_ext", val_pct=10, test_pct=10)
    assert "FARM_FINGERPRINT" in sql


def test_preprocessing_query_applies_nfkc_normalisation():
    sql = _build_preprocessing_query("proj.ds.raw_ext", val_pct=10, test_pct=10)
    assert "NFKC" in sql


def test_preprocessing_query_strips_html_tags():
    sql = _build_preprocessing_query("proj.ds.raw_ext", val_pct=10, test_pct=10)
    # HTML tag removal is done via REGEXP_REPLACE with a pattern matching <...>
    assert "REGEXP_REPLACE" in sql


def test_preprocessing_query_is_string():
    result = _build_preprocessing_query("proj.ds.raw_ext", val_pct=10, test_pct=10)
    assert isinstance(result, str)


# ─── _gcs_split_output_paths ──────────────────────────────────────────────────

def test_split_paths_has_all_three_keys():
    paths = _gcs_split_output_paths("my-bucket")
    assert set(paths.keys()) == {"train", "val", "test"}


def test_split_paths_all_start_with_gs():
    paths = _gcs_split_output_paths("my-bucket")
    for uri in paths.values():
        assert uri.startswith("gs://my-bucket/")


def test_split_paths_filenames_match_split_names():
    paths = _gcs_split_output_paths("my-bucket")
    for split in ("train", "val", "test"):
        assert paths[split].endswith(f"{split}.parquet")


def test_split_paths_share_single_timestamp():
    """All three URIs must reference the same versioned directory."""
    paths = _gcs_split_output_paths("my-bucket")
    timestamps = set()
    for uri in paths.values():
        m = re.search(r"processed/(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})/", uri)
        assert m, f"No timestamp in {uri}"
        timestamps.add(m.group(1))
    assert len(timestamps) == 1


def test_split_paths_contain_processed_segment():
    paths = _gcs_split_output_paths("my-bucket")
    for uri in paths.values():
        assert "/data/processed/" in uri


def test_split_paths_use_bucket_name():
    paths = _gcs_split_output_paths("custom-bucket-name")
    for uri in paths.values():
        assert "custom-bucket-name" in uri


# ─── preprocess_and_split ────────────────────────────────────────────────────

def _mock_bq_client():
    """BigQuery client where every job/table call succeeds immediately."""
    mock_job = MagicMock()
    mock_job.result.return_value = None

    client = MagicMock()
    client.query.return_value         = mock_job
    client.extract_table.return_value = mock_job
    client.get_table.return_value.num_rows = 50
    return client


def test_preprocess_and_split_returns_three_split_keys():
    with patch("news_topic_classifier.features.bigquery.Client", return_value=_mock_bq_client()):
        result = preprocess_and_split(
            gcp_project="test-project",
            bq_dataset="TEST_DS",
            raw_gcs_uri="gs://bucket/data/raw/2026/bbc_news.parquet",
            gcs_bucket_data="test-bucket",
        )

    assert set(result.keys()) == {"train", "val", "test"}


def test_preprocess_and_split_uris_start_with_gs():
    with patch("news_topic_classifier.features.bigquery.Client", return_value=_mock_bq_client()):
        result = preprocess_and_split(
            gcp_project="test-project",
            bq_dataset="TEST_DS",
            raw_gcs_uri="gs://bucket/data/raw/2026/bbc_news.parquet",
            gcs_bucket_data="test-bucket",
        )

    for uri in result.values():
        assert uri.startswith("gs://test-bucket/")


def test_preprocess_and_split_creates_external_table_then_cleans_up():
    client = _mock_bq_client()
    with patch("news_topic_classifier.features.bigquery.Client", return_value=client):
        preprocess_and_split(
            gcp_project="test-project",
            bq_dataset="TEST_DS",
            raw_gcs_uri="gs://bucket/data/raw/2026/bbc_news.parquet",
            gcs_bucket_data="test-bucket",
        )

    # create_table called once (external table)
    client.create_table.assert_called_once()
    # extract_table called 3 times (one per split)
    assert client.extract_table.call_count == 3
    # delete_table called: 3 split temps + 1 staging + 1 external = 5, plus initial not_found_ok=True
    assert client.delete_table.call_count >= 5


def test_preprocess_and_split_custom_splits():
    with patch("news_topic_classifier.features.bigquery.Client", return_value=_mock_bq_client()):
        result = preprocess_and_split(
            gcp_project="test-project",
            bq_dataset="TEST_DS",
            raw_gcs_uri="gs://bucket/data/raw/2026/bbc_news.parquet",
            gcs_bucket_data="test-bucket",
            val_split=0.15,
            test_split=0.15,
        )

    assert set(result.keys()) == {"train", "val", "test"}
