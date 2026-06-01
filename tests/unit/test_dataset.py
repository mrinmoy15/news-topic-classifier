"""Unit tests for news_topic_classifier.dataset."""
from __future__ import annotations

import re
from unittest.mock import MagicMock, patch

import pytest
import torch

from news_topic_classifier.config import LABEL2ID
from news_topic_classifier.dataset import (
    BBCNewsDataset,
    _build_extraction_query,
    _gcs_output_path,
    extract_from_bigquery,
)


# ─── _gcs_output_path ────────────────────────────────────────────────────────

def test_gcs_output_path_starts_with_gs():
    assert _gcs_output_path("my-bucket").startswith("gs://my-bucket/")


def test_gcs_output_path_ends_with_parquet():
    assert _gcs_output_path("my-bucket").endswith("bbc_news.parquet")


def test_gcs_output_path_contains_raw_segment():
    assert "/data/raw/" in _gcs_output_path("my-bucket")


def test_gcs_output_path_timestamp_format():
    uri = _gcs_output_path("my-bucket")
    assert re.search(r"\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}", uri)


def test_gcs_output_path_uses_bucket_name():
    assert "custom-bucket" in _gcs_output_path("custom-bucket")


# ─── _build_extraction_query ─────────────────────────────────────────────────

def test_extraction_query_full_table_no_fingerprint():
    sql = _build_extraction_query("proj.ds.table", "body", "category", "title", None, None)
    assert "FARM_FINGERPRINT" not in sql


def test_extraction_query_sample_uses_fingerprint():
    sql = _build_extraction_query("proj.ds.table", "body", "category", "title", 1000, 100)
    assert "FARM_FINGERPRINT" in sql


def test_extraction_query_has_case_clause_for_all_labels():
    sql = _build_extraction_query("proj.ds.table", "body", "category", "title", None, None)
    for label in LABEL2ID:
        assert label in sql


def test_extraction_query_filters_null_text():
    sql = _build_extraction_query("proj.ds.table", "body", "category", "title", None, None)
    assert "IS NOT NULL" in sql


def test_extraction_query_references_source_table():
    sql = _build_extraction_query("proj.ds.table", "body", "category", "title", None, None)
    assert "proj.ds.table" in sql


def test_extraction_query_sample_pct_capped_at_100():
    # sample_size > input_table_size → pct should be capped at 100
    sql = _build_extraction_query("proj.ds.table", "body", "category", "title", 10, 9999)
    assert "< 100" in sql


def test_extraction_query_sample_pct_calculated_correctly():
    # 100 / 1000 = 10%
    sql = _build_extraction_query("proj.ds.table", "body", "category", "title", 1000, 100)
    assert "< 10" in sql


# ─── BBCNewsDataset ───────────────────────────────────────────────────────────

def test_dataset_len(tmp_parquet, dummy_tokenizer):
    ds = BBCNewsDataset(tmp_parquet, dummy_tokenizer, max_length=16)
    assert len(ds) == 10


def test_dataset_getitem_returns_correct_keys(tmp_parquet, dummy_tokenizer):
    ds = BBCNewsDataset(tmp_parquet, dummy_tokenizer, max_length=16)
    assert set(ds[0].keys()) == {"input_ids", "attention_mask", "label"}


def test_dataset_input_ids_shape(tmp_parquet, dummy_tokenizer):
    ds = BBCNewsDataset(tmp_parquet, dummy_tokenizer, max_length=16)
    assert ds[0]["input_ids"].shape == (16,)


def test_dataset_attention_mask_shape(tmp_parquet, dummy_tokenizer):
    ds = BBCNewsDataset(tmp_parquet, dummy_tokenizer, max_length=16)
    assert ds[0]["attention_mask"].shape == (16,)


def test_dataset_label_dtype_is_long(tmp_parquet, dummy_tokenizer):
    ds = BBCNewsDataset(tmp_parquet, dummy_tokenizer, max_length=16)
    assert ds[0]["label"].dtype == torch.long


def test_dataset_label_values_in_valid_range(tmp_parquet, dummy_tokenizer):
    ds = BBCNewsDataset(tmp_parquet, dummy_tokenizer, max_length=16)
    for i in range(len(ds)):
        assert 0 <= ds[i]["label"].item() < 5


def test_dataset_use_title_prepends_title(tmp_parquet):
    """When use_title=True the tokenizer receives 'title body' text."""
    seen = []

    class _CaptureTok:
        def __call__(self, text, **kwargs):
            seen.append(text)
            return {
                "input_ids":      torch.ones(1, 16, dtype=torch.long),
                "attention_mask": torch.ones(1, 16, dtype=torch.long),
            }

    ds = BBCNewsDataset(tmp_parquet, _CaptureTok(), max_length=16, use_title=True)
    _ = ds[0]
    assert seen[0].startswith("title 0")
    assert "body text 0" in seen[0]


def test_dataset_use_title_false_passes_body_only(tmp_parquet):
    """When use_title=False the tokenizer receives only the body text."""
    seen = []

    class _CaptureTok:
        def __call__(self, text, **kwargs):
            seen.append(text)
            return {
                "input_ids":      torch.ones(1, 16, dtype=torch.long),
                "attention_mask": torch.ones(1, 16, dtype=torch.long),
            }

    ds = BBCNewsDataset(tmp_parquet, _CaptureTok(), max_length=16, use_title=False)
    _ = ds[0]
    assert seen[0] == "body text 0"


# ─── extract_from_bigquery ───────────────────────────────────────────────────

def _mock_bq_client():
    """Build a MagicMock BigQuery client where every job succeeds immediately."""
    mock_job = MagicMock()
    mock_job.result.return_value = None

    client = MagicMock()
    client.query.return_value         = mock_job
    client.extract_table.return_value = mock_job
    client.get_table.return_value.num_rows = 100
    return client


def test_extract_from_bigquery_returns_gs_uri():
    with patch("news_topic_classifier.dataset.bigquery.Client", return_value=_mock_bq_client()):
        result = extract_from_bigquery(
            gcp_project="test-project",
            bq_dataset="TEST_DS",
            source_table="proj.ds.table",
            text_col="body",
            label_col="category",
            title_col="title",
            gcs_bucket_data="test-bucket",
            input_table_size=None,
        )

    assert result.startswith("gs://test-bucket/data/raw/")
    assert result.endswith("bbc_news.parquet")


def test_extract_from_bigquery_runs_query_and_export():
    client = _mock_bq_client()
    with patch("news_topic_classifier.dataset.bigquery.Client", return_value=client):
        extract_from_bigquery(
            gcp_project="test-project",
            bq_dataset="TEST_DS",
            source_table="proj.ds.table",
            text_col="body",
            label_col="category",
            title_col="title",
            gcs_bucket_data="test-bucket",
            input_table_size=None,
        )

    client.query.assert_called_once()
    client.extract_table.assert_called_once()
    client.delete_table.assert_called_once()


def test_extract_from_bigquery_with_sample_size():
    with patch("news_topic_classifier.dataset.bigquery.Client", return_value=_mock_bq_client()):
        result = extract_from_bigquery(
            gcp_project="test-project",
            bq_dataset="TEST_DS",
            source_table="proj.ds.table",
            text_col="body",
            label_col="category",
            title_col="title",
            gcs_bucket_data="test-bucket",
            input_table_size=2225,
            sample_size=500,
        )

    assert result.startswith("gs://test-bucket/")
