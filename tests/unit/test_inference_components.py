"""Unit tests for the three inference KFP components.

All GCP I/O (BigQuery, GCS) is mocked.  No real credentials or network
access is required.
"""
from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from pipelines.components.fetch_inference_data import fetch_inference_data_component
from pipelines.components.run_batch_inference import run_batch_inference_component
from pipelines.components.write_inference_results import write_inference_results_component

_fetch_fn = fetch_inference_data_component.python_func
_infer_fn = run_batch_inference_component.python_func
_write_fn = write_inference_results_component.python_func

ID2LABEL = {0: "business", 1: "entertainment", 2: "politics", 3: "sport", 4: "tech"}

# ─── Shared helpers ───────────────────────────────────────────────────────────

_SAMPLE_ROWS = [
    {"title": f"Title {i}", "body": f"Body text for article {i}", "true_label": "tech"}
    for i in range(6)
]

_PRED_ROWS = [
    {
        "prediction_date":     "2026-06-06",
        "run_timestamp":       "2026-06-06T11:00:00+00:00",
        "title":               f"Title {i}",
        "body":                f"Body {i}",
        "true_label":          "tech",
        "predicted_label":     "tech",
        "confidence":          0.95,
        "score_business":      0.01,
        "score_entertainment": 0.01,
        "score_politics":      0.01,
        "score_sport":         0.01,
        "score_tech":          0.95,
        "day_partition":       0,
    }
    for i in range(6)
]


def _mock_bq_client(rows=None):
    rows = rows if rows is not None else _SAMPLE_ROWS
    mock_bq = MagicMock()
    mock_bq.return_value.query.return_value.result.return_value = rows
    return mock_bq


def _mock_gcs_client(download_rows=None):
    """GCS client whose download_to_filename writes a real Parquet to disk."""
    rows = download_rows if download_rows is not None else _SAMPLE_ROWS

    def _write_parquet(path):
        pq.write_table(pa.Table.from_pylist(rows), path)

    mock_gcs = MagicMock()
    mock_gcs.return_value.bucket.return_value.blob.return_value.download_to_filename.side_effect = _write_parquet
    return mock_gcs


# ═══════════════════════════════════════════════════════════════════════════════
# fetch_inference_data_component
# ═══════════════════════════════════════════════════════════════════════════════

class TestFetchInferenceData:

    def test_explicit_day_used_in_query(self):
        mock_bq = _mock_bq_client()
        with patch("google.cloud.bigquery.Client", mock_bq), \
             patch("google.cloud.storage.Client", _mock_gcs_client()):
            _fetch_fn(
                gcp_project="proj", bq_dataset="ds",
                gcs_bucket_data="bucket", day=7,
            )
        query_sql = mock_bq.return_value.query.call_args[0][0]
        assert "day_num = 7" in query_sql.replace(" ", "").replace("\n", "").replace("=", " = ")

    def test_auto_day_is_valid_partition(self):
        """day=-1 should compute (UTC day - 1) % 30 — always in 0..29."""
        bq_calls = []

        def _capturing_bq(*args, **kwargs):
            client = MagicMock()
            def _query(sql, **kw):
                bq_calls.append(sql)
                q = MagicMock()
                q.result.return_value = _SAMPLE_ROWS
                return q
            client.query.side_effect = _query
            return client

        with patch("google.cloud.bigquery.Client", _capturing_bq), \
             patch("google.cloud.storage.Client", _mock_gcs_client()):
            _fetch_fn(gcp_project="proj", bq_dataset="ds", gcs_bucket_data="bucket", day=-1)

        sql = bq_calls[0]
        day_val = (datetime.now(timezone.utc).day - 1) % 30
        assert f"day_num = {day_val}" in sql

    def test_raises_on_empty_result(self):
        with patch("google.cloud.bigquery.Client", _mock_bq_client(rows=[])), \
             patch("google.cloud.storage.Client", _mock_gcs_client()):
            with pytest.raises(ValueError, match="No articles found"):
                _fetch_fn(
                    gcp_project="proj", bq_dataset="ds",
                    gcs_bucket_data="bucket", day=0,
                )

    def test_returns_gcs_uri_with_day(self):
        with patch("google.cloud.bigquery.Client", _mock_bq_client()), \
             patch("google.cloud.storage.Client", _mock_gcs_client()):
            result = _fetch_fn(
                gcp_project="proj", bq_dataset="ds",
                gcs_bucket_data="my-bucket", day=3,
            )
        assert result == "gs://my-bucket/inference/day=3/input.parquet"

    def test_uploads_parquet_to_gcs(self):
        mock_gcs = _mock_gcs_client()
        with patch("google.cloud.bigquery.Client", _mock_bq_client()), \
             patch("google.cloud.storage.Client", mock_gcs):
            _fetch_fn(
                gcp_project="proj", bq_dataset="ds",
                gcs_bucket_data="bucket", day=1,
            )
        upload_call = mock_gcs.return_value.bucket.return_value.blob.return_value.upload_from_filename
        upload_call.assert_called_once()

    def test_source_table_appears_in_query(self):
        bq_calls = []

        def _capturing_bq(*args, **kwargs):
            client = MagicMock()
            def _query(sql, **kw):
                bq_calls.append(sql)
                q = MagicMock()
                q.result.return_value = _SAMPLE_ROWS
                return q
            client.query.side_effect = _query
            return client

        with patch("google.cloud.bigquery.Client", _capturing_bq), \
             patch("google.cloud.storage.Client", _mock_gcs_client()):
            _fetch_fn(
                gcp_project="proj", bq_dataset="ds", gcs_bucket_data="bucket",
                source_table="my-proj.dataset.table", day=0,
            )

        assert "my-proj.dataset.table" in bq_calls[0]


# ═══════════════════════════════════════════════════════════════════════════════
# run_batch_inference_component
# ═══════════════════════════════════════════════════════════════════════════════

def _fake_tokenizer(texts, **kwargs):
    n = len(texts)
    return {
        "input_ids":      torch.zeros(n, 16, dtype=torch.long),
        "attention_mask": torch.ones(n, 16, dtype=torch.long),
    }


def _fake_model(n_texts):
    mock = MagicMock()
    mock.config.id2label = ID2LABEL
    mock.side_effect = lambda **kw: SimpleNamespace(
        logits=torch.zeros(kw["input_ids"].shape[0], 5)
    )
    return mock


class TestRunBatchInference:

    def _run(self, rows=None, day=0):
        rows = rows or _SAMPLE_ROWS
        mock_gcs = _mock_gcs_client(download_rows=rows)
        model = _fake_model(len(rows))

        with patch("google.cloud.storage.Client", mock_gcs), \
             patch("news_topic_classifier.modeling.train.download_base_model",
                   return_value="/tmp/fake-model"), \
             patch("news_topic_classifier.modeling.predict.load_model_tokenizer",
                   return_value=(model, _fake_tokenizer)):
            result = _infer_fn(
                gcp_project="proj",
                gcs_model_uri="gs://bucket/model/",
                gcs_input_uri="gs://bucket/inference/day=0/input.parquet",
                gcs_bucket_data="bucket",
                day=day,
                batch_size=4,
                max_seq_length=16,
            )
        return result

    def test_returns_gcs_predictions_uri(self):
        result = self._run(day=2)
        assert result == "gs://bucket/inference/day=2/predictions.parquet"

    def test_output_row_count_matches_input(self):
        written_rows = []
        orig_gcs = _mock_gcs_client(download_rows=_SAMPLE_ROWS)

        def _capture_upload(path):
            written_rows.extend(pq.read_table(path).to_pylist())

        orig_gcs.return_value.bucket.return_value.blob.return_value.upload_from_filename.side_effect = _capture_upload
        model = _fake_model(len(_SAMPLE_ROWS))

        with patch("google.cloud.storage.Client", orig_gcs), \
             patch("news_topic_classifier.modeling.train.download_base_model",
                   return_value="/tmp/fake-model"), \
             patch("news_topic_classifier.modeling.predict.load_model_tokenizer",
                   return_value=(model, _fake_tokenizer)):
            _infer_fn(
                gcp_project="proj",
                gcs_model_uri="gs://bucket/model/",
                gcs_input_uri="gs://bucket/inference/day=0/input.parquet",
                gcs_bucket_data="bucket",
                day=0, batch_size=4, max_seq_length=16,
            )

        assert len(written_rows) == len(_SAMPLE_ROWS)

    def test_output_rows_have_all_required_keys(self):
        written_rows = []
        orig_gcs = _mock_gcs_client(download_rows=_SAMPLE_ROWS)
        orig_gcs.return_value.bucket.return_value.blob.return_value.upload_from_filename.side_effect = (
            lambda path: written_rows.extend(pq.read_table(path).to_pylist())
        )
        model = _fake_model(len(_SAMPLE_ROWS))

        with patch("google.cloud.storage.Client", orig_gcs), \
             patch("news_topic_classifier.modeling.train.download_base_model",
                   return_value="/tmp/fake-model"), \
             patch("news_topic_classifier.modeling.predict.load_model_tokenizer",
                   return_value=(model, _fake_tokenizer)):
            _infer_fn(
                gcp_project="proj",
                gcs_model_uri="gs://bucket/model/",
                gcs_input_uri="gs://bucket/inference/day=0/input.parquet",
                gcs_bucket_data="bucket",
                day=0, batch_size=4, max_seq_length=16,
            )

        required = {
            "prediction_date", "run_timestamp", "title", "body",
            "true_label", "predicted_label", "confidence",
            "score_business", "score_entertainment", "score_politics",
            "score_sport", "score_tech", "day_partition",
        }
        assert required.issubset(set(written_rows[0].keys()))

    def test_day_partition_stored_in_output_rows(self):
        written_rows = []
        orig_gcs = _mock_gcs_client(download_rows=_SAMPLE_ROWS)
        orig_gcs.return_value.bucket.return_value.blob.return_value.upload_from_filename.side_effect = (
            lambda path: written_rows.extend(pq.read_table(path).to_pylist())
        )
        model = _fake_model(len(_SAMPLE_ROWS))

        with patch("google.cloud.storage.Client", orig_gcs), \
             patch("news_topic_classifier.modeling.train.download_base_model",
                   return_value="/tmp/fake-model"), \
             patch("news_topic_classifier.modeling.predict.load_model_tokenizer",
                   return_value=(model, _fake_tokenizer)):
            _infer_fn(
                gcp_project="proj",
                gcs_model_uri="gs://bucket/model/",
                gcs_input_uri="gs://bucket/inference/day=0/input.parquet",
                gcs_bucket_data="bucket",
                day=11, batch_size=4, max_seq_length=16,
            )

        assert all(r["day_partition"] == 11 for r in written_rows)

    def test_calls_download_base_model(self):
        mock_download = MagicMock(return_value="/tmp/fake-model")
        model = _fake_model(len(_SAMPLE_ROWS))

        with patch("google.cloud.storage.Client", _mock_gcs_client()), \
             patch("news_topic_classifier.modeling.train.download_base_model", mock_download), \
             patch("news_topic_classifier.modeling.predict.load_model_tokenizer",
                   return_value=(model, _fake_tokenizer)):
            _infer_fn(
                gcp_project="proj",
                gcs_model_uri="gs://bucket/model/",
                gcs_input_uri="gs://bucket/inference/day=0/input.parquet",
                gcs_bucket_data="bucket",
                day=0, batch_size=4, max_seq_length=16,
            )

        mock_download.assert_called_once_with(
            gcs_model_uri="gs://bucket/model/",
            local_dir="/tmp/bert-bbc-finetuned",
            gcp_project="proj",
        )

    def test_confidence_is_float_between_0_and_1(self):
        written_rows = []
        orig_gcs = _mock_gcs_client(download_rows=_SAMPLE_ROWS)
        orig_gcs.return_value.bucket.return_value.blob.return_value.upload_from_filename.side_effect = (
            lambda path: written_rows.extend(pq.read_table(path).to_pylist())
        )
        model = _fake_model(len(_SAMPLE_ROWS))

        with patch("google.cloud.storage.Client", orig_gcs), \
             patch("news_topic_classifier.modeling.train.download_base_model",
                   return_value="/tmp/fake-model"), \
             patch("news_topic_classifier.modeling.predict.load_model_tokenizer",
                   return_value=(model, _fake_tokenizer)):
            _infer_fn(
                gcp_project="proj",
                gcs_model_uri="gs://bucket/model/",
                gcs_input_uri="gs://bucket/inference/day=0/input.parquet",
                gcs_bucket_data="bucket",
                day=0, batch_size=4, max_seq_length=16,
            )

        for row in written_rows:
            assert 0.0 <= row["confidence"] <= 1.0

    def test_score_columns_sum_to_one(self):
        written_rows = []
        orig_gcs = _mock_gcs_client(download_rows=_SAMPLE_ROWS[:1])
        orig_gcs.return_value.bucket.return_value.blob.return_value.upload_from_filename.side_effect = (
            lambda path: written_rows.extend(pq.read_table(path).to_pylist())
        )
        model = _fake_model(1)

        with patch("google.cloud.storage.Client", orig_gcs), \
             patch("news_topic_classifier.modeling.train.download_base_model",
                   return_value="/tmp/fake-model"), \
             patch("news_topic_classifier.modeling.predict.load_model_tokenizer",
                   return_value=(model, _fake_tokenizer)):
            _infer_fn(
                gcp_project="proj",
                gcs_model_uri="gs://bucket/model/",
                gcs_input_uri="gs://bucket/inference/day=0/input.parquet",
                gcs_bucket_data="bucket",
                day=0, batch_size=4, max_seq_length=16,
            )

        score_cols = ["score_business", "score_entertainment", "score_politics",
                      "score_sport", "score_tech"]
        for row in written_rows:
            total = sum(row[c] for c in score_cols)
            assert abs(total - 1.0) < 1e-4


# ═══════════════════════════════════════════════════════════════════════════════
# write_inference_results_component
# ═══════════════════════════════════════════════════════════════════════════════

class TestWriteInferenceResults:

    def _make_gcs_with_preds(self, rows=None):
        rows = rows or _PRED_ROWS
        mock_gcs = MagicMock()
        mock_gcs.return_value.bucket.return_value.blob.return_value.download_to_filename.side_effect = (
            lambda path: pq.write_table(pa.Table.from_pylist(rows), path)
        )
        return mock_gcs

    def _run(self, rows=None, bq_errors=None):
        rows = rows or _PRED_ROWS
        mock_gcs = self._make_gcs_with_preds(rows)
        mock_bq = MagicMock()
        mock_bq.return_value.insert_rows_json.return_value = bq_errors or []

        with patch("google.cloud.storage.Client", mock_gcs), \
             patch("google.cloud.bigquery.Client", mock_bq), \
             patch("google.cloud.bigquery.Table", MagicMock()), \
             patch("google.cloud.bigquery.TimePartitioning", MagicMock()), \
             patch("google.cloud.bigquery.TimePartitioningType", MagicMock()):
            return mock_bq, _write_fn(
                gcp_project="proj",
                bq_dataset="ds",
                gcs_predictions_uri="gs://bucket/inference/day=0/predictions.parquet",
            )

    def test_returns_row_count(self):
        _, result = self._run()
        assert result == len(_PRED_ROWS)

    def test_creates_table_with_exists_ok(self):
        mock_bq, _ = self._run()
        mock_bq.return_value.create_table.assert_called_once()
        _, kwargs = mock_bq.return_value.create_table.call_args
        assert kwargs.get("exists_ok") is True

    def test_insert_rows_called_with_all_rows(self):
        mock_bq, _ = self._run()
        call_args = mock_bq.return_value.insert_rows_json.call_args
        table_ref = call_args[0][0]
        rows_arg  = call_args[0][1]
        assert table_ref == "proj.ds.news_topic_classifier_predictions"
        assert len(rows_arg) == len(_PRED_ROWS)

    def test_raises_on_bq_insert_errors(self):
        with pytest.raises(RuntimeError, match="BigQuery insert errors"):
            self._run(bq_errors=[{"index": 0, "errors": [{"reason": "invalid"}]}])

    def test_custom_predictions_table_name(self):
        rows = _PRED_ROWS
        mock_gcs = self._make_gcs_with_preds(rows)
        mock_bq = MagicMock()
        mock_bq.return_value.insert_rows_json.return_value = []

        with patch("google.cloud.storage.Client", mock_gcs), \
             patch("google.cloud.bigquery.Client", mock_bq), \
             patch("google.cloud.bigquery.Table", MagicMock()), \
             patch("google.cloud.bigquery.TimePartitioning", MagicMock()), \
             patch("google.cloud.bigquery.TimePartitioningType", MagicMock()):
            _write_fn(
                gcp_project="proj",
                bq_dataset="ds",
                gcs_predictions_uri="gs://bucket/inference/day=0/predictions.parquet",
                predictions_table="custom_predictions",
            )

        call_args = mock_bq.return_value.insert_rows_json.call_args
        assert "custom_predictions" in call_args[0][0]
