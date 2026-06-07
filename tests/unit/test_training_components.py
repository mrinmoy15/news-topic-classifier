"""Unit tests for the five training-pipeline KFP components.

All GCP I/O and heavy ML dependencies (BERT model, tokenizer, training loop)
are mocked.  No real credentials or GPU are required.
"""
from __future__ import annotations

from collections import namedtuple
from unittest.mock import MagicMock, patch


# ═══════════════════════════════════════════════════════════════════════════════
# extract_component
# ═══════════════════════════════════════════════════════════════════════════════

class TestExtractComponent:

    def _run(self, **overrides):
        from pipelines.components.extract import extract_component
        fn = extract_component.python_func

        kwargs = dict(
            gcp_project="proj", bq_dataset="ds",
            source_table="my-proj.ds.tbl",
            text_col="body", label_col="category", title_col="title",
            gcs_bucket_data="bucket",
        )
        kwargs.update(overrides)

        mock_extract = MagicMock(return_value="gs://bucket/data/raw/bbc_news.parquet")
        with patch("news_topic_classifier.dataset.extract_from_bigquery", mock_extract):
            return fn(**kwargs), mock_extract

    def test_returns_gcs_uri(self):
        result, _ = self._run()
        assert result.startswith("gs://")

    def test_calls_extract_from_bigquery(self):
        _, mock_fn = self._run()
        mock_fn.assert_called_once()

    def test_passes_gcp_project(self):
        _, mock_fn = self._run()
        _, kw = mock_fn.call_args
        assert kw["gcp_project"] == "proj"

    def test_passes_source_table(self):
        _, mock_fn = self._run()
        _, kw = mock_fn.call_args
        assert kw["source_table"] == "my-proj.ds.tbl"

    def test_zero_sample_size_converted_to_none(self):
        _, mock_fn = self._run(sample_size=0)
        _, kw = mock_fn.call_args
        assert kw["sample_size"] is None

    def test_nonzero_sample_size_passed_through(self):
        _, mock_fn = self._run(sample_size=500)
        _, kw = mock_fn.call_args
        assert kw["sample_size"] == 500


# ═══════════════════════════════════════════════════════════════════════════════
# preprocess_component
# ═══════════════════════════════════════════════════════════════════════════════

class TestPreprocessComponent:

    def _run(self, **overrides):
        from pipelines.components.preprocess import preprocess_component
        fn = preprocess_component.python_func

        kwargs = dict(
            gcp_project="proj", bq_dataset="ds",
            gcs_bucket_data="bucket",
            gcs_raw_uri="gs://bucket/data/raw/bbc_news.parquet",
        )
        kwargs.update(overrides)

        mock_splits = {
            "train": "gs://bucket/data/processed/2026-06-06T10-00-00/train.parquet",
            "val":   "gs://bucket/data/processed/2026-06-06T10-00-00/val.parquet",
            "test":  "gs://bucket/data/processed/2026-06-06T10-00-00/test.parquet",
        }
        mock_preprocess = MagicMock(return_value=mock_splits)
        with patch("news_topic_classifier.features.preprocess_and_split", mock_preprocess):
            return fn(**kwargs), mock_preprocess

    def test_returns_gcs_directory_uri(self):
        result, _ = self._run()
        assert result.startswith("gs://")
        assert result.endswith("/")

    def test_calls_preprocess_and_split(self):
        _, mock_fn = self._run()
        mock_fn.assert_called_once()

    def test_passes_gcp_project(self):
        _, mock_fn = self._run()
        _, kw = mock_fn.call_args
        assert kw["gcp_project"] == "proj"

    def test_output_is_common_directory(self):
        result, _ = self._run()
        # All three splits share the same timestamped directory
        assert "2026-06-06T10-00-00" in result

    def test_custom_val_split_forwarded(self):
        _, mock_fn = self._run(val_split=0.2)
        _, kw = mock_fn.call_args
        assert kw["val_split"] == 0.2


# ═══════════════════════════════════════════════════════════════════════════════
# evaluate_component
# ═══════════════════════════════════════════════════════════════════════════════

class TestEvaluateComponent:

    def _run(self):
        from pipelines.components.evaluate import evaluate_component
        fn = evaluate_component.python_func

        mock_report = MagicMock(return_value="gs://artifacts/outputs/2026-06-06/")
        with patch("news_topic_classifier.modeling.report.generate_report", mock_report):
            result = fn(
                gcp_project="proj",
                gcs_bucket_artifacts="artifacts-bucket",
                gcs_predictions_uri="gs://bucket/data/predictions.parquet",
                mlflow_run_id="run123",
                mlflow_tracking_uri="http://mlflow:5000",
                project_name="bbc-news",
                environment_name="dev",
                bert_base_model="bert-base-uncased",
                num_labels=5,
                max_seq_length=512,
                epochs=3,
                batch_size=16,
                lr=2e-5,
                warmup_steps=100,
                weight_decay=0.01,
                early_stopping_patience=3,
            )
        return result, mock_report

    def test_returns_gcs_outputs_uri(self):
        result, _ = self._run()
        assert result.startswith("gs://")

    def test_calls_generate_report(self):
        _, mock_fn = self._run()
        mock_fn.assert_called_once()

    def test_passes_run_id(self):
        _, mock_fn = self._run()
        _, kw = mock_fn.call_args
        assert kw["run_id"] == "run123"

    def test_passes_predictions_uri(self):
        _, mock_fn = self._run()
        _, kw = mock_fn.call_args
        assert kw["gcs_predictions_uri"] == "gs://bucket/data/predictions.parquet"

    def test_config_contains_training_params(self):
        _, mock_fn = self._run()
        _, kw = mock_fn.call_args
        cfg = kw["cfg"]
        assert cfg.training.epochs == 3
        assert cfg.training.batch_size == 16
        assert cfg.model.bert_base_model == "bert-base-uncased"


# ═══════════════════════════════════════════════════════════════════════════════
# predict_component
# ═══════════════════════════════════════════════════════════════════════════════

class TestPredictComponent:

    def _run(self):
        from pipelines.components.predict import predict_component
        fn = predict_component.python_func

        mock_tokenizer = MagicMock()
        mock_dataset   = MagicMock()
        mock_loader    = MagicMock()
        gcs_preds_uri  = "gs://bucket/data/predictions/predictions.parquet"

        with patch("news_topic_classifier.modeling.train.download_base_model",
                   return_value="/tmp/finetuned-model"), \
             patch("news_topic_classifier.modeling.train.download_splits",
                   return_value=("/tmp/train.parquet", "/tmp/val.parquet", "/tmp/test.parquet")), \
             patch("transformers.BertTokenizerFast.from_pretrained", return_value=mock_tokenizer), \
             patch("news_topic_classifier.dataset.BBCNewsDataset", return_value=mock_dataset), \
             patch("torch.utils.data.DataLoader", return_value=mock_loader), \
             patch("news_topic_classifier.modeling.predict.predict",
                   return_value=(MagicMock(), gcs_preds_uri)):

            result = fn(
                gcp_project="proj",
                gcs_bucket_data="data-bucket",
                gcs_splits_dir="gs://bucket/data/processed/2026-06-06/",
                gcs_model_uri="gs://artifacts/models/bert/",
                mlflow_tracking_uri="http://mlflow:5000",
                project_name="bbc-news",
                max_seq_length=512,
                batch_size=16,
            )
        return result

    def test_returns_gcs_predictions_uri(self):
        result = self._run()
        assert result.startswith("gs://")
        assert "predictions" in result

    def test_download_base_model_called(self):
        with patch("news_topic_classifier.modeling.train.download_base_model",
                   return_value="/tmp/finetuned-model") as mock_dl, \
             patch("news_topic_classifier.modeling.train.download_splits",
                   return_value=("/tmp/train.parquet", "/tmp/val.parquet", "/tmp/test.parquet")), \
             patch("transformers.BertTokenizerFast.from_pretrained", return_value=MagicMock()), \
             patch("news_topic_classifier.dataset.BBCNewsDataset", return_value=MagicMock()), \
             patch("torch.utils.data.DataLoader", return_value=MagicMock()), \
             patch("news_topic_classifier.modeling.predict.predict",
                   return_value=(MagicMock(), "gs://bucket/preds.parquet")):
            from pipelines.components.predict import predict_component
            predict_component.python_func(
                gcp_project="proj", gcs_bucket_data="bucket",
                gcs_splits_dir="gs://bucket/processed/",
                gcs_model_uri="gs://artifacts/model/",
                mlflow_tracking_uri="http://mlflow:5000",
                project_name="bbc", max_seq_length=512, batch_size=16,
            )
        mock_dl.assert_called_once()

    def test_download_splits_called(self):
        with patch("news_topic_classifier.modeling.train.download_base_model",
                   return_value="/tmp/model"), \
             patch("news_topic_classifier.modeling.train.download_splits",
                   return_value=("/tmp/train.parquet", "/tmp/val.parquet", "/tmp/test.parquet")) as mock_splits, \
             patch("transformers.BertTokenizerFast.from_pretrained", return_value=MagicMock()), \
             patch("news_topic_classifier.dataset.BBCNewsDataset", return_value=MagicMock()), \
             patch("torch.utils.data.DataLoader", return_value=MagicMock()), \
             patch("news_topic_classifier.modeling.predict.predict",
                   return_value=(MagicMock(), "gs://bucket/preds.parquet")):
            from pipelines.components.predict import predict_component
            predict_component.python_func(
                gcp_project="proj", gcs_bucket_data="bucket",
                gcs_splits_dir="gs://bucket/processed/",
                gcs_model_uri="gs://artifacts/model/",
                mlflow_tracking_uri="http://mlflow:5000",
                project_name="bbc", max_seq_length=512, batch_size=16,
            )
        mock_splits.assert_called_once()


# ═══════════════════════════════════════════════════════════════════════════════
# train_component
# ═══════════════════════════════════════════════════════════════════════════════

class TestTrainComponent:

    _KWARGS = dict(
        gcp_project="proj",
        gcs_bucket_artifacts="artifacts-bucket",
        gcs_splits_dir="gs://bucket/data/processed/2026-06-06/",
        mlflow_tracking_uri="http://mlflow:5000",
        project_name="bbc-news",
        bert_base_model="bert-base-uncased",
        num_labels=5,
        max_seq_length=512,
        epochs=3,
        batch_size=16,
        lr=2e-5,
        warmup_steps=100,
        weight_decay=0.01,
        early_stopping_patience=3,
    )

    def _run(self):
        from pipelines.components.train import train_component
        fn = train_component.python_func

        TrainOutputs = namedtuple("TrainOutputs", ["gcs_model_uri", "mlflow_run_id"])
        mock_model    = MagicMock()
        mock_tokenizer = MagicMock()
        mock_loader   = MagicMock()

        with patch("news_topic_classifier.modeling.train.download_base_model",
                   return_value="/tmp/base-model"), \
             patch("news_topic_classifier.modeling.train.download_splits",
                   return_value=("/tmp/train.parquet", "/tmp/val.parquet", "/tmp/test.parquet")), \
             patch("transformers.BertTokenizerFast.from_pretrained", return_value=mock_tokenizer), \
             patch("news_topic_classifier.modeling.bert_classifier.build_model",
                   return_value=mock_model), \
             patch("news_topic_classifier.modeling.train.build_dataloaders",
                   return_value=(mock_loader, mock_loader, mock_loader)), \
             patch("news_topic_classifier.modeling.train.train",
                   return_value=(MagicMock(), "gs://artifacts/model/", "run-abc")):
            mock_model.to.return_value = mock_model
            result = fn(**self._KWARGS)
        return result

    def test_returns_namedtuple_with_gcs_model_uri(self):
        result = self._run()
        assert hasattr(result, "gcs_model_uri")
        assert result.gcs_model_uri.startswith("gs://")

    def test_returns_namedtuple_with_mlflow_run_id(self):
        result = self._run()
        assert hasattr(result, "mlflow_run_id")
        assert result.mlflow_run_id == "run-abc"

    def test_download_base_model_called(self):
        from pipelines.components.train import train_component
        fn = train_component.python_func
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model

        with patch("news_topic_classifier.modeling.train.download_base_model",
                   return_value="/tmp/base-model") as mock_dl, \
             patch("news_topic_classifier.modeling.train.download_splits",
                   return_value=("/tmp/train.parquet", "/tmp/val.parquet", "/tmp/test.parquet")), \
             patch("transformers.BertTokenizerFast.from_pretrained", return_value=MagicMock()), \
             patch("news_topic_classifier.modeling.bert_classifier.build_model",
                   return_value=mock_model), \
             patch("news_topic_classifier.modeling.train.build_dataloaders",
                   return_value=(MagicMock(), MagicMock(), MagicMock())), \
             patch("news_topic_classifier.modeling.train.train",
                   return_value=(MagicMock(), "gs://m/", "rid")):
            fn(**self._KWARGS)

        mock_dl.assert_called_once()

    def test_train_function_called(self):
        from pipelines.components.train import train_component
        fn = train_component.python_func
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model

        with patch("news_topic_classifier.modeling.train.download_base_model",
                   return_value="/tmp/base-model"), \
             patch("news_topic_classifier.modeling.train.download_splits",
                   return_value=("/tmp/train.parquet", "/tmp/val.parquet", "/tmp/test.parquet")), \
             patch("transformers.BertTokenizerFast.from_pretrained", return_value=MagicMock()), \
             patch("news_topic_classifier.modeling.bert_classifier.build_model",
                   return_value=mock_model), \
             patch("news_topic_classifier.modeling.train.build_dataloaders",
                   return_value=(MagicMock(), MagicMock(), MagicMock())), \
             patch("news_topic_classifier.modeling.train.train",
                   return_value=(MagicMock(), "gs://m/", "rid")) as mock_train:
            fn(**self._KWARGS)

        mock_train.assert_called_once()
