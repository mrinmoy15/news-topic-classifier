"""Unit tests for the pipeline submission entrypoints.

Covers _build_parameter_values and _compile in both run_pipeline.py
and run_inference_pipeline.py.  No real Vertex AI or KFP compilation
is performed — both aiplatform and kfp.compiler are mocked.
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch


def _make_cfg(*, inference: bool = False):
    """Return a MagicMock that mimics the Hydra config structure."""
    cfg = MagicMock()
    cfg.environment.gcp_project          = "cs-cdwp-data-dev2188"
    cfg.environment.gcp_region           = "us-central1"
    cfg.environment.gcs_bucket_data      = "dev-data-bucket"
    cfg.environment.gcs_bucket_artifacts = "dev-artifacts-bucket"
    cfg.environment.bq_dataset           = "DATA_SCNCE_DEV_DATA"
    cfg.environment.artifact_registry_repo = "news-topic-classifier"
    cfg.environment.name                 = "dev"
    cfg.environment.vertex_ai_sa         = "sa@project.iam.gserviceaccount.com"
    cfg.environment.mlflow.tracking_uri  = "http://mlflow:5000"
    cfg.data.bq_source_table             = "bigquery-public-data.bbc_news.fulltext"
    cfg.data.bq_text_column              = "body"
    cfg.data.bq_label_column             = "category"
    cfg.data.bq_title_column             = "title"
    cfg.project.name                     = "bbc-news-classifier"
    cfg.model.bert_base_model            = "bert-base-uncased"
    cfg.model.num_labels                 = 5
    cfg.model.max_seq_length             = 512
    cfg.training.epochs                  = 3
    cfg.training.batch_size              = 16
    cfg.training.lr                      = 2e-5
    cfg.training.warmup_steps            = 100
    cfg.training.weight_decay            = 0.01
    cfg.training.early_stopping_patience = 3
    cfg.training.val_split               = 0.1
    cfg.training.test_split              = 0.1
    return cfg


# ═══════════════════════════════════════════════════════════════════════════════
# run_pipeline.py  (training)
# ═══════════════════════════════════════════════════════════════════════════════

class TestRunTrainingPipeline:

    def test_build_parameter_values_gcp_project(self):
        from pipelines.run_pipeline import _build_parameter_values
        params = _build_parameter_values(_make_cfg())
        assert params["gcp_project"] == "cs-cdwp-data-dev2188"

    def test_build_parameter_values_all_keys(self):
        from pipelines.run_pipeline import _build_parameter_values
        params = _build_parameter_values(_make_cfg())
        required = {
            "gcp_project", "gcp_region", "gcs_bucket_data", "gcs_bucket_artifacts",
            "bq_dataset", "source_table", "text_col", "label_col", "title_col",
            "mlflow_tracking_uri", "project_name", "environment_name",
            "bert_base_model", "num_labels", "max_seq_length",
            "epochs", "batch_size", "lr", "warmup_steps",
            "weight_decay", "early_stopping_patience", "val_split", "test_split",
        }
        assert required.issubset(params.keys())

    def test_build_parameter_values_training_hyperparams(self):
        from pipelines.run_pipeline import _build_parameter_values
        params = _build_parameter_values(_make_cfg())
        assert params["epochs"] == 3
        assert params["batch_size"] == 16
        assert params["lr"] == 2e-5

    def test_compile_calls_kfp_compiler(self):
        from pipelines.run_pipeline import _compile
        mock_compiler_inst = MagicMock()

        with patch("kfp.compiler.Compiler", return_value=mock_compiler_inst), \
             patch("pathlib.Path.mkdir"):
            _compile(_make_cfg())

        mock_compiler_inst.compile.assert_called_once()

    def test_compile_sets_trainer_image_env(self):
        from pipelines.run_pipeline import _compile
        with patch("kfp.compiler.Compiler", return_value=MagicMock()), \
             patch("pathlib.Path.mkdir"):
            _compile(_make_cfg())

        assert "TRAINER_IMAGE" in os.environ
        assert "cs-cdwp-data-dev2188" in os.environ["TRAINER_IMAGE"]
        assert "news-topic-classifier" in os.environ["TRAINER_IMAGE"]

    def test_compile_passes_compiled_path(self):
        from pipelines.run_pipeline import _compile, COMPILED_PATH
        mock_compiler_inst = MagicMock()

        with patch("kfp.compiler.Compiler", return_value=mock_compiler_inst), \
             patch("pathlib.Path.mkdir"):
            _compile(_make_cfg())

        _, kw = mock_compiler_inst.compile.call_args
        assert kw["package_path"] == COMPILED_PATH


# ═══════════════════════════════════════════════════════════════════════════════
# run_inference_pipeline.py
# ═══════════════════════════════════════════════════════════════════════════════

class TestRunInferencePipeline:

    def test_build_parameter_values_gcp_project(self):
        from pipelines.run_inference_pipeline import _build_parameter_values
        params = _build_parameter_values(_make_cfg())
        assert params["gcp_project"] == "cs-cdwp-data-dev2188"

    def test_build_parameter_values_gcs_model_uri_format(self):
        from pipelines.run_inference_pipeline import _build_parameter_values
        params = _build_parameter_values(_make_cfg())
        assert params["gcs_model_uri"].startswith("gs://")
        assert "bert-bbc-finetuned" in params["gcs_model_uri"]

    def test_build_parameter_values_all_keys(self):
        from pipelines.run_inference_pipeline import _build_parameter_values
        params = _build_parameter_values(_make_cfg())
        required = {
            "gcp_project", "gcs_bucket_data", "gcs_model_uri", "bq_dataset",
            "source_table", "predictions_table", "batch_size", "max_seq_length",
        }
        assert required.issubset(params.keys())
        assert "day" not in params

    def test_compile_calls_kfp_compiler(self):
        from pipelines.run_inference_pipeline import _compile
        mock_compiler_inst = MagicMock()

        with patch("kfp.compiler.Compiler", return_value=mock_compiler_inst), \
             patch("pathlib.Path.mkdir"):
            _compile(_make_cfg())

        mock_compiler_inst.compile.assert_called_once()

    def test_compile_sets_trainer_image_env(self):
        from pipelines.run_inference_pipeline import _compile
        with patch("kfp.compiler.Compiler", return_value=MagicMock()), \
             patch("pathlib.Path.mkdir"):
            _compile(_make_cfg())

        assert "TRAINER_IMAGE" in os.environ
        assert "cs-cdwp-data-dev2188" in os.environ["TRAINER_IMAGE"]

    def test_compile_passes_compiled_path(self):
        from pipelines.run_inference_pipeline import _compile, COMPILED_PATH
        mock_compiler_inst = MagicMock()

        with patch("kfp.compiler.Compiler", return_value=mock_compiler_inst), \
             patch("pathlib.Path.mkdir"):
            _compile(_make_cfg())

        _, kw = mock_compiler_inst.compile.call_args
        assert kw["package_path"] == COMPILED_PATH
