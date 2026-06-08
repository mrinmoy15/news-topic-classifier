"""Unit tests for the KFP pipeline definitions.

Verifies that each pipeline correctly wires its component steps —
the right components are called with the right arguments and outputs
are chained in the correct order.

No GCP or Vertex AI credentials are required; all components are mocked.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch


def _unwrap(pipeline_fn):
    """Return the raw Python function underneath @dsl.pipeline.

    KFP 2.x wraps pipelines as GraphComponent objects with a .pipeline_func
    attribute that holds the original Python function.
    """
    return pipeline_fn.pipeline_func


# ═══════════════════════════════════════════════════════════════════════════════
# inference_pipeline
# ═══════════════════════════════════════════════════════════════════════════════

class TestInferencePipeline:

    _KWARGS = dict(
        gcp_project="proj",
        gcs_bucket_data="data-bucket",
        gcs_model_uri="gs://artifacts/models/bert/",
        bq_dataset="MY_DATASET",
    )

    def _run(self, **overrides):
        fetch_task  = MagicMock()
        infer_task  = MagicMock()
        fetch_task.output = "gs://data-bucket/inference/samples/input.parquet"
        infer_task.output = "gs://data-bucket/inference/samples/predictions.parquet"

        with patch("pipelines.inference_pipeline.fetch_inference_data_component") as mock_fetch, \
             patch("pipelines.inference_pipeline.run_batch_inference_component") as mock_infer, \
             patch("pipelines.inference_pipeline.write_inference_results_component") as mock_write:

            mock_fetch.return_value = fetch_task
            mock_infer.return_value = infer_task

            from pipelines.inference_pipeline import inference_pipeline
            kwargs = {**self._KWARGS, **overrides}
            _unwrap(inference_pipeline)(**kwargs)

            return mock_fetch, mock_infer, mock_write, fetch_task, infer_task

    def test_fetch_component_called(self):
        mock_fetch, *_ = self._run()
        mock_fetch.assert_called_once()

    def test_infer_component_called(self):
        _, mock_infer, *_ = self._run()
        mock_infer.assert_called_once()

    def test_write_component_called(self):
        _, _, mock_write, *_ = self._run()
        mock_write.assert_called_once()

    def test_fetch_receives_gcp_project(self):
        mock_fetch, *_ = self._run()
        _, kwargs = mock_fetch.call_args
        assert kwargs["gcp_project"] == "proj"

    def test_fetch_output_passed_to_infer(self):
        mock_fetch, mock_infer, _, fetch_task, _ = self._run()
        _, infer_kwargs = mock_infer.call_args
        assert infer_kwargs["gcs_input_uri"] == fetch_task.output

    def test_infer_output_passed_to_write(self):
        _, mock_infer, mock_write, _, infer_task = self._run()
        _, write_kwargs = mock_write.call_args
        assert write_kwargs["gcs_predictions_uri"] == infer_task.output

    def test_custom_predictions_table_forwarded(self):
        _, _, mock_write, *_ = self._run(predictions_table="custom_preds")
        _, write_kw = mock_write.call_args
        assert write_kw["predictions_table"] == "custom_preds"

    def test_batch_size_forwarded_to_infer(self):
        _, mock_infer, *_ = self._run(batch_size=64)
        _, infer_kw = mock_infer.call_args
        assert infer_kw["batch_size"] == 64

    def test_max_seq_length_forwarded_to_infer(self):
        _, mock_infer, *_ = self._run(max_seq_length=256)
        _, infer_kw = mock_infer.call_args
        assert infer_kw["max_seq_length"] == 256


# ═══════════════════════════════════════════════════════════════════════════════
# training_pipeline
# ═══════════════════════════════════════════════════════════════════════════════

class TestTrainingPipeline:

    _KWARGS = dict(
        gcp_project="proj",
        gcp_region="us-central1",
        gcs_bucket_data="data-bucket",
        gcs_bucket_artifacts="artifacts-bucket",
        bq_dataset="MY_DATASET",
        source_table="bigquery-public-data.bbc_news.fulltext",
        text_col="body",
        label_col="category",
        title_col="title",
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

    def _run(self, **overrides):
        extract_task   = MagicMock()
        preprocess_task = MagicMock()
        train_task     = MagicMock()
        predict_task   = MagicMock()
        evaluate_task  = MagicMock()

        extract_task.output           = "gs://data/raw.parquet"
        preprocess_task.output        = "gs://data/processed/"
        train_task.outputs            = {"gcs_model_uri": "gs://artifacts/model/", "mlflow_run_id": "run123"}
        predict_task.output           = "gs://data/predictions.parquet"

        # register_model_component returns a task with .after()
        register_task = MagicMock()
        register_task.after.return_value = register_task

        # train_component returns a task with .set_cpu_limit().set_memory_limit()
        train_chain = MagicMock()
        train_chain.set_cpu_limit.return_value = train_chain
        train_chain.set_memory_limit.return_value = train_task

        with patch("pipelines.training_pipeline.extract_component")       as mock_extract, \
             patch("pipelines.training_pipeline.preprocess_component")    as mock_preprocess, \
             patch("pipelines.training_pipeline.train_component")         as mock_train, \
             patch("pipelines.training_pipeline.predict_component")       as mock_predict, \
             patch("pipelines.training_pipeline.evaluate_component")      as mock_evaluate, \
             patch("pipelines.training_pipeline.register_model_component") as mock_register:

            mock_extract.return_value    = extract_task
            mock_preprocess.return_value = preprocess_task
            mock_train.return_value      = train_chain
            mock_predict.return_value    = predict_task
            mock_evaluate.return_value   = evaluate_task
            mock_register.return_value   = register_task

            from pipelines.training_pipeline import training_pipeline
            kwargs = {**self._KWARGS, **overrides}
            _unwrap(training_pipeline)(**kwargs)

            return (mock_extract, mock_preprocess, mock_train,
                    mock_predict, mock_evaluate, mock_register,
                    extract_task, preprocess_task, train_task,
                    predict_task, evaluate_task)

    def test_all_six_components_called(self):
        mocks = self._run()
        for mock in mocks[:6]:
            mock.assert_called_once()

    def test_extract_receives_gcp_project(self):
        mock_extract, *_ = self._run()
        _, kw = mock_extract.call_args
        assert kw["gcp_project"] == "proj"

    def test_extract_output_passed_to_preprocess(self):
        mock_extract, mock_preprocess, *rest = self._run()
        _, kw = mock_preprocess.call_args
        extract_task = rest[6 - 2]  # extract_task is index 6
        # extract_task.output is passed as gcs_raw_uri
        assert kw["gcs_raw_uri"] == mock_extract.return_value.output

    def test_preprocess_output_passed_to_train(self):
        _, mock_preprocess, mock_train, *_ = self._run()
        _, kw = mock_train.call_args
        assert kw["gcs_splits_dir"] == mock_preprocess.return_value.output

    def test_register_receives_gcp_region(self):
        *_, mock_register, _e, _pp, _t, _p, _ev = self._run()
        _, kw = mock_register.call_args
        assert kw["gcp_region"] == "us-central1"

    def test_cpu_and_memory_limits_applied_to_train(self):
        _, _, mock_train, *_ = self._run()
        train_chain = mock_train.return_value
        train_chain.set_cpu_limit.assert_called_once_with("8")
        train_chain.set_memory_limit.assert_called_once_with("32G")
