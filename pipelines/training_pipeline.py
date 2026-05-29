from __future__ import annotations

from kfp import compiler, dsl

from pipelines.components.evaluate import evaluate_component
from pipelines.components.extract import extract_component
from pipelines.components.preprocess import preprocess_component
from pipelines.components.predict import predict_component
from pipelines.components.train import train_component


@dsl.pipeline(name="bbc-news-training-pipeline")
def training_pipeline(
    gcp_project: str,
    gcs_bucket_data: str,
    gcs_bucket_artifacts: str,
    bq_dataset: str,
    source_table: str,
    text_col: str,
    label_col: str,
    title_col: str,
    mlflow_tracking_uri: str,
    project_name: str,
    environment_name: str,
    bert_base_model: str,
    num_labels: int,
    max_seq_length: int,
    epochs: int,
    batch_size: int,
    lr: float,
    warmup_steps: int,
    weight_decay: float,
    early_stopping_patience: int,
    val_split: float = 0.1,
    test_split: float = 0.1,
    input_table_size: int = None,
    sample_size: int = None,
) -> None:
    """
    End-to-end BBC News BERT training pipeline.

    Steps
    -----
    1. extract    — BQ -> GCS raw Parquet
    2. preprocess — GCS raw -> GCS train/val/test splits
    3. train      — fine-tune BERT, upload model to GCS
    4. predict    — run inference on test split, save predictions to GCS
    5. evaluate   — generate plots + Word report, upload to GCS
    """

    # Step 1 — Extract 
    extract_task = extract_component(
        gcp_project=gcp_project,
        bq_dataset=bq_dataset,
        source_table=source_table,
        text_col=text_col,
        label_col=label_col,
        title_col=title_col,
        gcs_bucket_data=gcs_bucket_data,
        input_table_size=input_table_size,
        sample_size=sample_size,
    )

    # Step 2 — Preprocess
    preprocess_task = preprocess_component(
        gcp_project=gcp_project,
        bq_dataset=bq_dataset,
        gcs_bucket_data=gcs_bucket_data,
        gcs_raw_uri=extract_task.output,
        val_split=val_split,
        test_split=test_split,
    )

    # Step 3 — Train
    train_task = train_component(
        gcp_project=gcp_project,
        gcs_bucket_artifacts=gcs_bucket_artifacts,
        gcs_splits_dir=preprocess_task.output,
        mlflow_tracking_uri=mlflow_tracking_uri,
        project_name=project_name,
        bert_base_model=bert_base_model,
        num_labels=num_labels,
        max_seq_length=max_seq_length,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        early_stopping_patience=early_stopping_patience,
    )

    # Step 4 — Predict
    predict_task = predict_component(
        gcp_project=gcp_project,
        gcs_bucket_data=gcs_bucket_data,
        gcs_splits_dir=preprocess_task.output,
        gcs_model_uri=train_task.outputs["gcs_model_uri"],
        mlflow_tracking_uri=mlflow_tracking_uri,
        project_name=project_name,
        max_seq_length=max_seq_length,
        batch_size=batch_size,
    )

    # Step 5 — Evaluate
    evaluate_component(
        gcp_project=gcp_project,
        gcs_bucket_artifacts=gcs_bucket_artifacts,
        gcs_predictions_uri=predict_task.output,
        mlflow_run_id=train_task.outputs["mlflow_run_id"],
        mlflow_tracking_uri=mlflow_tracking_uri,
        project_name=project_name,
        environment_name=environment_name,
        bert_base_model=bert_base_model,
        num_labels=num_labels,
        max_seq_length=max_seq_length,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        early_stopping_patience=early_stopping_patience,
    )


if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=training_pipeline,
        package_path="pipelines/compiled/training_pipeline.yaml",
    )
    print("Pipeline compiled -> pipelines/compiled/training_pipeline.yaml")
