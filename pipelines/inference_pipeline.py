from kfp import compiler, dsl

from pipelines.components.fetch_inference_data import fetch_inference_data_component
from pipelines.components.run_batch_inference import run_batch_inference_component
from pipelines.components.write_inference_results import write_inference_results_component


@dsl.pipeline(name="bbc-news-inference-pipeline")
def inference_pipeline(
    gcp_project: str,
    gcs_bucket_data: str,
    gcs_model_uri: str,
    bq_dataset: str,
    source_table: str = "bigquery-public-data.bbc_news.fulltext",
    predictions_table: str = "news_topic_classifier_predictions",
    day: int = -1,
    batch_size: int = 32,
    max_seq_length: int = 512,
) -> None:
    """
    Daily BBC News batch inference pipeline.

    Steps
    -----
    1. fetch_data     — BQ partition → GCS Parquet (~74 articles)
    2. run_inference  — GCS Parquet + BERT model → predictions GCS Parquet
    3. write_results  — predictions GCS Parquet → BigQuery table

    The ``day`` parameter defaults to -1, which causes each component to
    auto-compute (UTC_day - 1) % 30 at runtime.  Pass an explicit value
    (0-29) to reprocess a specific partition.
    """

    # Step 1 — Fetch today's data from BigQuery
    fetch_task = fetch_inference_data_component(
        gcp_project=gcp_project,
        bq_dataset=bq_dataset,
        gcs_bucket_data=gcs_bucket_data,
        source_table=source_table,
        day=day,
    )

    # Step 2 — Run BERT inference
    infer_task = run_batch_inference_component(
        gcp_project=gcp_project,
        gcs_model_uri=gcs_model_uri,
        gcs_input_uri=fetch_task.output,
        gcs_bucket_data=gcs_bucket_data,
        day=day,
        batch_size=batch_size,
        max_seq_length=max_seq_length,
    )

    # Step 3 — Write predictions to BigQuery
    write_inference_results_component(
        gcp_project=gcp_project,
        bq_dataset=bq_dataset,
        gcs_predictions_uri=infer_task.output,
        predictions_table=predictions_table,
    )


if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=inference_pipeline,
        package_path="pipelines/compiled/inference_pipeline.yaml",
    )
    print("Pipeline compiled -> pipelines/compiled/inference_pipeline.yaml")
