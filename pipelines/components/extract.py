import os

from kfp import dsl

_TRAINER_IMAGE = os.environ.get(
    "TRAINER_IMAGE",
    "us-central1-docker.pkg.dev/cs-cdwp-data-dev2188/news-topic-classifier/trainer:latest",
)


@dsl.component(base_image=_TRAINER_IMAGE)
def extract_component(
    gcp_project: str,
    bq_dataset: str,
    source_table: str,
    text_col: str,
    label_col: str,
    title_col: str,
    gcs_bucket_data: str,
    input_table_size: int = 0,
    sample_size: int = 0,
) -> str:
    """
    KFP component — extract BBC News data from BigQuery and write to GCS Parquet.

    Thin wrapper around news_topic_classifier.dataset.extract_from_bigquery().
    The heavy logic lives in the core module; this component only wires
    Vertex AI pipeline I/O to function arguments.

    Parameters
    ----------
    gcp_project : str
        GCP project ID.
    bq_dataset : str
        BQ dataset for the temp extraction table.
    source_table : str
        Fully qualified BQ source table.
    text_col : str
        Column name for article body text.
    label_col : str
        Column name for category label.
    title_col : str
        Column name for article title.
    gcs_bucket_data : str
        GCS bucket name for raw Parquet output.
    input_table_size : int
        Total row count of the source table. 0 = not set (required when sample_size > 0).
    sample_size : int
        0 = full table (default). >0 = deterministic FARM_FINGERPRINT sample.

    Returns
    -------
    str
        GCS URI of the written raw Parquet file — passed to preprocess_component.
    """
    from news_topic_classifier.dataset import extract_from_bigquery

    # 0 is the sentinel meaning "not set" — convert back to None for the core function
    gcs_raw_uri = extract_from_bigquery(
        gcp_project=gcp_project,
        bq_dataset=bq_dataset,
        source_table=source_table,
        text_col=text_col,
        label_col=label_col,
        title_col=title_col,
        gcs_bucket_data=gcs_bucket_data,
        input_table_size=input_table_size or None,
        sample_size=sample_size or None,
    )

    return gcs_raw_uri
