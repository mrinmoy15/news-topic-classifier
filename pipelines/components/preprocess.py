from __future__ import annotations

from kfp import dsl


@dsl.component(
    base_image="us-central1-docker.pkg.dev/cs-cdwp-data-dev2188/news-topic-classifier/trainer:latest",
)
def preprocess_component(
    gcp_project: str,
    bq_dataset: str,
    gcs_bucket_data: str,
    gcs_raw_uri: str,
    val_split: float = 0.1,
    test_split: float = 0.1,
) -> str:
    """
    KFP component — clean and split raw BBC News Parquet into train/val/test on GCS.

    Thin wrapper around news_topic_classifier.features.preprocess_and_split().
    Receives the raw GCS URI from extract_component and outputs a GCS directory
    URI containing train.parquet, val.parquet, and test.parquet.

    Parameters
    ----------
    gcp_project : str
        GCP project ID.
    bq_dataset : str
        BQ dataset for temp tables used during preprocessing.
    gcs_bucket_data : str
        GCS bucket name for processed split output.
    gcs_raw_uri : str
        GCS URI of the raw Parquet produced by extract_component.
    val_split : float
        Fraction of data for validation e.g. 0.1 for 10%.
    test_split : float
        Fraction of data for test e.g. 0.1 for 10%.

    Returns
    -------
    str
        GCS directory URI containing the three split Parquet files, e.g.
        ``gs://bucket/data/processed/2026-05-29T10-00-00/``
        Passed directly to train_component as gcs_splits_dir.
    """
    import json

    from news_topic_classifier.features import preprocess_and_split

    splits = preprocess_and_split(
        gcp_project=gcp_project,
        bq_dataset=bq_dataset,
        raw_gcs_uri=gcs_raw_uri,
        gcs_bucket_data=gcs_bucket_data,
        val_split=val_split,
        test_split=test_split,
    )

    # splits = {"train": "gs://...", "val": "gs://...", "test": "gs://..."}
    # All three share the same timestamped directory — return the common prefix.
    gcs_splits_dir = splits["train"].rsplit("/", 1)[0] + "/"

    return gcs_splits_dir
