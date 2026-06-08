import os

from kfp import dsl

_TRAINER_IMAGE = os.environ.get(
    "TRAINER_IMAGE",
    "us-central1-docker.pkg.dev/cs-cdwp-data-dev2188/news-topic-classifier/trainer:latest",
)


@dsl.component(base_image=_TRAINER_IMAGE)
def fetch_inference_data_component(
    gcp_project: str,
    gcs_bucket_data: str,
    source_table: str = "bigquery-public-data.bbc_news.fulltext",
) -> str:
    """
    KFP component — draw a random sample of 100 articles from the BBC News public
    dataset and write them to GCS as a Parquet file.

    Parameters
    ----------
    gcp_project : str
        GCP project ID (used for billing).
    gcs_bucket_data : str
        GCS bucket to write the input Parquet file.
    source_table : str
        Fully-qualified BigQuery source table.

    Returns
    -------
    str
        GCS URI of the written Parquet file — passed to run_batch_inference_component.
    """
    import os
    import tempfile

    import pyarrow as pa
    import pyarrow.parquet as pq
    from google.cloud import bigquery, storage

    query = f"""
    SELECT title, body, category
    FROM `{source_table}`
    WHERE body IS NOT NULL AND title IS NOT NULL
    ORDER BY RAND()
    LIMIT 100
    """

    bq = bigquery.Client(project=gcp_project)
    rows = [dict(r) for r in bq.query(query).result()]
    print(f"Fetched {len(rows)} articles from {source_table}")

    if not rows:
        raise ValueError(f"No articles found in {source_table}")

    table = pa.Table.from_pylist(rows)

    gcs_uri = f"gs://{gcs_bucket_data}/inference/samples/input.parquet"
    gcs_path = gcs_uri.replace("gs://", "")
    bucket_name, blob_name = gcs_path.split("/", 1)

    fd, tmp_path = tempfile.mkstemp(suffix=".parquet")
    os.close(fd)
    try:
        pq.write_table(table, tmp_path)
        storage.Client(project=gcp_project).bucket(bucket_name).blob(blob_name).upload_from_filename(tmp_path)
    finally:
        os.unlink(tmp_path)

    print(f"Input data written to {gcs_uri}")
    return gcs_uri
