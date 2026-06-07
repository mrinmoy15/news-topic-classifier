import os

from kfp import dsl

_TRAINER_IMAGE = os.environ.get(
    "TRAINER_IMAGE",
    "us-central1-docker.pkg.dev/cs-cdwp-data-dev2188/news-topic-classifier/trainer:latest",
)


@dsl.component(base_image=_TRAINER_IMAGE)
def fetch_inference_data_component(
    gcp_project: str,
    bq_dataset: str,
    gcs_bucket_data: str,
    source_table: str = "bigquery-public-data.bbc_news.fulltext",
    day: int = -1,
) -> str:
    """
    KFP component — query today's BBC News partition from BigQuery and write to GCS Parquet.

    Selects ~74 articles using the same MOD(ROW_NUMBER(), 30) partitioning
    scheme as the batch predict script so every article is covered over a
    rolling 30-day window.

    Parameters
    ----------
    gcp_project : str
        GCP project ID.
    bq_dataset : str
        BigQuery dataset (used only for billing; source is the public table).
    gcs_bucket_data : str
        GCS bucket to write the input Parquet file.
    source_table : str
        Fully-qualified BigQuery source table.
    day : int
        Partition index 0-29.  -1 = auto-compute from current UTC date.

    Returns
    -------
    str
        GCS URI of the written Parquet file — passed to run_batch_inference_component.
    """
    import os
    import tempfile
    from datetime import datetime, timezone

    import pyarrow as pa
    import pyarrow.parquet as pq
    from google.cloud import bigquery, storage

    if day < 0:
        day = (datetime.now(timezone.utc).day - 1) % 30

    query = f"""
    SELECT title, body, category AS true_label
    FROM (
        SELECT
            title,
            body,
            category,
            MOD(ROW_NUMBER() OVER (ORDER BY title), 30) AS day_num
        FROM `{source_table}`
        WHERE body IS NOT NULL AND title IS NOT NULL
    )
    WHERE day_num = {day}
    """

    bq = bigquery.Client(project=gcp_project)
    rows = [dict(r) for r in bq.query(query).result()]
    print(f"Fetched {len(rows)} articles for day_partition={day}")

    if not rows:
        raise ValueError(f"No articles found for day partition {day} in {source_table}")

    table = pa.Table.from_pylist(rows)

    gcs_uri = f"gs://{gcs_bucket_data}/inference/day={day}/input.parquet"
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
