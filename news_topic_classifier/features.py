from __future__ import annotations

import logging
from datetime import datetime, timezone
from google.cloud import bigquery

logger = logging.getLogger(__name__)

# ==========================================================
# SECTION 1 — SQL builder
# ==========================================================
def _build_preprocessing_query(
    source_table: str,
    val_pct: int,
    test_pct: int,
) -> str:
    """
    Build the BQ preprocessing SQL.

    Reads from the raw external table (GCS Parquet), applies text cleaning,
    and assigns each row a deterministic stratified split label.

    Parameters
    ----------
    source_table : str
        Fully qualified BQ table or external table ID.
    val_pct : int
        Validation split as a whole-number percentage  e.g. 10 for 10 %.
    test_pct : int
        Test split as a whole-number percentage  e.g. 10 for 10 %.

    Returns
    -------
    str
        SQL string ready to pass to `client.query()`.
    
    """
    query = rf"""
        WITH numbered AS (
            SELECT
                title,
                category,
                text,
                label,
                ROW_NUMBER() OVER (
                    PARTITION BY category
                    ORDER BY     FARM_FINGERPRINT(title)
                ) AS rn
            FROM `{source_table}`
        )
        SELECT
            NORMALIZE(
                TRIM(REGEXP_REPLACE(title, r'\s+', ' ')),
                NFKC
            ) AS title,

            category,
            label,

            NORMALIZE(
                TRIM(
                    REGEXP_REPLACE(
                        REGEXP_REPLACE(
                            REGEXP_REPLACE(text, r'<[^>]+>', ' '),
                            r'&(?:amp|lt|gt|quot|apos|nbsp);|&#\d+;', ' '
                        ),
                        r'\s+', ' '
                    )
                ),
                NFKC
            ) AS text,

            CASE
                WHEN MOD(rn, 100) < {test_pct}   THEN 'test'
                WHEN MOD(rn, 100) < {test_pct} + {val_pct}  THEN 'val'
                ELSE  'train'
            END AS split

        FROM numbered
    """

    logger.info("*" * 80)
    logger.info("Constructed preprocessing query:\n%s", query)
    logger.info("*" * 80)

    return query


# ==========================================================
# SECTION 2 — GCS path builder
# ==========================================================
def _gcs_split_output_paths(gcs_bucket_data: str) -> dict[str, str]:
    """
    Build versioned GCS Parquet paths for all three splits under one timestamp.

    Pattern:
        gs://{bucket}/data/processed/{YYYY-MM-DDTHH-MM-SS}/{split}.parquet

    All splits share the same timestamp so a preprocessing run is one
    atomic versioned artifact — you always know which val/test set belongs
    to which train set.

    Parameters
    ----------
    gcs_bucket_data : str
        GCS bucket name (no gs:// prefix)  e.g. `my-project-data`

    Returns
    -------
    dict[str, str]
        `{"train": "gs://...", "val": "gs://...", "test": "gs://..."}`
    """

    ts   = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    base = f"gs://{gcs_bucket_data}/data/processed/{ts}"

    return {
        "train": f"{base}/train.parquet",
        "val":   f"{base}/val.parquet",
        "test":  f"{base}/test.parquet",
    }


# ==========================================================
# SECTION 3 — Public entry point
# ==========================================================
def preprocess_and_split(
    gcp_project: str,
    bq_dataset: str,
    raw_gcs_uri: str,
    gcs_bucket_data: str,
    val_split: float = 0.1,
    test_split: float = 0.1,
) -> dict[str, str]:
    """
    Clean and split the raw BBC News Parquet into train / val / test on GCS.

    Flow
    ----
    1. Create a BQ external table pointing at the raw GCS Parquet URI
    2. Run the cleaning + split-assignment SQL -> staging BQ temp table.
    3. For each split: filter staging table -> split temp table -> GCS Parquet.
       Delete each split temp table immediately after export.
    4. Delete staging table + drop external table.

    Parameters
    ----------
    gcp_project : str
        GCP project ID  e.g. `my-project-123`.
    bq_dataset : str
        BQ dataset for temp tables  e.g. `DATA_SCNCE_DEV_DATA`.
    raw_gcs_uri : str
        GCS URI of the raw Parquet produced by `extract_from_bigquery()`
        e.g. `gs://my-bucket/data/raw/2026-05-27T10-00-00/bbc_news.parquet`.
    gcs_bucket_data : str
        GCS bucket name for processed output  e.g. `my-project-data`.
    val_split : float
        Fraction of data for validation  e.g. `0.1` for 10 %.
    test_split : float
        Fraction of data for test  e.g. `0.1` for 10 %.

    Returns
    -------
    dict[str, str]
        GCS URIs for each split:
        ``{"train": "gs://...", "val": "gs://...", "test": "gs://..."}``.
        Passed as input artifacts to the training pipeline component.
    """

    val_pct  = round(val_split  * 100)
    test_pct = round(test_split * 100)

    client = bigquery.Client(project=gcp_project)

    external_table_id = f"{gcp_project}.{bq_dataset}.bbc_news_raw_ext"
    staging_table_id  = f"{gcp_project}.{bq_dataset}.bbc_news_preprocessed_temp"

    # ------------------------------------------------------------------
    # Step 1 — Create external table from raw GCS Parquet
    # ------------------------------------------------------------------
    # Delete first so a stale pointer from a previous run is never reused.
    client.delete_table(external_table_id, not_found_ok=True)

    external_config = bigquery.ExternalConfig("PARQUET")
    external_config.source_uris = [raw_gcs_uri]
    external_config.autodetect  = True

    ext_table = bigquery.Table(external_table_id)
    ext_table.external_data_configuration = external_config
    client.create_table(ext_table)

    logger.info("External table created: %s -> %s", external_table_id, raw_gcs_uri)


    # ------------------------------------------------------------------
    # Step 2 — Run cleaning + split SQL -> staging table
    # -------------------------------------------------------------------

    sql = _build_preprocessing_query(
        source_table=external_table_id,
        val_pct=val_pct,
        test_pct=test_pct,
    )

    job_config = bigquery.QueryJobConfig(
        destination=staging_table_id,
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
    )
    
    logger.info("Running preprocessing query -> %s", staging_table_id)
    query_job = client.query(sql, job_config=job_config)
    # Wait for completion.
    query_job.result()

    logger.info(
        "Staging table written — %d rows.",
        client.get_table(staging_table_id).num_rows,
    )

    # ------------------------------------------------------------------
    # Step 3 — Export each split -> GCS Parquet
    # ------------------------------------------------------------------
    split_paths = _gcs_split_output_paths(gcs_bucket_data)

    extract_config = bigquery.ExtractJobConfig(
        destination_format=bigquery.DestinationFormat.PARQUET,
        compression=bigquery.Compression.SNAPPY,
    )

    for split_name, gcs_path in split_paths.items():
        
        split_table_id = f"{gcp_project}.{bq_dataset}.bbc_news_{split_name}_temp"
        
        # Filter staging -> per-split temp table

        filter_query = f"""
            SELECT 
                title, 
                category, 
                text, 
                label
            FROM 
                `{staging_table_id}`
            WHERE  
                split = '{split_name}'
        """
        filter_job = client.query(
            filter_query,
            job_config=bigquery.QueryJobConfig(
                destination=split_table_id,
                write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
            ),
        )
        filter_job.result()

        logger.info(
            "%s -> %d rows",
            split_name,
            client.get_table(split_table_id).num_rows,
        )

        # Export split temp table -> GCS
        extract_job = client.extract_table(
            source=split_table_id,
            destination_uris=[gcs_path],
            job_config=extract_config,
        )

        extract_job.result()

        
        logger.info("%s exported -> %s", split_name, gcs_path)

        client.delete_table(split_table_id)
        logger.info("Temp table %s deleted.", split_table_id)

    # ------------------------------------------------------------------
    # Step 4 — Cleanup staging table + external table
    # ------------------------------------------------------------------
    client.delete_table(staging_table_id)
    logger.info("Staging table %s deleted.", staging_table_id)

    client.delete_table(external_table_id)
    logger.info("External table %s dropped.", external_table_id)

    return split_paths



if __name__ == "__main__":

    import hydra
    from omegaconf import DictConfig
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    @hydra.main(
        config_path=str(PROJECT_ROOT / "conf"),
        config_name="config",
        version_base=None,
    )
    def main(cfg: DictConfig) -> None:

        print("\n" + "=" * 80)
        print("BQ External Table -> Clean + Split -> GCS Parquet")
        print("=" * 80)

        split_uris = preprocess_and_split(
            gcp_project     = cfg.environment.gcp_project,
            bq_dataset      = cfg.environment.bq_dataset,
            raw_gcs_uri     = cfg.data.raw_gcs_uri,
            gcs_bucket_data = cfg.environment.gcs_bucket_data,
            val_split       = cfg.training.val_split,
            test_split      = cfg.training.test_split,
        )

        print("\nProcessed splits written:")
        for split_name, uri in split_uris.items():
            print(f"  {split_name:<5} -> {uri}")

    main()




