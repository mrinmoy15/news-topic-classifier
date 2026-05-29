from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional
import pyarrow.parquet as pq
import torch
from google.cloud import bigquery
from torch.utils.data import Dataset

from news_topic_classifier.config import LABEL2ID

logger = logging.getLogger(__name__)


# =============================================================================
# SECTION 1 — SQL builder
# =============================================================================
def _build_extraction_query(
    source_table: str,
    text_col: str,
    label_col: str,
    title_col: str,
    input_table_size: Optional[int],
    sample_size: Optional[int],
) -> str:
    """
    Build the BQ extraction SQL.

    Full table when sample_size is None.
    FARM_FINGERPRINT-based deterministic sampling when sample_size is set.

    Parameters
    ----------
    source_table : str
        Fully qualified BQ table e.g. `bigquery-public-data.bbc_news.fulltext`
    text_col : str
        Column containing article body text (`body`).
    label_col : str
        Column containing labels (`category`).
    title_col : str
        Column containing article titles (`title`).
    input_table_size : int or None
        Total row count of the input table. Required when sample_size is set.
    sample_size : int or None
        None = full table. int = deterministic FARM_FINGERPRINT sample.

    Returns
    -------
    str
        SQL string ready to pass to ``client.query()``.
    """

    case_clauses = "\n".join(
        f"WHEN '{cat}' THEN {lid}"
        for cat, lid in LABEL2ID.items()
    )

    label_case = f"""
        CASE {label_col}
            {case_clauses}
            ELSE NULL
        END AS label
    """

    base_filter = f"""
        WHERE {text_col}  IS NOT NULL
          AND {label_col} IS NOT NULL
          AND {title_col} IS NOT NULL
    """

    if sample_size is None:
        query = f"""
            SELECT
                title,
                {label_col} AS category,
                {text_col}  AS text,
                {label_case}
            FROM `{source_table}`
            {base_filter}
        """
    else:

        pct = min(round(sample_size / input_table_size * 100), 100)
        query = f"""
            SELECT
                title,
                {label_col} AS category,
                {text_col}  AS text,
                {label_case}
            FROM `{source_table}`
            {base_filter}
                AND MOD(ABS(FARM_FINGERPRINT({label_col})), 100) < {pct}
        """
    
    # logg the query
    logger.info("*" * 80)
    logger.info("Constructed input BQ query: \n%s", query)
    logger.info("*" * 80)

    return query




# =============================================================================
# SECTION 2 — GCS write
# =============================================================================
def _gcs_output_path(gcs_bucket_data: str) -> str:
    """
    Build a versioned GCS Parquet path using UTC timestamp.

    Pattern: `gs://{bucket}/data/raw/{YYYY-MM-DDTHH-MM-SS}/bbc_news.parquet`

    Using HH-MM-SS (hyphens not colons) so the path is valid on all OS
    and unambiguous in GCS object names.
    """
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    return f"gs://{gcs_bucket_data}/data/raw/{ts}/bbc_news.parquet"


# =============================================================================
# SECTION 3 — Public entry point
# =============================================================================
def extract_from_bigquery(
    gcp_project: str,
    bq_dataset: str,
    source_table: str,
    text_col: str,
    label_col: str,
    title_col: str,
    gcs_bucket_data: str,
    input_table_size: Optional[int],
    sample_size: Optional[int] = None,
) -> str:
    """
    Extract BBC news data from BigQuery and write to GCS as Parquet.
    
    Flow
    ----
    1. Run filtered SQL with label mapping → temp BQ table
    2. Export temp BQ table → GCS Parquet (no data in Python memory)
    3. Delete temp BQ table

    Parameters
    ----------
    gcp_project : str
        GCP project ID e.g. `cs-cdwp-data-dev2188`.
    bq_dataset : str
        BQ dataset to write the temp table into e.g. `DATA_SCNCE_DEV_DATA`.
    source_table : str
        Fully qualified source BQ table e.g. `bigquery-public-data.bbc_news.fulltext`.
    text_col : str
        text column name (`body`).
    label_col : str
        Label column name (`category`).
    title_col : str
        Title column name (`title`).
    gcs_output_uri : str
        GCS URI e.g. `gs://model-data/data/raw/2026-05-25T14-32-01/bbc_news.parquet`.
    input_table_size : int or None
        Total row count of the source table. Required only when sample_size is set.
    sample_size : int or None
        None = full table (production default).
        int  = deterministic FARM_FINGERPRINT sample (dev / testing).
    
    Returns
    -------
    str
        GCS URI of the written Parquet file — passed as output artifact
        to the next pipeline step (preprocess component).
    """

    client = bigquery.Client(project=gcp_project)

    # ------------------------------------------------------------------
    # Step 1 — Run filtered SQL -> temp BQ table
    # ------------------------------------------------------------------

    temp_table_id = (
        f"{gcp_project}.{bq_dataset}.bbc_news_extract_temp"
    )

    sql = _build_extraction_query(
        source_table=source_table,
        text_col=text_col,
        label_col=label_col,
        title_col=title_col,
        input_table_size=input_table_size,
        sample_size=sample_size,
    )

    job_config = bigquery.QueryJobConfig(
        destination=temp_table_id,
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
    )

    logger.info("Running extraction query -> temp table %s", temp_table_id)

    query_job = client.query(sql, job_config=job_config)
    
    # Wait for completion
    query_job.result()

    logger.info(
        "Temp table written - %d rows.",
        client.get_table(temp_table_id).num_rows,
    )

    # ------------------------------------------------------------------
    # Step 2 — Export temp BQ table -> GCS Parquet
    # ------------------------------------------------------------------

    extract_job_config = bigquery.ExtractJobConfig(
        destination_format=bigquery.DestinationFormat.PARQUET,
        compression=bigquery.Compression.SNAPPY,
    )

    gcs_output_uri = _gcs_output_path(gcs_bucket_data)

    logger.info("Exporting temp table -> %s", gcs_output_uri)

    extract_job = client.extract_table(
        source=temp_table_id,
        destination_uris=[gcs_output_uri],
        job_config=extract_job_config,
    )
    
    # Wait for completion
    extract_job.result()

    logger.info("Export complete -> %s", gcs_output_uri)

    # ------------------------------------------------------------------
    # Step 3 — Delete temp BQ table
    # ------------------------------------------------------------------
    client.delete_table(temp_table_id)
    logger.info("Temp table %s deleted.", temp_table_id)

    return gcs_output_uri


# =============================================================================
# SECTION 4 — Dataset
# =============================================================================
class BBCNewsDataset(Dataset):

    """
    PyTorch Dataset for BBC news classification.

    Reads from a local Parquet file via PyArrow memory-mapping.
    Each __getitem__ call reads only the requested row from the memory-mapped
    file, then delegates tokenization, truncation, and padding to the
    HuggingFace fast tokenizer in a single Rust call.

    Parameters
    ----------
    local_parquet_path : str
        Local path to the Parquet file.
    tokenizer : PreTrainedTokenizerFast
        HuggingFace fast tokenizer — loaded once externally and passed in.
        Must be a fast (Rust-backed) tokenizer for efficient truncation.
    max_length : int
        Maximum token sequence length including [CLS] and [SEP] (512 for BERT).
        Texts longer than this are truncated from the right by the tokenizer.
    use_title : bool
        If True, prepend title to body text before tokenization.

    Examples
    --------
    >>> dataset = BBCNewsDataset(
    ...     local_parquet_path="/tmp/train.parquet",
    ...     tokenizer=tokenizer,
    ... )
    >>> len(dataset)
    1747
    >>> dataset[0].keys()
    dict_keys(['input_ids', 'attention_mask', 'label'])
    """

    def __init__(
        self,
        local_parquet_path: str,
        tokenizer,
        max_length: int  = 512,
        use_title:  bool = False,
    ) -> None:
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.use_title  = use_title

        self.table = pq.read_table(
            local_parquet_path,
            columns=["text", "label", "title"] if use_title else ["text", "label"],
            memory_map=True,
        )

        logger.info(
            "BBCNewsDataset ready - %d rows, memory-mapped from %s",
            len(self.table),
            local_parquet_path,
        )

    def __len__(self) -> int:
        return len(self.table)
    
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        
        text  = self.table.column("text")[idx].as_py()
        label = self.table.column("label")[idx].as_py()

        # Title + body combination if enabled
        if self.use_title:
            title = self.table.column("title")[idx].as_py()
            text  = title + " " + text

        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        tokens = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        return {
            "input_ids": tokens,
            "attention_mask": attention_mask,
            "label": torch.tensor(label, dtype=torch.long),
        }


if __name__ == "__main__":
    
    import hydra
    from omegaconf import DictConfig
    from transformers import BertTokenizer
    from pathlib import Path
    import os
    from google.cloud import storage

    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    @hydra.main(
        config_path= str(PROJECT_ROOT / "conf"),
        config_name="config",
        version_base=None,
    )
    def main(cfg: DictConfig) -> None:

        print("\n" + "=" * 80)
        print("BQ extract → GCS Parquet")
        print("=" * 80)

        gcs_uri = extract_from_bigquery(
            gcp_project=cfg.environment.gcp_project,
            bq_dataset=cfg.environment.bq_dataset,
            source_table=cfg.data.bq_source_table,
            text_col=cfg.data.bq_text_column,
            label_col=cfg.data.bq_label_column,
            title_col=cfg.data.bq_title_column,
            gcs_bucket_data=cfg.environment.gcs_bucket_data,
            input_table_size=None,
            sample_size=None,
        )
        print(f"Written to: {gcs_uri}")


        print("\n" + "=" * 80)
        print("BBCNewsDataset")
        print("=" * 80)

        # Download from GCS -> local
        local_dir = (
            "/tmp" 
            if os.getenv("CLOUD_ML_PROJECT_ID") 
            else (PROJECT_ROOT / "data" / "interim")
        )
     
        local_data_path =  str(Path(local_dir) / "bbc_news.parquet")
        
        path = gcs_uri.replace("gs://", "")
        bucket_name = path.split("/")[0]
        blob_name   = "/".join(path.split("/")[1:])

        storage_client = storage.Client(project=cfg.environment.gcp_project)
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.download_to_filename(local_data_path)
        print(f"Downloaded to {local_data_path}")

        LOCAL_MODEL_PATH = PROJECT_ROOT / "models" / "base-models" / "bert-base-uncased"

        # Load tokenizer from local path
        tokenizer = BertTokenizer.from_pretrained(
            LOCAL_MODEL_PATH,
            local_files_only=True,
        )

        # Instantiate dataset
        dataset = BBCNewsDataset(
            local_parquet_path=local_data_path,
            tokenizer=tokenizer,
            max_length=cfg.model.max_seq_length,
            use_title=False,
        )

        print(f"Dataset length : {len(dataset)}")
        print(f"Sample keys    : {dataset[0].keys()}")
        print(f"input_ids shape: {dataset[0]['input_ids'].shape}")
        print(f"label          : {dataset[0]['label']}")

    main()



