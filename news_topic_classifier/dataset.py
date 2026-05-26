# =============================================================================
# news_topic_classifier/dataset.py
#
# Responsibilities:
#   1. extract_from_bigquery() — BQ → Parquet (GCS or local)
#   2. load_from_filesystem()  — local Parquet/CSV → HuggingFace Dataset
#   3. BBCNewsDataset          — PyTorch Dataset wrapping HuggingFace Dataset
#
# Data flow:
#   BigQuery
#       └── extract_from_bigquery()
#               ├── local:      saves to data/raw/sample.parquet
#               └── dev/pp/prd: exports directly to GCS via BQ export job
#
#   Parquet (GCS or local)
#       └── load_from_filesystem() / load_from_gcs()
#               └── HuggingFace Dataset (memory mapped, arrow backed)
#                       └── BBCNewsDataset (PyTorch Dataset wrapper)
#                               └── DataLoader (batches for training)
# =============================================================================
 
from __future__ import annotations
 
import logging
from pathlib import Path
from typing import Optional
 
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset as HFDataset
from google.cloud import bigquery
from google.cloud import bigquery_storage
from omegaconf import DictConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase
 
from news_topic_classifier.config import LABEL2ID
 
logger = logging.getLogger(__name__)