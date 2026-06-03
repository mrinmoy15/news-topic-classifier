"""Shared fixtures for all unit tests."""
from __future__ import annotations

from unittest.mock import MagicMock

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from omegaconf import OmegaConf

LABEL2ID  = {"business": 0, "entertainment": 1, "politics": 2, "sport": 3, "tech": 4}
ID2LABEL  = {v: k for k, v in LABEL2ID.items()}
NUM_LABELS = 5
SEQ_LEN    = 16  # tiny for fast tests


# ─── Config ──────────────────────────────────────────────────────────────────

@pytest.fixture
def cfg():
    return OmegaConf.create({
        "project":  {"name": "test-project"},
        "model":    {
            "bert_base_model": "bert-base-uncased",
            "num_labels": NUM_LABELS,
            "max_seq_length": SEQ_LEN,
        },
        "training": {
            "epochs": 2,
            "batch_size": 2,
            "lr": 2e-5,
            "warmup_steps": 0,
            "weight_decay": 0.01,
            "val_split": 0.1,
            "test_split": 0.1,
            "early_stopping_patience": 3,
        },
        "environment": {
            "gcp_project": "test-project",
            "gcs_bucket_data": "test-data-bucket",
            "gcs_bucket_artifacts": "test-artifacts-bucket",
            "mlflow": {"tracking_uri": "sqlite:///test_mlflow.db"},
        },
    })


# ─── Parquet helpers ─────────────────────────────────────────────────────────

def _write_parquet(path, n: int = 10) -> None:
    labels = [i % NUM_LABELS for i in range(n)]
    pq.write_table(
        pa.table({
            "text":     pa.array([f"body text {i}" for i in range(n)]),
            "label":    pa.array(labels, type=pa.int64()),
            "title":    pa.array([f"title {i}" for i in range(n)]),
            "category": pa.array([ID2LABEL[l] for l in labels]),
        }),
        str(path),
    )


@pytest.fixture
def tmp_parquet(tmp_path):
    """Single Parquet file with 10 rows."""
    p = tmp_path / "data.parquet"
    _write_parquet(p, n=10)
    return str(p)


@pytest.fixture
def tmp_split_parquets(tmp_path):
    """Three Parquet files: train (8), val (4), test (4)."""
    for name, n in [("train", 8), ("val", 4), ("test", 4)]:
        _write_parquet(tmp_path / f"{name}.parquet", n=n)
    return (
        str(tmp_path / "train.parquet"),
        str(tmp_path / "val.parquet"),
        str(tmp_path / "test.parquet"),
    )


# ─── Tokenizer stub ──────────────────────────────────────────────────────────

class _FakeTokenizer:
    """Minimal tokenizer that returns fixed-shape tensors — no BERT needed."""

    def __call__(self, text, max_length=16, padding=None, truncation=None, return_tensors=None):
        return {
            "input_ids":      torch.ones(1, max_length, dtype=torch.long),
            "attention_mask": torch.ones(1, max_length, dtype=torch.long),
        }

    def save_pretrained(self, path):
        pass


@pytest.fixture
def dummy_tokenizer():
    return _FakeTokenizer()


# ─── Model stub ──────────────────────────────────────────────────────────────

class _FakeOutput:
    def __init__(self, logits, loss=None):
        self.logits = logits
        self.loss   = loss


class _FakeModel(torch.nn.Module):
    """Tiny linear layer with the same interface as BertForSequenceClassification."""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(SEQ_LEN, NUM_LABELS)
        self.config = MagicMock()
        self.config.id2label = ID2LABEL
        self.config.label2id = LABEL2ID

    def forward(self, input_ids, attention_mask, labels=None):
        # input_ids: (B, SEQ_LEN) → logits: (B, NUM_LABELS)
        logits = self.linear(input_ids.float())
        loss   = torch.nn.functional.cross_entropy(logits, labels) if labels is not None else None
        return _FakeOutput(logits=logits, loss=loss)

    def save_pretrained(self, path):
        pass


@pytest.fixture
def dummy_model():
    return _FakeModel()
