"""
Integration tests for the end-to-end BBC News classification pipeline.

Test tiers
----------
Tier 1 — data pipeline  (~2-3 min, no GPU needed)
    test_01_extract        BigQuery → GCS Parquet (50-row sample)
    test_02_preprocess     raw Parquet → train/val/test GCS Parquet splits

Tier 2 — ML pipeline  (marked `slow`, ~20 min, requires base model on GCS)
    test_03_train          download splits → 1-epoch BERT fine-tune → GCS model
    test_04_predict        fine-tuned model → test-set predictions → GCS Parquet
    test_05_report         MLflow run data + predictions → GCS report artefacts

Run all tiers:
    INTEGRATION_TESTS=true pytest tests/integration/ -m integration -v

Run data-pipeline only (skip slow):
    INTEGRATION_TESTS=true pytest tests/integration/ -m "integration and not slow" -v
"""
from __future__ import annotations

import os

import pytest
import torch
from google.cloud import storage

from news_topic_classifier.config import ID2LABEL, LABEL2ID, NUM_LABELS
from news_topic_classifier.dataset import extract_from_bigquery
from news_topic_classifier.features import preprocess_and_split


# ---------------------------------------------------------------------------
# Tier 1 — data pipeline
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_01_extract(cfg, test_run_prefix, pipeline_artifacts):
    """Extract 50 rows from BigQuery and write a Parquet file to GCS."""
    # 500 rows → ~100 per category → rn 1-9 test, 10-19 val, 20-99 train.
    # 50 rows was too small: with ~10 rows per category rn only reaches 10,
    # so the train partition (rn >= 20) ends up empty and DataLoader raises
    # ValueError: num_samples should be a positive integer value, got 0.
    gcs_uri = extract_from_bigquery(
        gcp_project=cfg.environment.gcp_project,
        bq_dataset=cfg.environment.bq_dataset,
        source_table=cfg.data.bq_source_table,
        text_col=cfg.data.bq_text_column,
        label_col=cfg.data.bq_label_column,
        title_col=cfg.data.bq_title_column,
        gcs_bucket_data=cfg.environment.gcs_bucket_data,
        input_table_size=2225,
        sample_size=500,
    )

    # Verify the file was written to GCS
    path_no_scheme = gcs_uri.replace("gs://", "")
    bucket_name    = path_no_scheme.split("/")[0]
    blob_name      = "/".join(path_no_scheme.split("/")[1:])

    client = storage.Client(project=cfg.environment.gcp_project)
    blob   = client.bucket(bucket_name).blob(blob_name)

    assert blob.exists(), f"Expected GCS file not found: {gcs_uri}"
    assert gcs_uri.endswith("bbc_news.parquet")

    pipeline_artifacts["raw_gcs_uri"] = gcs_uri
    print(f"\n[extract] raw Parquet → {gcs_uri}")


@pytest.mark.integration
def test_02_preprocess(cfg, test_run_prefix, pipeline_artifacts):
    """Clean the raw Parquet and split into train / val / test on GCS."""
    assert "raw_gcs_uri" in pipeline_artifacts, "test_01_extract must run first"

    split_paths = preprocess_and_split(
        gcp_project=cfg.environment.gcp_project,
        bq_dataset=cfg.environment.bq_dataset,
        raw_gcs_uri=pipeline_artifacts["raw_gcs_uri"],
        gcs_bucket_data=cfg.environment.gcs_bucket_data,
        val_split=cfg.training.val_split,
        test_split=cfg.training.test_split,
    )

    client = storage.Client(project=cfg.environment.gcp_project)

    for split, gcs_uri in split_paths.items():
        path_no_scheme = gcs_uri.replace("gs://", "")
        bucket_name    = path_no_scheme.split("/")[0]
        blob_name      = "/".join(path_no_scheme.split("/")[1:])
        blob           = client.bucket(bucket_name).blob(blob_name)

        assert blob.exists(), f"Split '{split}' not found at {gcs_uri}"
        assert gcs_uri.endswith(f"{split}.parquet")
        print(f"[preprocess] {split:5} → {gcs_uri}")

    assert set(split_paths.keys()) == {"train", "val", "test"}
    pipeline_artifacts["split_paths"] = split_paths


# ---------------------------------------------------------------------------
# Tier 2 — ML pipeline (slow)
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.slow
def test_03_train(cfg, test_run_prefix, pipeline_artifacts, tmp_path):
    """Download splits, fine-tune BERT for 1 epoch, upload model to GCS."""
    assert "split_paths" in pipeline_artifacts, "test_02_preprocess must run first"

    from news_topic_classifier.modeling.bert_classifier import build_model
    from news_topic_classifier.modeling.train import (
        build_dataloaders,
        download_base_model,
        download_splits,
        train,
    )
    from transformers import BertTokenizerFast

    local_dir = str(tmp_path / "data")

    # Derive the GCS directory from the train URI using string ops, NOT pathlib.Path.
    # pathlib.Path converts forward slashes to backslashes on Windows, which breaks
    # GCS bucket names (e.g. "gs:\bucket\..." instead of "gs://bucket/...").
    train_uri        = pipeline_artifacts["split_paths"]["train"]
    gcs_splits_dir   = train_uri.rsplit("/", 1)[0] + "/"

    train_path, val_path, test_path = download_splits(
        gcs_processed_dir=gcs_splits_dir,
        local_dir=local_dir,
        gcp_project=cfg.environment.gcp_project,
    )

    # Download base model from GCS
    gcs_base_model = (
        f"gs://{cfg.environment.gcs_bucket_artifacts}"
        f"/models/base-models/{cfg.model.bert_base_model}/"
    )
    model_local = download_base_model(
        gcs_model_uri=gcs_base_model,
        local_dir=str(tmp_path / "base-model"),
        gcp_project=cfg.environment.gcp_project,
    )

    tokenizer   = BertTokenizerFast.from_pretrained(model_local, local_files_only=True)
    model       = build_model(model_local, NUM_LABELS, ID2LABEL, LABEL2ID)
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model       = model.to(device)

    train_loader, val_loader, _ = build_dataloaders(
        train_path, val_path, test_path, tokenizer, cfg
    )

    save_path = str(tmp_path / "finetuned")

    best_val_acc, gcs_model_uri, run_id = train(
        cfg=cfg,
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        save_path=save_path,
    )

    assert isinstance(best_val_acc, float)
    assert 0.0 <= best_val_acc <= 1.0
    assert gcs_model_uri.startswith("gs://")
    assert isinstance(run_id, str) and len(run_id) > 0

    pipeline_artifacts["gcs_model_uri"] = gcs_model_uri
    pipeline_artifacts["run_id"]        = run_id
    pipeline_artifacts["test_path"]     = test_path
    print(f"\n[train] val_acc={best_val_acc:.4f}  model→{gcs_model_uri}  run={run_id}")


@pytest.mark.integration
@pytest.mark.slow
def test_04_predict(cfg, test_run_prefix, pipeline_artifacts, tmp_path):
    """Load fine-tuned model, run inference on test split, save predictions to GCS."""
    if "gcs_model_uri" not in pipeline_artifacts:
        pytest.skip("test_03_train did not complete — skipping")

    from news_topic_classifier.modeling.predict import predict
    from news_topic_classifier.modeling.train import build_dataloaders, download_base_model
    from news_topic_classifier.dataset import BBCNewsDataset
    from transformers import BertTokenizerFast
    from torch.utils.data import DataLoader

    # Download fine-tuned model
    model_local = download_base_model(
        gcs_model_uri=pipeline_artifacts["gcs_model_uri"],
        local_dir=str(tmp_path / "finetuned"),
        gcp_project=cfg.environment.gcp_project,
    )

    tokenizer   = BertTokenizerFast.from_pretrained(model_local, local_files_only=True)
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dataset = BBCNewsDataset(
        local_parquet_path=pipeline_artifacts["test_path"],
        tokenizer=tokenizer,
        max_length=cfg.model.max_seq_length,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=0,
    )

    gcs_predictions_dir = (
        f"gs://{cfg.environment.gcs_bucket_data}/{test_run_prefix}/predictions/"
    )

    metrics, gcs_predictions_uri = predict(
        cfg=cfg,
        model_path=model_local,
        test_loader=test_loader,
        device=device,
        gcs_output_dir=gcs_predictions_dir,
    )

    # Verify predictions file on GCS
    path_no_scheme = gcs_predictions_uri.replace("gs://", "")
    blob = (
        storage.Client(project=cfg.environment.gcp_project)
        .bucket(path_no_scheme.split("/")[0])
        .blob("/".join(path_no_scheme.split("/")[1:]))
    )
    assert blob.exists(), f"Predictions not found at {gcs_predictions_uri}"
    assert "accuracy" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0

    pipeline_artifacts["gcs_predictions_uri"] = gcs_predictions_uri
    print(f"\n[predict] accuracy={metrics['accuracy']:.4f}  predictions→{gcs_predictions_uri}")


@pytest.mark.integration
@pytest.mark.slow
def test_05_report(cfg, test_run_prefix, pipeline_artifacts):
    """Generate training curves, confusion matrix, Word report, upload to GCS."""
    if "run_id" not in pipeline_artifacts:
        pytest.skip("test_03_train did not complete — skipping")
    if "gcs_predictions_uri" not in pipeline_artifacts:
        pytest.skip("test_04_predict did not complete — skipping")

    from news_topic_classifier.modeling.report import generate_report

    gcs_output_dir = generate_report(
        cfg=cfg,
        run_id=pipeline_artifacts["run_id"],
        gcs_predictions_uri=pipeline_artifacts["gcs_predictions_uri"],
    )

    assert gcs_output_dir.startswith("gs://")

    # Verify at least the training curves PNG was uploaded
    client      = storage.Client(project=cfg.environment.gcp_project)
    path        = gcs_output_dir.rstrip("/").replace("gs://", "")
    bucket_name = path.split("/")[0]
    prefix      = "/".join(path.split("/")[1:])
    blobs       = list(client.bucket(bucket_name).list_blobs(prefix=prefix))

    filenames = [b.name.rsplit("/", 1)[-1] for b in blobs]
    assert "training_curves.png" in filenames,  "training_curves.png not found in report output"
    assert "confusion_matrix.png" in filenames, "confusion_matrix.png not found in report output"
    assert "model_report.docx" in filenames,    "model_report.docx not found in report output"

    print(f"\n[report] {len(blobs)} artefacts → {gcs_output_dir}")
    for f in sorted(filenames):
        print(f"         {f}")
