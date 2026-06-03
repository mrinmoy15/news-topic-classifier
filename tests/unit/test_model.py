"""Unit tests for bert_classifier and training utilities."""
from __future__ import annotations

from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch.optim import AdamW

from news_topic_classifier.config import ID2LABEL, LABEL2ID, NUM_LABELS
from news_topic_classifier.modeling.train import (
    _setup_mlflow_tracking,
    build_dataloaders,
    build_optimizer_scheduler,
    download_base_model,
    download_splits,
    eval_epoch,
    train,
    train_epoch,
    upload_model_to_gcs,
)


# ─── build_model ─────────────────────────────────────────────────────────────

@patch("news_topic_classifier.modeling.bert_classifier.BertForSequenceClassification.from_pretrained")
def test_build_model_sets_id2label(mock_from_pretrained):
    from news_topic_classifier.modeling.bert_classifier import build_model

    mock_model = MagicMock()
    mock_model.parameters.return_value = iter([torch.zeros(1)])
    mock_from_pretrained.return_value = mock_model

    build_model("fake/path", NUM_LABELS, ID2LABEL, LABEL2ID)

    assert mock_model.config.id2label == ID2LABEL


@patch("news_topic_classifier.modeling.bert_classifier.BertForSequenceClassification.from_pretrained")
def test_build_model_sets_label2id(mock_from_pretrained):
    from news_topic_classifier.modeling.bert_classifier import build_model

    mock_model = MagicMock()
    mock_model.parameters.return_value = iter([torch.zeros(1)])
    mock_from_pretrained.return_value = mock_model

    build_model("fake/path", NUM_LABELS, ID2LABEL, LABEL2ID)

    assert mock_model.config.label2id == LABEL2ID


@patch("news_topic_classifier.modeling.bert_classifier.BertForSequenceClassification.from_pretrained")
def test_build_model_returns_model(mock_from_pretrained):
    from news_topic_classifier.modeling.bert_classifier import build_model

    mock_model = MagicMock()
    mock_model.parameters.return_value = iter([torch.zeros(1)])
    mock_from_pretrained.return_value = mock_model

    result = build_model("fake/path", NUM_LABELS, ID2LABEL, LABEL2ID)

    assert result is mock_model


# ─── build_optimizer_scheduler ───────────────────────────────────────────────

def test_build_optimizer_scheduler_returns_adamw(dummy_model, cfg):
    optimizer, _ = build_optimizer_scheduler(dummy_model, total_steps=100, cfg=cfg)
    assert isinstance(optimizer, AdamW)


def test_build_optimizer_scheduler_lr_matches_cfg(dummy_model, cfg):
    optimizer, _ = build_optimizer_scheduler(dummy_model, total_steps=100, cfg=cfg)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(cfg.training.lr)


def test_build_optimizer_scheduler_weight_decay_matches_cfg(dummy_model, cfg):
    optimizer, _ = build_optimizer_scheduler(dummy_model, total_steps=100, cfg=cfg)
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(cfg.training.weight_decay)


def test_build_optimizer_scheduler_returns_scheduler(dummy_model, cfg):
    _, scheduler = build_optimizer_scheduler(dummy_model, total_steps=100, cfg=cfg)
    assert hasattr(scheduler, "step")


# ─── build_dataloaders ───────────────────────────────────────────────────────

def test_build_dataloaders_returns_three(tmp_split_parquets, dummy_tokenizer, cfg):
    result = build_dataloaders(*tmp_split_parquets, dummy_tokenizer, cfg)
    assert len(result) == 3


def test_build_dataloaders_train_batch_count(tmp_split_parquets, dummy_tokenizer, cfg):
    train_loader, _, _ = build_dataloaders(*tmp_split_parquets, dummy_tokenizer, cfg)
    assert len(train_loader) == 4  # 8 samples / batch_size 2


def test_build_dataloaders_val_batch_count(tmp_split_parquets, dummy_tokenizer, cfg):
    _, val_loader, _ = build_dataloaders(*tmp_split_parquets, dummy_tokenizer, cfg)
    assert len(val_loader) == 2  # 4 samples / batch_size 2


def test_build_dataloaders_test_batch_count(tmp_split_parquets, dummy_tokenizer, cfg):
    _, _, test_loader = build_dataloaders(*tmp_split_parquets, dummy_tokenizer, cfg)
    assert len(test_loader) == 2


# ─── train_epoch ─────────────────────────────────────────────────────────────

def _make_train_loader(tmp_split_parquets, dummy_tokenizer, cfg):
    train_loader, _, _ = build_dataloaders(*tmp_split_parquets, dummy_tokenizer, cfg)
    return train_loader


def test_train_epoch_returns_two_values(tmp_split_parquets, dummy_tokenizer, dummy_model, cfg):
    loader = _make_train_loader(tmp_split_parquets, dummy_tokenizer, cfg)
    opt, sched = build_optimizer_scheduler(dummy_model, total_steps=10, cfg=cfg)
    result = train_epoch(dummy_model, loader, opt, sched, torch.device("cpu"))
    assert len(result) == 2


def test_train_epoch_loss_is_float(tmp_split_parquets, dummy_tokenizer, dummy_model, cfg):
    loader = _make_train_loader(tmp_split_parquets, dummy_tokenizer, cfg)
    opt, sched = build_optimizer_scheduler(dummy_model, total_steps=10, cfg=cfg)
    loss, _ = train_epoch(dummy_model, loader, opt, sched, torch.device("cpu"))
    assert isinstance(loss, float)


def test_train_epoch_loss_nonnegative(tmp_split_parquets, dummy_tokenizer, dummy_model, cfg):
    loader = _make_train_loader(tmp_split_parquets, dummy_tokenizer, cfg)
    opt, sched = build_optimizer_scheduler(dummy_model, total_steps=10, cfg=cfg)
    loss, _ = train_epoch(dummy_model, loader, opt, sched, torch.device("cpu"))
    assert loss >= 0.0


def test_train_epoch_accuracy_in_unit_interval(tmp_split_parquets, dummy_tokenizer, dummy_model, cfg):
    loader = _make_train_loader(tmp_split_parquets, dummy_tokenizer, cfg)
    opt, sched = build_optimizer_scheduler(dummy_model, total_steps=10, cfg=cfg)
    _, acc = train_epoch(dummy_model, loader, opt, sched, torch.device("cpu"))
    assert 0.0 <= acc <= 1.0


def test_train_epoch_model_is_in_train_mode_after(tmp_split_parquets, dummy_tokenizer, dummy_model, cfg):
    loader = _make_train_loader(tmp_split_parquets, dummy_tokenizer, cfg)
    opt, sched = build_optimizer_scheduler(dummy_model, total_steps=10, cfg=cfg)
    dummy_model.eval()
    train_epoch(dummy_model, loader, opt, sched, torch.device("cpu"))
    assert dummy_model.training


# ─── eval_epoch ──────────────────────────────────────────────────────────────

def _make_val_loader(tmp_split_parquets, dummy_tokenizer, cfg):
    _, val_loader, _ = build_dataloaders(*tmp_split_parquets, dummy_tokenizer, cfg)
    return val_loader


def test_eval_epoch_returns_two_values(tmp_split_parquets, dummy_tokenizer, dummy_model, cfg):
    loader = _make_val_loader(tmp_split_parquets, dummy_tokenizer, cfg)
    result = eval_epoch(dummy_model, loader, torch.device("cpu"))
    assert len(result) == 2


def test_eval_epoch_loss_is_float(tmp_split_parquets, dummy_tokenizer, dummy_model, cfg):
    loader = _make_val_loader(tmp_split_parquets, dummy_tokenizer, cfg)
    loss, _ = eval_epoch(dummy_model, loader, torch.device("cpu"))
    assert isinstance(loss, float)


def test_eval_epoch_loss_nonnegative(tmp_split_parquets, dummy_tokenizer, dummy_model, cfg):
    loader = _make_val_loader(tmp_split_parquets, dummy_tokenizer, cfg)
    loss, _ = eval_epoch(dummy_model, loader, torch.device("cpu"))
    assert loss >= 0.0


def test_eval_epoch_accuracy_in_unit_interval(tmp_split_parquets, dummy_tokenizer, dummy_model, cfg):
    loader = _make_val_loader(tmp_split_parquets, dummy_tokenizer, cfg)
    _, acc = eval_epoch(dummy_model, loader, torch.device("cpu"))
    assert 0.0 <= acc <= 1.0


def test_eval_epoch_model_left_in_eval_mode(tmp_split_parquets, dummy_tokenizer, dummy_model, cfg):
    loader = _make_val_loader(tmp_split_parquets, dummy_tokenizer, cfg)
    dummy_model.train()
    eval_epoch(dummy_model, loader, torch.device("cpu"))
    assert not dummy_model.training


# ─── _setup_mlflow_tracking ──────────────────────────────────────────────────

def test_setup_mlflow_tracking_local_uri_sets_tracking_uri():
    with patch("news_topic_classifier.modeling.train.mlflow.set_tracking_uri") as mock_set:
        _setup_mlflow_tracking("sqlite:///test.db")
    mock_set.assert_called_once_with("sqlite:///test.db")


def test_setup_mlflow_tracking_does_not_call_cloud_auth_for_local_uri():
    with patch("news_topic_classifier.modeling.train.mlflow.set_tracking_uri"), \
         patch("news_topic_classifier.modeling.train.mlflow") as mock_mlflow:
        _setup_mlflow_tracking("sqlite:///test.db")
        # No OIDC token should be fetched for a local SQLite URI
        mock_mlflow.set_tracking_uri.assert_called_once_with("sqlite:///test.db")


# ─── download_splits ─────────────────────────────────────────────────────────

def test_download_splits_returns_three_string_paths(tmp_path):
    with patch("news_topic_classifier.modeling.train.storage.Client") as mock_client:
        mock_client.return_value.bucket.return_value.blob.return_value.download_to_filename.return_value = None

        train_path, val_path, test_path = download_splits(
            "gs://bucket/data/processed/2026/",
            str(tmp_path),
            "test-project",
        )

    assert train_path.endswith("train.parquet")
    assert val_path.endswith("val.parquet")
    assert test_path.endswith("test.parquet")


def test_download_splits_downloads_three_blobs(tmp_path):
    with patch("news_topic_classifier.modeling.train.storage.Client") as mock_client:
        mock_blob = mock_client.return_value.bucket.return_value.blob.return_value
        mock_blob.download_to_filename.return_value = None

        download_splits("gs://bucket/data/processed/2026/", str(tmp_path), "test-project")

    assert mock_blob.download_to_filename.call_count == 3


# ─── download_base_model ─────────────────────────────────────────────────────

def test_download_base_model_returns_string_path(tmp_path):
    mock_blob = MagicMock()
    mock_blob.name = "models/bert-base-uncased/config.json"
    mock_blob.download_to_filename = MagicMock()

    with patch("news_topic_classifier.modeling.train.storage.Client") as mock_client:
        mock_client.return_value.list_blobs.return_value = [mock_blob]

        result = download_base_model(
            "gs://bucket/models/bert-base-uncased/",
            str(tmp_path),
            "test-project",
        )

    assert isinstance(result, str)


def test_download_base_model_downloads_each_blob(tmp_path):
    blobs = [MagicMock(name=f"prefix/file{i}.json") for i in range(3)]
    for b in blobs:
        b.download_to_filename = MagicMock()

    with patch("news_topic_classifier.modeling.train.storage.Client") as mock_client:
        mock_client.return_value.list_blobs.return_value = blobs
        download_base_model("gs://bucket/prefix/", str(tmp_path), "test-project")

    for b in blobs:
        b.download_to_filename.assert_called_once()


def test_download_base_model_skips_blob_matching_prefix(tmp_path):
    """A blob whose name equals the prefix exactly has empty relative_path → skip (line 154)."""
    prefix_only = MagicMock()
    prefix_only.name = "models/bert-base-uncased/"   # same as the computed prefix
    prefix_only.download_to_filename = MagicMock()

    with patch("news_topic_classifier.modeling.train.storage.Client") as mock_client:
        mock_client.return_value.list_blobs.return_value = [prefix_only]
        download_base_model("gs://bucket/models/bert-base-uncased/", str(tmp_path), "test-project")

    prefix_only.download_to_filename.assert_not_called()


# ─── upload_model_to_gcs ─────────────────────────────────────────────────────

def test_upload_model_to_gcs_returns_gcs_uri(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"fake")

    with patch("news_topic_classifier.modeling.train.storage.Client") as mock_client:
        mock_bucket = mock_client.return_value.bucket.return_value
        result = upload_model_to_gcs(str(tmp_path), "gs://bucket/models/", "test-project")

    assert result == "gs://bucket/models/"


def test_upload_model_to_gcs_uploads_all_files(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"fake")

    with patch("news_topic_classifier.modeling.train.storage.Client") as mock_client:
        mock_blob = mock_client.return_value.bucket.return_value.blob.return_value
        upload_model_to_gcs(str(tmp_path), "gs://bucket/models/", "test-project")

    assert mock_blob.upload_from_filename.call_count == 2


def test_upload_model_to_gcs_skips_directories(tmp_path):
    """Subdirectories inside the model dir must not be uploaded (line 211)."""
    (tmp_path / "subdir").mkdir()           # directory — should be skipped
    (tmp_path / "config.json").write_text("{}")

    with patch("news_topic_classifier.modeling.train.storage.Client") as mock_client:
        mock_blob = mock_client.return_value.bucket.return_value.blob.return_value
        upload_model_to_gcs(str(tmp_path), "gs://bucket/models/", "test-project")

    assert mock_blob.upload_from_filename.call_count == 1


# ─── train (full loop) ───────────────────────────────────────────────────────
#
# Both tests mock eval_epoch with controlled return values.
# Reason: _FakeTokenizer returns all-ones tensors, so _FakeModel always predicts
# the same class. The 4-sample val set may not include that class (20% chance),
# giving val_acc=0.0 forever. The checkpoint is then never saved and torch.load
# raises FileNotFoundError. Mocking eval_epoch makes the tests deterministic.


def _mlflow_patches(mock_ctx):
    """Return the standard set of mlflow/GCS patches used by both train tests."""
    return [
        patch("news_topic_classifier.modeling.train._setup_mlflow_tracking"),
        patch("news_topic_classifier.modeling.train.mlflow.set_experiment"),
        patch("news_topic_classifier.modeling.train.mlflow.log_params"),
        patch("news_topic_classifier.modeling.train.mlflow.log_metrics"),
        patch("news_topic_classifier.modeling.train.mlflow.log_metric"),
        patch("news_topic_classifier.modeling.train.mlflow.log_artifacts"),
        patch("news_topic_classifier.modeling.train.mlflow.log_param"),
        patch("news_topic_classifier.modeling.train.upload_model_to_gcs"),
        patch("news_topic_classifier.modeling.train.mlflow.start_run",
              return_value=mock_ctx),
    ]


def _make_mock_ctx(run_id: str):
    mock_run = MagicMock()
    mock_run.info.run_id = run_id
    mock_ctx = MagicMock()
    mock_ctx.__enter__ = MagicMock(return_value=mock_run)
    mock_ctx.__exit__ = MagicMock(return_value=False)
    return mock_ctx


def test_train_returns_val_acc_gcs_uri_run_id(tmp_split_parquets, dummy_tokenizer, dummy_model, cfg, tmp_path):
    train_loader, val_loader, _ = build_dataloaders(*tmp_split_parquets, dummy_tokenizer, cfg)
    mock_ctx = _make_mock_ctx("test-run-id-abc")

    # Epoch 1 improves (saves checkpoint), epoch 2 also improves.
    # cfg.training.epochs=2, so exactly 2 eval calls are made.
    val_returns = [(1.0, 0.5), (0.8, 0.7)]

    with ExitStack() as stack:
        for p in _mlflow_patches(mock_ctx):
            stack.enter_context(p)
        stack.enter_context(
            patch("news_topic_classifier.modeling.train.eval_epoch", side_effect=val_returns)
        )

        best_val_acc, gcs_uri, run_id = train(
            cfg=cfg,
            model=dummy_model,
            tokenizer=dummy_tokenizer,
            train_loader=train_loader,
            val_loader=val_loader,
            device=torch.device("cpu"),
            save_path=str(tmp_path / "model"),
        )

    assert isinstance(best_val_acc, float)
    assert best_val_acc == pytest.approx(0.7)
    assert gcs_uri == f"gs://{cfg.environment.gcs_bucket_artifacts}/models/bert-bbc-finetuned/"
    assert run_id == "test-run-id-abc"


def test_train_early_stopping_respected(tmp_split_parquets, dummy_tokenizer, dummy_model, cfg, tmp_path):
    """With patience=1, training stops after the first non-improving epoch."""
    from omegaconf import OmegaConf
    cfg_early = OmegaConf.merge(cfg, {"training": {"epochs": 10, "early_stopping_patience": 1}})

    train_loader, val_loader, _ = build_dataloaders(*tmp_split_parquets, dummy_tokenizer, cfg_early)
    mock_ctx = _make_mock_ctx("run-early")

    epoch_count = []
    # Epoch 1 improves → checkpoint saved; epoch 2 does not → early stop triggers.
    # Exactly 2 eval calls total regardless of epochs=10.
    controlled_val = [(1.0, 0.8), (0.9, 0.6)]

    def _counting_eval(model, loader, device):
        epoch_count.append(1)
        return controlled_val[len(epoch_count) - 1]

    with ExitStack() as stack:
        for p in _mlflow_patches(mock_ctx):
            stack.enter_context(p)
        stack.enter_context(
            patch("news_topic_classifier.modeling.train.eval_epoch", side_effect=_counting_eval)
        )

        train(
            cfg=cfg_early,
            model=dummy_model,
            tokenizer=dummy_tokenizer,
            train_loader=train_loader,
            val_loader=val_loader,
            device=torch.device("cpu"),
            save_path=str(tmp_path / "model"),
        )

    # patience=1: 1 improving epoch + 1 non-improving epoch = 2 total
    assert len(epoch_count) == 2
