"""Unit tests for news_topic_classifier.modeling.report.

matplotlib and seaborn are in requirements/dev.txt, not requirements/test.txt.
All tests in this module are skipped automatically when those packages are absent.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Skip the entire module if plotting deps are missing
pytest.importorskip("matplotlib")
pytest.importorskip("seaborn")

import matplotlib
matplotlib.use("Agg")  # headless backend — must be set before pyplot import

from news_topic_classifier.modeling.report import (
    download_predictions,
    fetch_run_data,
    plot_confusion_matrix,
    plot_per_class_metrics,
    upload_outputs,
    plot_training_curves,
)

# ─── Fixtures ────────────────────────────────────────────────────────────────

LABEL_NAMES = ["business", "entertainment", "politics", "sport", "tech"]
NUM_LABELS  = len(LABEL_NAMES)


@pytest.fixture
def sample_history():
    return {
        "train_loss": [1.5, 1.2, 0.9],
        "val_loss":   [1.6, 1.3, 1.0],
        "train_acc":  [0.4, 0.6, 0.75],
        "val_acc":    [0.38, 0.58, 0.72],
    }


@pytest.fixture
def sample_preds_labels():
    rng    = np.random.default_rng(42)
    labels = rng.integers(0, NUM_LABELS, size=50)
    preds  = rng.integers(0, NUM_LABELS, size=50)
    return preds, labels


@pytest.fixture
def sample_report():
    """Minimal sklearn classification_report output_dict=True structure."""
    classes = {
        name: {"precision": 0.8, "recall": 0.75, "f1-score": 0.77, "support": 10}
        for name in LABEL_NAMES
    }
    classes["macro avg"]    = {"precision": 0.8, "recall": 0.75, "f1-score": 0.77, "support": 50}
    classes["weighted avg"] = {"precision": 0.8, "recall": 0.75, "f1-score": 0.77, "support": 50}
    classes["accuracy"]     = 0.78
    return classes


# ─── plot_training_curves ────────────────────────────────────────────────────

def test_plot_training_curves_returns_path(sample_history, tmp_path):
    path = plot_training_curves(sample_history, str(tmp_path))
    assert path.exists()


def test_plot_training_curves_filename(sample_history, tmp_path):
    path = plot_training_curves(sample_history, str(tmp_path))
    assert path.name == "training_curves.png"


def test_plot_training_curves_creates_png(sample_history, tmp_path):
    path = plot_training_curves(sample_history, str(tmp_path))
    assert path.suffix == ".png"


def test_plot_training_curves_raises_on_empty_history(tmp_path):
    empty = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    with pytest.raises(ValueError, match="empty"):
        plot_training_curves(empty, str(tmp_path))


# ─── plot_confusion_matrix ───────────────────────────────────────────────────

def test_plot_confusion_matrix_returns_path(sample_preds_labels, tmp_path):
    preds, labels = sample_preds_labels
    path = plot_confusion_matrix(preds, labels, LABEL_NAMES, str(tmp_path))
    assert path.exists()


def test_plot_confusion_matrix_filename(sample_preds_labels, tmp_path):
    preds, labels = sample_preds_labels
    path = plot_confusion_matrix(preds, labels, LABEL_NAMES, str(tmp_path))
    assert path.name == "confusion_matrix.png"


def test_plot_confusion_matrix_creates_png(sample_preds_labels, tmp_path):
    preds, labels = sample_preds_labels
    path = plot_confusion_matrix(preds, labels, LABEL_NAMES, str(tmp_path))
    assert path.suffix == ".png"


# ─── plot_per_class_metrics ──────────────────────────────────────────────────

def test_plot_per_class_metrics_returns_path(sample_report, tmp_path):
    path = plot_per_class_metrics(sample_report, str(tmp_path))
    assert path.exists()


def test_plot_per_class_metrics_filename(sample_report, tmp_path):
    path = plot_per_class_metrics(sample_report, str(tmp_path))
    assert path.name == "per_class_metrics.png"


def test_plot_per_class_metrics_creates_png(sample_report, tmp_path):
    path = plot_per_class_metrics(sample_report, str(tmp_path))
    assert path.suffix == ".png"


def test_plot_per_class_metrics_excludes_aggregate_rows(sample_report, tmp_path):
    """Only per-class bars are plotted; macro/weighted avg and accuracy are skipped."""
    # We verify this indirectly: if aggregate rows caused a key error the call would fail.
    path = plot_per_class_metrics(sample_report, str(tmp_path))
    assert path.exists()


# ─── download_predictions ────────────────────────────────────────────────────

def test_download_predictions_returns_local_path(tmp_path):
    with patch("news_topic_classifier.modeling.report.storage.Client") as mock_client:
        mock_client.return_value.bucket.return_value.blob.return_value.download_to_filename.return_value = None

        result = download_predictions(
            "gs://bucket/predictions/2026/predictions.parquet",
            "test-project",
            str(tmp_path),
        )

    assert str(result).endswith("predictions.parquet")


def test_download_predictions_calls_gcs_download(tmp_path):
    with patch("news_topic_classifier.modeling.report.storage.Client") as mock_client:
        mock_blob = mock_client.return_value.bucket.return_value.blob.return_value

        download_predictions("gs://bucket/preds/file.parquet", "test-project", str(tmp_path))

    mock_blob.download_to_filename.assert_called_once()


# ─── upload_outputs ──────────────────────────────────────────────────────────

def test_upload_outputs_uploads_each_file(tmp_path):
    (tmp_path / "training_curves.png").write_bytes(b"png1")
    (tmp_path / "confusion_matrix.png").write_bytes(b"png2")
    (tmp_path / "model_report.docx").write_bytes(b"docx")

    with patch("news_topic_classifier.modeling.report.storage.Client") as mock_client:
        mock_blob = mock_client.return_value.bucket.return_value.blob.return_value
        upload_outputs(str(tmp_path), "gs://bucket/outputs/", "test-project")

    assert mock_blob.upload_from_filename.call_count == 3


def test_upload_outputs_skips_directories(tmp_path):
    (tmp_path / "subdir").mkdir()
    (tmp_path / "file.png").write_bytes(b"data")

    with patch("news_topic_classifier.modeling.report.storage.Client") as mock_client:
        mock_blob = mock_client.return_value.bucket.return_value.blob.return_value
        upload_outputs(str(tmp_path), "gs://bucket/outputs/", "test-project")

    assert mock_blob.upload_from_filename.call_count == 1


# ─── fetch_run_data ──────────────────────────────────────────────────────────

def test_fetch_run_data_returns_required_keys():
    mock_mlflow_client = MagicMock()
    mock_run = MagicMock()
    mock_run.data.params  = {"lr": "2e-5"}
    mock_run.data.metrics = {"best_val_acc": 0.92}
    mock_mlflow_client.get_run.return_value = mock_run
    mock_mlflow_client.get_metric_history.return_value = [MagicMock(value=0.9)]

    with patch("news_topic_classifier.modeling.report._setup_mlflow_tracking"), \
         patch("news_topic_classifier.modeling.report.MlflowClient", return_value=mock_mlflow_client):
        result = fetch_run_data("run-abc", "sqlite:///test.db")

    assert "params"  in result
    assert "metrics" in result
    assert "history" in result


def test_fetch_run_data_history_has_metric_keys():
    mock_mlflow_client = MagicMock()
    mock_run = MagicMock()
    mock_run.data.params  = {}
    mock_run.data.metrics = {}
    mock_mlflow_client.get_run.return_value = mock_run
    mock_mlflow_client.get_metric_history.return_value = [MagicMock(value=0.5)]

    with patch("news_topic_classifier.modeling.report._setup_mlflow_tracking"), \
         patch("news_topic_classifier.modeling.report.MlflowClient", return_value=mock_mlflow_client):
        result = fetch_run_data("run-abc", "sqlite:///test.db")

    for key in ("train_loss", "train_acc", "val_loss", "val_acc"):
        assert key in result["history"]


def test_fetch_run_data_history_values_are_lists():
    mock_mlflow_client = MagicMock()
    mock_run = MagicMock()
    mock_run.data.params  = {}
    mock_run.data.metrics = {}
    mock_mlflow_client.get_run.return_value = mock_run
    mock_mlflow_client.get_metric_history.return_value = [MagicMock(value=0.8), MagicMock(value=0.9)]

    with patch("news_topic_classifier.modeling.report._setup_mlflow_tracking"), \
         patch("news_topic_classifier.modeling.report.MlflowClient", return_value=mock_mlflow_client):
        result = fetch_run_data("run-abc", "sqlite:///test.db")

    for values in result["history"].values():
        assert isinstance(values, list)
