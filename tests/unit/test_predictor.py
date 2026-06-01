"""Unit tests for news_topic_classifier.modeling.predict."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from news_topic_classifier.config import ID2LABEL, NUM_LABELS
from news_topic_classifier.dataset import BBCNewsDataset
from news_topic_classifier.modeling.predict import (
    compute_metrics,
    load_model_tokenizer,
    predict,
    run_inference,
    save_predictions,
)


# ─── compute_metrics ─────────────────────────────────────────────────────────

def test_compute_metrics_perfect_accuracy():
    preds  = np.arange(5)
    labels = np.arange(5)
    assert compute_metrics(preds, labels, ID2LABEL)["accuracy"] == pytest.approx(1.0)


def test_compute_metrics_zero_accuracy():
    # every prediction is off by one class
    preds  = np.array([1, 2, 3, 4, 0])
    labels = np.array([0, 1, 2, 3, 4])
    assert compute_metrics(preds, labels, ID2LABEL)["accuracy"] == pytest.approx(0.0)


def test_compute_metrics_partial_accuracy():
    # 5 correct, 5 wrong — all 5 classes present in both arrays
    preds  = np.array([0, 1, 2, 3, 4, 1, 2, 3, 4, 0])
    labels = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
    assert compute_metrics(preds, labels, ID2LABEL)["accuracy"] == pytest.approx(0.5)


def test_compute_metrics_returns_accuracy_key():
    # All 5 classes must be present — compute_metrics uses a fixed 5-class target_names
    preds  = np.arange(5)
    labels = np.arange(5)
    assert "accuracy" in compute_metrics(preds, labels, ID2LABEL)


def test_compute_metrics_returns_report_key():
    preds  = np.arange(5)
    labels = np.arange(5)
    assert "report" in compute_metrics(preds, labels, ID2LABEL)


def test_compute_metrics_accuracy_is_float():
    preds  = np.arange(5)
    labels = np.arange(5)
    assert isinstance(compute_metrics(preds, labels, ID2LABEL)["accuracy"], float)


def test_compute_metrics_report_has_class_names():
    preds  = np.arange(5)
    labels = np.arange(5)
    report = compute_metrics(preds, labels, ID2LABEL)["report"]
    for name in ID2LABEL.values():
        assert name in report


def test_compute_metrics_report_class_has_precision_recall_f1():
    preds  = np.arange(5)
    labels = np.arange(5)
    report = compute_metrics(preds, labels, ID2LABEL)["report"]
    for cls in ID2LABEL.values():
        assert "precision" in report[cls]
        assert "recall"    in report[cls]
        assert "f1-score"  in report[cls]


# ─── run_inference ───────────────────────────────────────────────────────────

def _loader(tmp_parquet, dummy_tokenizer):
    ds = BBCNewsDataset(tmp_parquet, dummy_tokenizer, max_length=16)
    return DataLoader(ds, batch_size=2, shuffle=False), len(ds)


def test_run_inference_preds_shape(tmp_parquet, dummy_tokenizer, dummy_model):
    loader, n = _loader(tmp_parquet, dummy_tokenizer)
    preds, _, _ = run_inference(dummy_model, loader, torch.device("cpu"))
    assert preds.shape == (n,)


def test_run_inference_probs_shape(tmp_parquet, dummy_tokenizer, dummy_model):
    loader, n = _loader(tmp_parquet, dummy_tokenizer)
    _, probs, _ = run_inference(dummy_model, loader, torch.device("cpu"))
    assert probs.shape == (n, NUM_LABELS)


def test_run_inference_labels_shape_with_labels(tmp_parquet, dummy_tokenizer, dummy_model):
    loader, n = _loader(tmp_parquet, dummy_tokenizer)
    _, _, labels = run_inference(dummy_model, loader, torch.device("cpu"), has_labels=True)
    assert labels is not None and labels.shape == (n,)


def test_run_inference_labels_none_without_labels(tmp_parquet, dummy_tokenizer, dummy_model):
    loader, _ = _loader(tmp_parquet, dummy_tokenizer)
    _, _, labels = run_inference(dummy_model, loader, torch.device("cpu"), has_labels=False)
    assert labels is None


def test_run_inference_probs_sum_to_one(tmp_parquet, dummy_tokenizer, dummy_model):
    loader, n = _loader(tmp_parquet, dummy_tokenizer)
    _, probs, _ = run_inference(dummy_model, loader, torch.device("cpu"))
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(n), atol=1e-5)


def test_run_inference_preds_are_valid_class_indices(tmp_parquet, dummy_tokenizer, dummy_model):
    loader, _ = _loader(tmp_parquet, dummy_tokenizer)
    preds, _, _ = run_inference(dummy_model, loader, torch.device("cpu"))
    assert ((preds >= 0) & (preds < NUM_LABELS)).all()


def test_run_inference_probs_are_nonnegative(tmp_parquet, dummy_tokenizer, dummy_model):
    loader, _ = _loader(tmp_parquet, dummy_tokenizer)
    _, probs, _ = run_inference(dummy_model, loader, torch.device("cpu"))
    assert (probs >= 0).all()


# ─── save_predictions ────────────────────────────────────────────────────────

@pytest.fixture
def sample_preds():
    n      = 6
    preds  = np.array([i % NUM_LABELS for i in range(n)])
    probs  = np.random.dirichlet(np.ones(NUM_LABELS), size=n).astype(np.float32)
    labels = np.array([i % NUM_LABELS for i in range(n)])
    return preds, probs, labels


def _patch_save(monkeypatch, tmp_path, captured: list):
    """Set up working dir + patches so save_predictions writes locally."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "processed").mkdir(parents=True)

    def _fake_write(table, path, **kwargs):
        captured.append(table)

    return (
        patch("news_topic_classifier.modeling.predict.pq.write_table", _fake_write),
        patch("news_topic_classifier.modeling.predict.storage.Client"),
        patch("news_topic_classifier.modeling.predict.Path.unlink"),
    )


def test_save_predictions_schema_with_labels(sample_preds, monkeypatch, tmp_path):
    preds, probs, labels = sample_preds
    captured = []

    p1, p2, p3 = _patch_save(monkeypatch, tmp_path, captured)
    with p1, p2, p3:
        save_predictions(preds, probs, labels, ID2LABEL, "gs://bucket/pred", "proj")

    cols = captured[0].schema.names
    assert "predicted"  in cols
    assert "confidence" in cols
    assert "true_label" in cols
    assert "correct"    in cols
    for name in ID2LABEL.values():
        assert f"prob_{name}" in cols


def test_save_predictions_schema_without_labels(sample_preds, monkeypatch, tmp_path):
    preds, probs, _ = sample_preds
    captured = []

    p1, p2, p3 = _patch_save(monkeypatch, tmp_path, captured)
    with p1, p2, p3:
        save_predictions(preds, probs, None, ID2LABEL, "gs://bucket/pred", "proj")

    cols = captured[0].schema.names
    assert "true_label" not in cols
    assert "correct"    not in cols
    assert "predicted"  in cols


def test_save_predictions_row_count(sample_preds, monkeypatch, tmp_path):
    preds, probs, labels = sample_preds
    captured = []

    p1, p2, p3 = _patch_save(monkeypatch, tmp_path, captured)
    with p1, p2, p3:
        save_predictions(preds, probs, labels, ID2LABEL, "gs://bucket/pred", "proj")

    assert captured[0].num_rows == len(preds)


def test_save_predictions_returns_gcs_uri(sample_preds, monkeypatch, tmp_path):
    preds, probs, labels = sample_preds

    p1, p2, p3 = _patch_save(monkeypatch, tmp_path, [])
    with p1, p2, p3:
        uri = save_predictions(preds, probs, labels, ID2LABEL, "gs://bucket/pred", "proj")

    assert uri.startswith("gs://bucket/pred/")
    assert uri.endswith("predictions.parquet")


def test_save_predictions_confidence_values_in_range(sample_preds, monkeypatch, tmp_path):
    """Confidence column should hold the max-class probability, bounded [0, 1]."""
    preds, probs, labels = sample_preds
    captured = []

    p1, p2, p3 = _patch_save(monkeypatch, tmp_path, captured)
    with p1, p2, p3:
        save_predictions(preds, probs, labels, ID2LABEL, "gs://bucket/pred", "proj")

    confidence = captured[0].column("confidence").to_pylist()
    assert all(0.0 <= c <= 1.0 for c in confidence)


# ─── load_model_tokenizer ────────────────────────────────────────────────────

def test_load_model_tokenizer_returns_model_and_tokenizer(tmp_path):
    mock_model     = MagicMock()
    mock_tokenizer = MagicMock()
    mock_model.to.return_value = mock_model  # .to(device) returns self

    with patch("news_topic_classifier.modeling.predict.BertForSequenceClassification.from_pretrained",
               return_value=mock_model), \
         patch("news_topic_classifier.modeling.predict.BertTokenizerFast.from_pretrained",
               return_value=mock_tokenizer):
        model, tokenizer = load_model_tokenizer(str(tmp_path), torch.device("cpu"))

    assert model     is mock_model
    assert tokenizer is mock_tokenizer


def test_load_model_tokenizer_moves_model_to_device(tmp_path):
    mock_model = MagicMock()
    mock_model.to.return_value = mock_model

    with patch("news_topic_classifier.modeling.predict.BertForSequenceClassification.from_pretrained",
               return_value=mock_model), \
         patch("news_topic_classifier.modeling.predict.BertTokenizerFast.from_pretrained"):
        load_model_tokenizer(str(tmp_path), torch.device("cpu"))

    mock_model.to.assert_called_once_with(torch.device("cpu"))


def test_load_model_tokenizer_sets_eval_mode(tmp_path):
    mock_model = MagicMock()
    mock_model.to.return_value = mock_model

    with patch("news_topic_classifier.modeling.predict.BertForSequenceClassification.from_pretrained",
               return_value=mock_model), \
         patch("news_topic_classifier.modeling.predict.BertTokenizerFast.from_pretrained"):
        load_model_tokenizer(str(tmp_path), torch.device("cpu"))

    mock_model.eval.assert_called_once()


# ─── predict orchestrator ────────────────────────────────────────────────────
# run_inference, load_model_tokenizer, compute_metrics, and save_predictions are
# all mocked so no real model, GCS, or MLflow connection is needed.
# The test_loader is a plain MagicMock because predict() only passes it straight
# to run_inference (which is mocked) and log_params (also mocked).

def _make_run_cm():
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=MagicMock())
    cm.__exit__  = MagicMock(return_value=False)
    return cm


def _predict_all_patches(mock_run_cm, mock_model, mock_metrics):
    return [
        patch("news_topic_classifier.modeling.predict._setup_mlflow_tracking"),
        patch("news_topic_classifier.modeling.predict.mlflow.set_experiment"),
        patch("news_topic_classifier.modeling.predict.mlflow.start_run",
              return_value=mock_run_cm),
        patch("news_topic_classifier.modeling.predict.mlflow.log_params"),
        patch("news_topic_classifier.modeling.predict.mlflow.log_metric"),
        patch("news_topic_classifier.modeling.predict.mlflow.log_metrics"),
        patch("news_topic_classifier.modeling.predict.mlflow.log_param"),
        patch("news_topic_classifier.modeling.predict.load_model_tokenizer",
              return_value=(mock_model, MagicMock())),
        patch("news_topic_classifier.modeling.predict.run_inference",
              return_value=(
                  np.zeros(10, dtype=np.int64),
                  np.ones((10, NUM_LABELS)) / NUM_LABELS,
                  np.zeros(10, dtype=np.int64),
              )),
        patch("news_topic_classifier.modeling.predict.compute_metrics",
              return_value=mock_metrics),
        patch("news_topic_classifier.modeling.predict.save_predictions",
              return_value="gs://bucket/pred/file.parquet"),
    ]


def _make_mock_model():
    m = MagicMock()
    m.config.id2label = ID2LABEL
    return m


def _make_mock_metrics():
    return {
        "accuracy": 0.9,
        "report": {
            "business": {"precision": 0.9, "recall": 0.9, "f1-score": 0.9, "support": 10},
            "accuracy": 0.9,   # scalar — should be skipped by the log_metrics loop
        },
    }


def test_predict_returns_metrics_and_uri(cfg, tmp_path):
    from contextlib import ExitStack
    mock_model   = _make_mock_model()
    mock_metrics = _make_mock_metrics()
    mock_loader  = MagicMock()
    mock_loader.dataset.__len__ = MagicMock(return_value=10)

    with ExitStack() as stack:
        for p in _predict_all_patches(_make_run_cm(), mock_model, mock_metrics):
            stack.enter_context(p)

        metrics, uri = predict(
            cfg=cfg,
            model_path=str(tmp_path),
            test_loader=mock_loader,
            device=torch.device("cpu"),
            gcs_output_dir="gs://bucket/pred/",
        )

    assert metrics == mock_metrics
    assert uri == "gs://bucket/pred/file.parquet"


def test_predict_returns_empty_metrics_when_no_labels(cfg, tmp_path):
    """When run_inference returns labels=None, metrics dict is empty."""
    from contextlib import ExitStack
    mock_model  = _make_mock_model()
    mock_loader = MagicMock()

    no_label_patches = [
        patch("news_topic_classifier.modeling.predict._setup_mlflow_tracking"),
        patch("news_topic_classifier.modeling.predict.mlflow.set_experiment"),
        patch("news_topic_classifier.modeling.predict.mlflow.start_run",
              return_value=_make_run_cm()),
        patch("news_topic_classifier.modeling.predict.mlflow.log_params"),
        patch("news_topic_classifier.modeling.predict.mlflow.log_param"),
        patch("news_topic_classifier.modeling.predict.load_model_tokenizer",
              return_value=(mock_model, MagicMock())),
        patch("news_topic_classifier.modeling.predict.run_inference",
              return_value=(
                  np.zeros(5, dtype=np.int64),
                  np.ones((5, NUM_LABELS)) / NUM_LABELS,
                  None,   # no labels
              )),
        patch("news_topic_classifier.modeling.predict.save_predictions",
              return_value="gs://bucket/pred/file.parquet"),
    ]

    with ExitStack() as stack:
        for p in no_label_patches:
            stack.enter_context(p)

        metrics, uri = predict(
            cfg=cfg,
            model_path=str(tmp_path),
            test_loader=mock_loader,
            device=torch.device("cpu"),
            gcs_output_dir="gs://bucket/pred/",
        )

    assert metrics == {}
    assert uri == "gs://bucket/pred/file.parquet"
