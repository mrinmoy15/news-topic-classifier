"""Unit tests for dashboard/bq_queries.py.

Verifies that each query function returns well-formed SQL with the correct
project/dataset interpolation, expected column names, and key SQL constructs.
No BigQuery connection is required.
"""
from __future__ import annotations

import pytest

from dashboard.bq_queries import (
    confusion_data,
    daily_accuracy,
    label_distribution,
    llm_eval_sample,
    per_class_metrics,
    performance_trend,
    recent_predictions,
    summary_stats,
)

_PROJECT = "my-project"
_DATASET = "MY_DATASET"
_TABLE   = f"{_PROJECT}.{_DATASET}.news_topic_classifier_predictions"

# ─── Helpers ─────────────────────────────────────────────────────────────────

def _sql(fn, **kwargs):
    return fn(_PROJECT, _DATASET, **kwargs)


# ═══════════════════════════════════════════════════════════════════════════════
# Shared: all functions interpolate project and dataset
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("fn", [
    daily_accuracy,
    label_distribution,
    confusion_data,
    summary_stats,
    per_class_metrics,
    performance_trend,
])
def test_table_reference_contains_project_and_dataset(fn):
    sql = _sql(fn)
    assert _PROJECT in sql
    assert _DATASET in sql


@pytest.mark.parametrize("fn", [
    daily_accuracy,
    label_distribution,
    confusion_data,
    summary_stats,
    per_class_metrics,
    performance_trend,
    recent_predictions,
    llm_eval_sample,
])
def test_returns_non_empty_string(fn):
    sql = _sql(fn)
    assert isinstance(sql, str) and len(sql) > 20


# ═══════════════════════════════════════════════════════════════════════════════
# daily_accuracy
# ═══════════════════════════════════════════════════════════════════════════════

def test_daily_accuracy_has_prediction_date():
    assert "prediction_date" in _sql(daily_accuracy)

def test_daily_accuracy_has_accuracy_column():
    assert "accuracy" in _sql(daily_accuracy)

def test_daily_accuracy_groups_by_date():
    sql = _sql(daily_accuracy)
    assert "GROUP BY" in sql.upper() and "prediction_date" in sql


# ═══════════════════════════════════════════════════════════════════════════════
# label_distribution
# ═══════════════════════════════════════════════════════════════════════════════

def test_label_distribution_has_predicted_label():
    assert "predicted_label" in _sql(label_distribution)

def test_label_distribution_groups_by_date_and_label():
    sql = _sql(label_distribution).upper()
    assert "GROUP BY" in sql


# ═══════════════════════════════════════════════════════════════════════════════
# confusion_data
# ═══════════════════════════════════════════════════════════════════════════════

def test_confusion_data_has_true_and_predicted_labels():
    sql = _sql(confusion_data)
    assert "true_label" in sql
    assert "predicted_label" in sql

def test_confusion_data_has_count():
    assert "COUNT" in _sql(confusion_data).upper()


# ═══════════════════════════════════════════════════════════════════════════════
# per_class_metrics
# ═══════════════════════════════════════════════════════════════════════════════

def test_per_class_metrics_contains_all_five_labels():
    sql = _sql(per_class_metrics)
    for label in ("business", "entertainment", "politics", "sport", "tech"):
        assert label in sql

def test_per_class_metrics_has_safe_divide():
    assert "SAFE_DIVIDE" in _sql(per_class_metrics).upper()

def test_per_class_metrics_has_precision_recall_f1():
    sql = _sql(per_class_metrics)
    assert "precision" in sql
    assert "recall" in sql
    assert "f1" in sql

def test_per_class_metrics_has_support():
    assert "support" in _sql(per_class_metrics)

def test_per_class_metrics_uses_unnest():
    assert "UNNEST" in _sql(per_class_metrics).upper()


# ═══════════════════════════════════════════════════════════════════════════════
# performance_trend
# ═══════════════════════════════════════════════════════════════════════════════

def test_performance_trend_has_accuracy():
    assert "accuracy" in _sql(performance_trend)

def test_performance_trend_has_avg_confidence():
    assert "avg_confidence" in _sql(performance_trend)

def test_performance_trend_has_prediction_date():
    assert "prediction_date" in _sql(performance_trend)

def test_performance_trend_groups_by_date():
    sql = _sql(performance_trend).upper()
    assert "GROUP BY" in sql


# ═══════════════════════════════════════════════════════════════════════════════
# llm_eval_sample
# ═══════════════════════════════════════════════════════════════════════════════

def test_llm_eval_sample_default_limit():
    sql = _sql(llm_eval_sample)
    assert "LIMIT 20" in sql

def test_llm_eval_sample_custom_n():
    sql = llm_eval_sample(_PROJECT, _DATASET, n=10)
    assert "LIMIT 10" in sql

def test_llm_eval_sample_uses_rand_for_random_order():
    assert "RAND()" in _sql(llm_eval_sample).upper()

def test_llm_eval_sample_selects_body():
    assert "body" in _sql(llm_eval_sample)

def test_llm_eval_sample_selects_true_and_predicted_label():
    sql = _sql(llm_eval_sample)
    assert "true_label" in sql
    assert "predicted_label" in sql


# ═══════════════════════════════════════════════════════════════════════════════
# recent_predictions
# ═══════════════════════════════════════════════════════════════════════════════

def test_recent_predictions_default_days():
    sql = _sql(recent_predictions)
    assert "30" in sql  # default 30-day window

def test_recent_predictions_custom_days():
    sql = recent_predictions(_PROJECT, _DATASET, days=7)
    assert "7" in sql

def test_recent_predictions_has_confidence():
    assert "confidence" in _sql(recent_predictions)


# ═══════════════════════════════════════════════════════════════════════════════
# summary_stats
# ═══════════════════════════════════════════════════════════════════════════════

def test_summary_stats_has_overall_accuracy():
    assert "overall_accuracy" in _sql(summary_stats)

def test_summary_stats_has_avg_confidence():
    assert "avg_confidence" in _sql(summary_stats)

def test_summary_stats_has_first_and_latest_run():
    sql = _sql(summary_stats)
    assert "first_run" in sql
    assert "latest_run" in sql
