"""BigQuery SQL queries for the monitoring dashboard."""
from __future__ import annotations

_TABLE  = "{project}.{dataset}.news_topic_classifier_predictions"
_LABELS = ["business", "entertainment", "politics", "sport", "tech"]


def recent_predictions(project: str, dataset: str, days: int = 30) -> str:
    table = _TABLE.format(project=project, dataset=dataset)
    return f"""
    SELECT
        prediction_date,
        title,
        true_label,
        predicted_label,
        confidence,
        score_business,
        score_entertainment,
        score_politics,
        score_sport,
        score_tech
    FROM `{table}`
    WHERE prediction_date >= DATE_SUB(CURRENT_DATE(), INTERVAL {days} DAY)
    ORDER BY prediction_date DESC
    LIMIT 500
    """


def daily_accuracy(project: str, dataset: str) -> str:
    table = _TABLE.format(project=project, dataset=dataset)
    return f"""
    SELECT
        prediction_date,
        COUNT(*)                                                AS total,
        COUNTIF(predicted_label = true_label)                  AS correct,
        ROUND(COUNTIF(predicted_label = true_label) / COUNT(*), 4) AS accuracy
    FROM `{table}`
    GROUP BY prediction_date
    ORDER BY prediction_date
    """


def label_distribution(project: str, dataset: str) -> str:
    table = _TABLE.format(project=project, dataset=dataset)
    return f"""
    SELECT
        prediction_date,
        predicted_label,
        COUNT(*) AS count
    FROM `{table}`
    GROUP BY prediction_date, predicted_label
    ORDER BY prediction_date, predicted_label
    """


def confusion_data(project: str, dataset: str) -> str:
    table = _TABLE.format(project=project, dataset=dataset)
    return f"""
    SELECT
        true_label,
        predicted_label,
        COUNT(*) AS count
    FROM `{table}`
    GROUP BY true_label, predicted_label
    ORDER BY true_label, predicted_label
    """


def summary_stats(project: str, dataset: str) -> str:
    table = _TABLE.format(project=project, dataset=dataset)
    return f"""
    SELECT
        COUNT(*)                                                    AS total_predictions,
        COUNTIF(predicted_label = true_label)                       AS correct,
        ROUND(COUNTIF(predicted_label = true_label) / COUNT(*), 4)  AS overall_accuracy,
        ROUND(AVG(confidence), 4)                                   AS avg_confidence,
        MIN(prediction_date)                                        AS first_run,
        MAX(prediction_date)                                        AS latest_run
    FROM `{table}`
    """


def per_class_metrics(project: str, dataset: str) -> str:
    """Precision, recall, F1 and support per label — computed entirely in BigQuery."""
    table = _TABLE.format(project=project, dataset=dataset)
    labels_sql = ", ".join(f"'{l}'" for l in _LABELS)
    return f"""
    WITH base AS (
        SELECT true_label, predicted_label
        FROM `{table}`
    ),
    labels AS (
        SELECT label FROM UNNEST([{labels_sql}]) AS label
    ),
    support AS (
        SELECT true_label AS label, COUNT(*) AS support
        FROM base GROUP BY true_label
    ),
    tp AS (
        SELECT true_label AS label, COUNT(*) AS tp
        FROM base WHERE true_label = predicted_label GROUP BY true_label
    ),
    predicted_count AS (
        SELECT predicted_label AS label, COUNT(*) AS predicted
        FROM base GROUP BY predicted_label
    )
    SELECT
        l.label,
        COALESCE(s.support, 0)   AS support,
        COALESCE(t.tp, 0)        AS true_positives,
        ROUND(SAFE_DIVIDE(COALESCE(t.tp, 0), COALESCE(p.predicted, 0)), 4) AS precision,
        ROUND(SAFE_DIVIDE(COALESCE(t.tp, 0), COALESCE(s.support,   0)), 4) AS recall,
        ROUND(SAFE_DIVIDE(
            2 * SAFE_DIVIDE(COALESCE(t.tp, 0), COALESCE(p.predicted, 0))
              * SAFE_DIVIDE(COALESCE(t.tp, 0), COALESCE(s.support,   0)),
            SAFE_DIVIDE(COALESCE(t.tp, 0), COALESCE(p.predicted, 0))
          + SAFE_DIVIDE(COALESCE(t.tp, 0), COALESCE(s.support,   0))
        ), 4) AS f1
    FROM labels l
    LEFT JOIN support        s ON l.label = s.label
    LEFT JOIN tp             t ON l.label = t.label
    LEFT JOIN predicted_count p ON l.label = p.label
    ORDER BY l.label
    """


def performance_trend(project: str, dataset: str) -> str:
    """Daily accuracy and average confidence — used to visualise model drift."""
    table = _TABLE.format(project=project, dataset=dataset)
    return f"""
    SELECT
        prediction_date,
        COUNT(*)                                                          AS total,
        ROUND(COUNTIF(predicted_label = true_label) / COUNT(*), 4)       AS accuracy,
        ROUND(AVG(confidence), 4)                                         AS avg_confidence
    FROM `{table}`
    GROUP BY prediction_date
    ORDER BY prediction_date
    """


def llm_eval_sample(project: str, dataset: str, n: int = 20) -> str:
    """Random sample of recent predictions for LLM-as-judge evaluation."""
    table = _TABLE.format(project=project, dataset=dataset)
    return f"""
    SELECT
        title,
        body,
        true_label,
        predicted_label,
        ROUND(confidence, 4) AS confidence
    FROM `{table}`
    WHERE body IS NOT NULL AND CHAR_LENGTH(body) > 50
    ORDER BY RAND()
    LIMIT {n}
    """
