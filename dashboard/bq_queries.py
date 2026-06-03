"""BigQuery SQL queries for the monitoring dashboard."""
from __future__ import annotations

_TABLE = "{project}.{dataset}.news_topic_classifier_predictions"


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
