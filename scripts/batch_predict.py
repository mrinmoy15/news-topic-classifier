"""
Daily batch inference job for BBC News Topic Classifier.

Pulls today's partition of BBC News articles from BigQuery, runs BERT
classification in mini-batches, and writes predictions back to BigQuery.

The full BBC News dataset (2225 rows) is divided into 30 equal partitions
by ROW_NUMBER mod 30 — one partition per day of the month — so each run
processes ~74 articles.

Usage
-----
# Run for today's partition (computed from current UTC date):
    python scripts/batch_predict.py --environment dev

# Run for a specific day partition (0-29):
    python scripts/batch_predict.py --environment dev --day 5

# Triggered automatically by Cloud Scheduler at 6 AM ET (11 AM UTC) daily.
"""
from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone

from google.cloud import bigquery

from api.predictor import download_and_load, predict_texts

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment config
# ---------------------------------------------------------------------------

_ENV_CONFIG = {
    "dev": {
        "project":           "cs-cdwp-data-dev2188",
        "dataset":           "DATA_SCNCE_DEV_DATA",
        "model_gcs_uri":     "gs://cs-cdwp-data-dev2188-model-artifacts/models/bert-bbc-finetuned/",
        "predictions_table": "news_topic_classifier_predictions",
    },
    "prd": {
        "project":           "cs-cdwp-data-prd2188",
        "dataset":           "DATA_SCNCE_DATA",
        "model_gcs_uri":     "gs://cs-cdwp-data-prd2188-model-artifacts/models/bert-bbc-finetuned/",
        "predictions_table": "news_topic_classifier_predictions",
    },
}

_CHUNK_SIZE = 64  # articles per inference mini-batch

# ---------------------------------------------------------------------------
# BigQuery helpers
# ---------------------------------------------------------------------------

def _source_query(day: int) -> str:
    """Return SQL for today's ~74-row partition of BBC News."""
    return f"""
    SELECT title, body, category AS true_label
    FROM (
        SELECT
            title,
            body,
            category,
            MOD(ROW_NUMBER() OVER (ORDER BY title), 30) AS day_num
        FROM `bigquery-public-data.bbc_news.fulltext`
        WHERE body IS NOT NULL AND title IS NOT NULL
    )
    WHERE day_num = {day}
    """


def _ensure_table(client: bigquery.Client, project: str, dataset: str, table: str) -> None:
    """Create the predictions table if it does not already exist."""
    schema = [
        bigquery.SchemaField("prediction_date",    "DATE"),
        bigquery.SchemaField("run_timestamp",       "TIMESTAMP"),
        bigquery.SchemaField("title",               "STRING"),
        bigquery.SchemaField("body",                "STRING"),
        bigquery.SchemaField("true_label",          "STRING"),
        bigquery.SchemaField("predicted_label",     "STRING"),
        bigquery.SchemaField("confidence",          "FLOAT64"),
        bigquery.SchemaField("score_business",      "FLOAT64"),
        bigquery.SchemaField("score_entertainment", "FLOAT64"),
        bigquery.SchemaField("score_politics",      "FLOAT64"),
        bigquery.SchemaField("score_sport",         "FLOAT64"),
        bigquery.SchemaField("score_tech",          "FLOAT64"),
        bigquery.SchemaField("day_partition",       "INT64"),
    ]
    full_ref = f"{project}.{dataset}.{table}"
    bq_table = bigquery.Table(full_ref, schema=schema)
    bq_table.time_partitioning = bigquery.TimePartitioning(
        type_=bigquery.TimePartitioningType.DAY,
        field="prediction_date",
    )
    client.create_table(bq_table, exists_ok=True)
    logger.info("Predictions table ready: %s", full_ref)


# ---------------------------------------------------------------------------
# Main job
# ---------------------------------------------------------------------------

def run_batch_predict(environment: str, day: int | None = None) -> int:
    """Run the batch prediction job. Returns the number of rows written."""
    cfg = _ENV_CONFIG[environment]

    if day is None:
        day = (datetime.now(timezone.utc).day - 1) % 30
    logger.info("Batch prediction  |  env=%s  day=%d", environment, day)

    # Load model from GCS
    model, tokenizer, device = download_and_load(
        gcs_uri=cfg["model_gcs_uri"],
        gcp_project=cfg["project"],
    )

    # Fetch today's articles from BigQuery
    bq = bigquery.Client(project=cfg["project"])
    _ensure_table(bq, cfg["project"], cfg["dataset"], cfg["predictions_table"])

    rows = [
        {"title": r.title, "body": r.body, "true_label": r.true_label}
        for r in bq.query(_source_query(day)).result()
    ]
    logger.info("Fetched %d articles for day=%d", len(rows), day)

    if not rows:
        logger.warning("No rows for day=%d — nothing to write", day)
        return 0

    # Run inference in mini-batches
    now            = datetime.now(timezone.utc)
    pred_date      = now.date().isoformat()
    run_ts         = now.isoformat()
    output_rows    = []

    for start in range(0, len(rows), _CHUNK_SIZE):
        chunk = rows[start : start + _CHUNK_SIZE]
        # Use body only — consistent with training data format
        texts = [r["body"] for r in chunk]
        preds = predict_texts(texts, model, tokenizer, device)

        for row, pred in zip(chunk, preds):
            output_rows.append({
                "prediction_date":    pred_date,
                "run_timestamp":      run_ts,
                "title":              row["title"],
                "body":               row["body"],
                "true_label":         row["true_label"],
                "predicted_label":    pred.label,
                "confidence":         pred.confidence,
                "score_business":     pred.scores.get("business",      0.0),
                "score_entertainment":pred.scores.get("entertainment",  0.0),
                "score_politics":     pred.scores.get("politics",       0.0),
                "score_sport":        pred.scores.get("sport",          0.0),
                "score_tech":         pred.scores.get("tech",           0.0),
                "day_partition":      day,
            })

    # Write to BigQuery via streaming insert
    table_ref = f"{cfg['project']}.{cfg['dataset']}.{cfg['predictions_table']}"
    errors = bq.insert_rows_json(table_ref, output_rows)
    if errors:
        raise RuntimeError(f"BigQuery insert errors: {errors}")

    logger.info("Wrote %d predictions to %s", len(output_rows), table_ref)
    return len(output_rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Daily BBC News batch inference job")
    parser.add_argument(
        "--environment", "-e",
        choices=["dev", "prd"],
        default="dev",
        help="Target environment (default: dev)",
    )
    parser.add_argument(
        "--day",
        type=int,
        default=None,
        help="Day partition 0-29. Defaults to (UTC day of month - 1) %% 30.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    run_batch_predict(environment=args.environment, day=args.day)
