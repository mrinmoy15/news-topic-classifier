from __future__ import annotations

from kfp import dsl


@dsl.component(
    base_image="us-central1-docker.pkg.dev/cs-cdwp-data-dev2188/news-topic-classifier/trainer:latest",
)
def evaluate_component(
    gcp_project: str,
    gcs_bucket_artifacts: str,
    gcs_predictions_uri: str,
    mlflow_run_id: str,
    mlflow_tracking_uri: str,
    project_name: str,
    environment_name: str,
    bert_base_model: str,
    num_labels: int,
    max_seq_length: int,
    epochs: int,
    batch_size: int,
    lr: float,
    warmup_steps: int,
    weight_decay: float,
    early_stopping_patience: int,
) -> str:
    """
    KFP component — generate evaluation report and upload to GCS.

    Thin wrapper around news_topic_classifier.modeling.report.generate_report().
    Downloads predictions from GCS, fetches training history from MLflow,
    generates plots and a Word document, then uploads all outputs to GCS.

    Parameters
    ----------
    gcp_project : str
        GCP project ID.
    gcs_bucket_artifacts : str
        GCS bucket name for report output (outputs/ subfolder).
    gcs_predictions_uri : str
        GCS URI of the predictions Parquet from predict_component.
    mlflow_run_id : str
        MLflow run ID from the training run — used to fetch metric history.
    mlflow_tracking_uri : str
        MLflow tracking URI — GCS path in dev/pp/prd.
    project_name : str
        MLflow experiment name / project name for the Word report title.
    environment_name : str
        Environment name (dev/pp/prd) — included in the Word report.
    bert_base_model : str
        Base model name — included in the Word report config table.
    num_labels : int
        Number of output classes — included in the Word report config table.
    max_seq_length : int
        Maximum tokeniser sequence length — included in the Word report.
    epochs : int
        Max training epochs — included in the Word report config table.
    batch_size : int
        Batch size — included in the Word report config table.
    lr : float
        Learning rate — included in the Word report config table.
    warmup_steps : int
        Warmup steps — included in the Word report config table.
    weight_decay : float
        Weight decay — included in the Word report config table.
    early_stopping_patience : int
        Early stopping patience — included in the Word report config table.

    Returns
    -------
    str
        GCS URI of the outputs folder containing figures and Word document.
    """
    from omegaconf import OmegaConf

    from news_topic_classifier.modeling.report import generate_report

    # ── Build minimal config ──────────────────────────────────────────────────
    cfg = OmegaConf.create({
        "project": {"name": project_name},
        "environment": {
            "name":                 environment_name,
            "gcp_project":          gcp_project,
            "gcs_bucket_artifacts": gcs_bucket_artifacts,
            "mlflow":               {"tracking_uri": mlflow_tracking_uri},
        },
        "model": {
            "bert_base_model": bert_base_model,
            "num_labels":      num_labels,
            "max_seq_length":  max_seq_length,
        },
        "training": {
            "epochs":                   epochs,
            "batch_size":               batch_size,
            "lr":                       lr,
            "warmup_steps":             warmup_steps,
            "weight_decay":             weight_decay,
            "early_stopping_patience":  early_stopping_patience,
        },
    })

    # ── Generate report and upload to GCS ─────────────────────────────────────
    gcs_outputs_uri = generate_report(
        cfg=cfg,
        run_id=mlflow_run_id,
        gcs_predictions_uri=gcs_predictions_uri,
    )

    return gcs_outputs_uri
