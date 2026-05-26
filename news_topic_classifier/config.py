# =============================================================================
# news_topic_classifier/config.py
# Thin Hydra config loader + project-level constants
# =============================================================================

from __future__ import annotations

from omegaconf import DictConfig

# -----------------------------------------------------------------------------
# CONSTANTS
# -----------------------------------------------------------------------------

# BBC News label mapping — single source of truth
LABEL2ID: dict[str, int] = {
    "business":      0,
    "entertainment": 1,
    "politics":      2,
    "sport":         3,
    "tech":          4,
}

ID2LABEL: dict[int, str] = {v: k for k, v in LABEL2ID.items()}

NUM_LABELS = len(LABEL2ID)


# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------

def get_gcs_artifacts_uri(cfg: DictConfig) -> str:
    """GCS URI for model artifacts bucket."""
    return f"gs://{cfg.environment.gcs_bucket_artifacts}"


def get_gcs_data_uri(cfg: DictConfig) -> str:
    """GCS URI for data bucket."""
    return f"gs://{cfg.environment.gcs_bucket_data}"


def get_vertex_pipeline_root(cfg: DictConfig) -> str:
    """Vertex AI Pipeline root GCS path."""
    return f"{get_gcs_artifacts_uri(cfg)}/pipeline-root"


def get_artifact_registry_uri(cfg: DictConfig) -> str:
    """Base URI for Artifact Registry images."""
    return (
        f"{cfg.environment.gcp_region}-docker.pkg.dev"
        f"/{cfg.environment.gcp_project}"
        f"/{cfg.environment.artifact_registry_repo}"
    )


def get_bq_results_full_table(cfg: DictConfig) -> str:
    """Fully qualified BQ results table: project.dataset.table"""
    return (
        f"{cfg.environment.gcp_project}"
        f".{cfg.environment.bq_dataset}"
        f".{cfg.environment.bq_results_table}"
    )