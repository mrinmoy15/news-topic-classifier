from __future__ import annotations

# =============================================================================
# pipelines/run_pipeline.py
# Compile and submit the training pipeline to Vertex AI.
#
# Usage
# -----
# Submit to Vertex AI (default dev environment):
#   python pipelines/run_pipeline.py
#
# Submit to a different environment:
#   python pipelines/run_pipeline.py environment=pp
#   python pipelines/run_pipeline.py environment=prd
#
# Run locally with DockerRunner (for testing):
#   python pipelines/run_pipeline.py runner=local
# =============================================================================

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from news_topic_classifier.config import get_vertex_pipeline_root

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COMPILED_PATH = str(PROJECT_ROOT / "pipelines" / "compiled" / "training_pipeline.yaml")


def _compile() -> None:
    from kfp import compiler
    from pipelines.training_pipeline import training_pipeline

    Path(COMPILED_PATH).parent.mkdir(parents=True, exist_ok=True)
    compiler.Compiler().compile(
        pipeline_func=training_pipeline,
        package_path=COMPILED_PATH,
    )
    logger.info("Pipeline compiled -> %s", COMPILED_PATH)


def _build_parameter_values(cfg: DictConfig) -> dict:
    return {
        "gcp_project":              cfg.environment.gcp_project,
        "gcs_bucket_data":          cfg.environment.gcs_bucket_data,
        "gcs_bucket_artifacts":     cfg.environment.gcs_bucket_artifacts,
        "bq_dataset":               cfg.environment.bq_dataset,
        "source_table":             cfg.data.bq_source_table,
        "text_col":                 cfg.data.bq_text_column,
        "label_col":                cfg.data.bq_label_column,
        "title_col":                cfg.data.bq_title_column,
        "mlflow_tracking_uri":      cfg.environment.mlflow.tracking_uri,
        "project_name":             cfg.project.name,
        "environment_name":         cfg.environment.name,
        "bert_base_model":          cfg.model.bert_base_model,
        "num_labels":               cfg.model.num_labels,
        "max_seq_length":           cfg.model.max_seq_length,
        "epochs":                   cfg.training.epochs,
        "batch_size":               cfg.training.batch_size,
        "lr":                       cfg.training.lr,
        "warmup_steps":             cfg.training.warmup_steps,
        "weight_decay":             cfg.training.weight_decay,
        "early_stopping_patience":  cfg.training.early_stopping_patience,
        "val_split":                cfg.training.val_split,
        "test_split":               cfg.training.test_split,
    }


def _submit_vertex(cfg: DictConfig, parameter_values: dict) -> None:
    from google.cloud import aiplatform

    aiplatform.init(
        project=cfg.environment.gcp_project,
        location=cfg.environment.gcp_region,
    )

    job = aiplatform.PipelineJob(
        display_name="bbc-news-training-pipeline",
        template_path=COMPILED_PATH,
        pipeline_root=get_vertex_pipeline_root(cfg),
        parameter_values=parameter_values,
        enable_caching=True,
    )

    job.submit(service_account=cfg.environment.vertex_ai_sa)
    logger.info("Pipeline submitted. Job name: %s", job.display_name)
    logger.info("View at: https://console.cloud.google.com/vertex-ai/pipelines")


def _run_local(cfg: DictConfig, parameter_values: dict) -> None:
    import os
    from kfp import local
    from pipelines.training_pipeline import training_pipeline

    # Mount local GCP credentials into each component container
    gcloud_dir = str(Path.home() / ".config" / "gcloud")

    local.init(
        runner=local.DockerRunner(
            docker_client_kwargs={
                "volumes": {
                    gcloud_dir: {
                        "bind": "/root/.config/gcloud",
                        "mode": "ro",
                    }
                },
                "environment": {
                    "GOOGLE_APPLICATION_CREDENTIALS": (
                        "/root/.config/gcloud/application_default_credentials.json"
                    ),
                    "GOOGLE_CLOUD_PROJECT": cfg.environment.gcp_project,
                },
            }
        )
    )

    training_pipeline(**parameter_values)
    logger.info("Local pipeline run complete.")


@hydra.main(
    config_path=str(PROJECT_ROOT / "conf"),
    config_name="config",
    version_base=None,
)
def main(cfg: DictConfig) -> None:

    runner = OmegaConf.select(cfg, "runner", default="vertex")

    print("\n" + "=" * 80)
    print(f"BBC News Training Pipeline  |  env={cfg.environment.name}  |  runner={runner}")
    print("=" * 80)

    _compile()

    parameter_values = _build_parameter_values(cfg)

    if runner == "local":
        _run_local(cfg, parameter_values)
    else:
        _submit_vertex(cfg, parameter_values)


if __name__ == "__main__":
    main()
