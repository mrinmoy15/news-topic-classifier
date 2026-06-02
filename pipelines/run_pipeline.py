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
# =============================================================================

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

from news_topic_classifier.config import get_vertex_pipeline_root

logger = logging.getLogger(__name__)

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
COMPILED_PATH = str(PROJECT_ROOT / "pipelines" / "compiled" / "training_pipeline.yaml")


def _compile(cfg: DictConfig) -> None:
    import os
    import sys
    from kfp import compiler

    # Set TRAINER_IMAGE before importing components — decorators read it at import time.
    # Remove cached modules so decorators re-run with the correct image for this env.
    trainer_image = (
        f"us-central1-docker.pkg.dev/{cfg.environment.gcp_project}"
        f"/{cfg.environment.artifact_registry_repo}/trainer:latest"
    )
    os.environ["TRAINER_IMAGE"] = trainer_image
    for mod in list(sys.modules):
        if mod.startswith("pipelines"):
            del sys.modules[mod]

    from pipelines.training_pipeline import training_pipeline

    Path(COMPILED_PATH).parent.mkdir(parents=True, exist_ok=True)
    compiler.Compiler().compile(
        pipeline_func=training_pipeline,
        package_path=COMPILED_PATH,
    )
    logger.info("Pipeline compiled with image: %s", trainer_image)
    logger.info("Pipeline compiled -> %s", COMPILED_PATH)


def _build_parameter_values(cfg: DictConfig) -> dict:
    return {
        "gcp_project":             cfg.environment.gcp_project,
        "gcp_region":              cfg.environment.gcp_region,
        "gcs_bucket_data":         cfg.environment.gcs_bucket_data,
        "gcs_bucket_artifacts":    cfg.environment.gcs_bucket_artifacts,
        "bq_dataset":              cfg.environment.bq_dataset,
        "source_table":            cfg.data.bq_source_table,
        "text_col":                cfg.data.bq_text_column,
        "label_col":               cfg.data.bq_label_column,
        "title_col":               cfg.data.bq_title_column,
        "mlflow_tracking_uri":     cfg.environment.mlflow.tracking_uri,
        "project_name":            cfg.project.name,
        "environment_name":        cfg.environment.name,
        "bert_base_model":         cfg.model.bert_base_model,
        "num_labels":              cfg.model.num_labels,
        "max_seq_length":          cfg.model.max_seq_length,
        "epochs":                  cfg.training.epochs,
        "batch_size":              cfg.training.batch_size,
        "lr":                      cfg.training.lr,
        "warmup_steps":            cfg.training.warmup_steps,
        "weight_decay":            cfg.training.weight_decay,
        "early_stopping_patience": cfg.training.early_stopping_patience,
        "val_split":               cfg.training.val_split,
        "test_split":              cfg.training.test_split,
    }


@hydra.main(
    config_path=str(PROJECT_ROOT / "conf"),
    config_name="config",
    version_base=None,
)
def main(cfg: DictConfig) -> None:

    print("\n" + "=" * 80)
    print(f"BBC News Training Pipeline  |  env={cfg.environment.name}")
    print("=" * 80)

    _compile(cfg)

    from google.cloud import aiplatform

    aiplatform.init(
        project=cfg.environment.gcp_project,
        location=cfg.environment.gcp_region,
    )

    job = aiplatform.PipelineJob(
        display_name="bbc-news-training-pipeline",
        template_path=COMPILED_PATH,
        pipeline_root=get_vertex_pipeline_root(cfg),
        parameter_values=_build_parameter_values(cfg),
        enable_caching=True,
    )

    job.submit(service_account=cfg.environment.vertex_ai_sa)
    logger.info("Pipeline submitted. Job name: %s", job.display_name)
    logger.info("View at: https://console.cloud.google.com/vertex-ai/pipelines")


if __name__ == "__main__":
    main()
