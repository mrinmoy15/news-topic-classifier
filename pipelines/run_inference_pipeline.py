# =============================================================================
# pipelines/run_inference_pipeline.py
# Compile and submit the inference pipeline to Vertex AI.
#
# Usage
# -----
# Submit for today's partition (default dev):
#   python pipelines/run_inference_pipeline.py
#
# Submit to a specific environment:
#   python pipelines/run_inference_pipeline.py environment=prd
#
# Reprocess a specific day partition (0-29):
#   python pipelines/run_inference_pipeline.py environment=prd day=5
#
# Disable caching (force re-run even if inputs are unchanged):
#   python pipelines/run_inference_pipeline.py +enable_caching=False
# =============================================================================

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

from news_topic_classifier.config import get_vertex_pipeline_root

logger = logging.getLogger(__name__)

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
COMPILED_PATH = str(PROJECT_ROOT / "pipelines" / "compiled" / "inference_pipeline.yaml")


def _compile(cfg: DictConfig) -> None:
    import os
    import sys
    from kfp import compiler

    trainer_image = (
        f"us-central1-docker.pkg.dev/{cfg.environment.gcp_project}"
        f"/{cfg.environment.artifact_registry_repo}/trainer:latest"
    )
    os.environ["TRAINER_IMAGE"] = trainer_image
    for mod in list(sys.modules):
        if mod.startswith("pipelines"):
            del sys.modules[mod]

    from pipelines.inference_pipeline import inference_pipeline

    Path(COMPILED_PATH).parent.mkdir(parents=True, exist_ok=True)
    compiler.Compiler().compile(
        pipeline_func=inference_pipeline,
        package_path=COMPILED_PATH,
    )
    logger.info("Inference pipeline compiled with image: %s", trainer_image)
    logger.info("Inference pipeline compiled -> %s", COMPILED_PATH)


def _build_parameter_values(cfg: DictConfig) -> dict:
    gcs_model_uri = (
        f"gs://{cfg.environment.gcs_bucket_artifacts}/models/bert-bbc-finetuned/"
    )
    return {
        "gcp_project":        cfg.environment.gcp_project,
        "gcs_bucket_data":    cfg.environment.gcs_bucket_data,
        "gcs_model_uri":      gcs_model_uri,
        "bq_dataset":         cfg.environment.bq_dataset,
        "source_table":       cfg.data.bq_source_table,
        "predictions_table":  "news_topic_classifier_predictions",
        "batch_size":         cfg.training.batch_size,
        "max_seq_length":     cfg.model.max_seq_length,
    }


@hydra.main(
    config_path=str(PROJECT_ROOT / "conf"),
    config_name="config",
    version_base=None,
)
def main(cfg: DictConfig) -> None:

    print("\n" + "=" * 80)
    print(f"BBC News Inference Pipeline  |  env={cfg.environment.name}")
    print("=" * 80)

    _compile(cfg)

    from google.cloud import aiplatform

    aiplatform.init(
        project=cfg.environment.gcp_project,
        location=cfg.environment.gcp_region,
    )

    job = aiplatform.PipelineJob(
        display_name="bbc-news-inference-pipeline",
        template_path=COMPILED_PATH,
        pipeline_root=get_vertex_pipeline_root(cfg),
        parameter_values=_build_parameter_values(cfg),
        enable_caching=cfg.get("enable_caching", False),
    )

    job.submit(service_account=cfg.environment.vertex_ai_sa)
    logger.info("Inference pipeline submitted. Job name: %s", job.display_name)
    logger.info("View at: https://console.cloud.google.com/vertex-ai/pipelines")


if __name__ == "__main__":
    main()
