"""
Register a fine-tuned BERT model to Vertex AI Model Registry.

Usage
-----
# Register the default fine-tuned model for a given environment:
    python scripts/register_model.py --environment dev

# Register a specific GCS model URI:
    python scripts/register_model.py --environment dev \
        --gcs-model-uri gs://cs-cdwp-data-dev2188-model-artifacts/models/bert-bbc-finetuned/

# Pin a custom display name or version description:
    python scripts/register_model.py --environment dev \
        --display-name "bert-bbc-v2" \
        --version-description "Trained on 2026-06-01 run, val_acc=0.97"

Notes
-----
- Requires GCP credentials: `gcloud auth application-default login`
- The model artifact must be at the GCS URI before calling this script.
- The environment's trainer image is used as the serving container by default.
  Override with --serving-container-uri when a dedicated API image is available.
- Each call creates a new *version* of the same display-name model resource
  in Model Registry (Vertex AI handles versioning automatically).
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone

from google.cloud import aiplatform

# ---------------------------------------------------------------------------
# Environment → GCP config mapping
# ---------------------------------------------------------------------------

_ENV_CONFIG = {
    "dev": {
        "project":          "cs-cdwp-data-dev2188",
        "region":           "us-central1",
        "artifacts_bucket": "cs-cdwp-data-dev2188-model-artifacts",
        "trainer_image":    "us-central1-docker.pkg.dev/cs-cdwp-data-dev2188/news-topic-classifier/trainer:latest",
    },
    "pp": {
        "project":          "cs-cdwp-data-pp2188",
        "region":           "us-central1",
        "artifacts_bucket": "cs-cdwp-data-pp2188-model-artifacts",
        "trainer_image":    "us-central1-docker.pkg.dev/cs-cdwp-data-pp2188/news-topic-classifier/trainer:latest",
    },
    "prd": {
        "project":          "cs-cdwp-data-prd2188",
        "region":           "us-central1",
        "artifacts_bucket": "cs-cdwp-data-prd2188-model-artifacts",
        "trainer_image":    "us-central1-docker.pkg.dev/cs-cdwp-data-prd2188/news-topic-classifier/trainer:latest",
    },
}

# Health-check and prediction route expected by Vertex AI online prediction.
_HEALTH_ROUTE   = "/health"
_PREDICT_ROUTE  = "/predict"
_CONTAINER_PORT = 8080


# ---------------------------------------------------------------------------
# Core registration function
# ---------------------------------------------------------------------------

def register_model(
    environment: str,
    gcs_model_uri: str | None,
    display_name: str,
    version_description: str,
    serving_container_uri: str | None = None,
) -> aiplatform.Model:
    """Upload the fine-tuned model artifact to Vertex AI Model Registry."""

    cfg = _ENV_CONFIG[environment]

    if gcs_model_uri is None:
        gcs_model_uri = (
            f"gs://{cfg['artifacts_bucket']}/models/bert-bbc-finetuned/"
        )

    if serving_container_uri is None:
        serving_container_uri = cfg["trainer_image"]

    print(f"\nRegistering model to Vertex AI Model Registry")
    print(f"  Environment  : {environment}")
    print(f"  Project      : {cfg['project']}")
    print(f"  Region       : {cfg['region']}")
    print(f"  Artifact URI : {gcs_model_uri}")
    print(f"  Display name : {display_name}")
    print(f"  Container    : {serving_container_uri}")
    print()

    aiplatform.init(
        project=cfg["project"],
        location=cfg["region"],
    )

    model = aiplatform.Model.upload(
        display_name=display_name,
        artifact_uri=gcs_model_uri,
        serving_container_image_uri=serving_container_uri,
        serving_container_predict_route=_PREDICT_ROUTE,
        serving_container_health_route=_HEALTH_ROUTE,
        serving_container_ports=[_CONTAINER_PORT],
        description=(
            "BERT-base-uncased fine-tuned on BBC News for 5-class topic "
            "classification (business / entertainment / politics / sport / tech)."
        ),
        version_description=version_description,
        labels={
            "environment": environment,
            "base_model":  "bert-base-uncased",
            "task":        "text-classification",
            "dataset":     "bbc-news",
        },
        sync=True,
    )

    print(f"Model registered successfully.")
    print(f"  Resource name : {model.resource_name}")
    print(f"  Version ID    : {model.version_id}")
    print(f"\nView in console:")
    print(
        f"  https://console.cloud.google.com/vertex-ai/models"
        f"?project={cfg['project']}"
    )

    return model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Register a fine-tuned BERT model to Vertex AI Model Registry.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--environment", "-e",
        choices=["dev", "pp", "prd"],
        default="dev",
        help="Target GCP environment (default: dev)",
    )
    parser.add_argument(
        "--gcs-model-uri",
        default=None,
        help=(
            "GCS URI of the fine-tuned model directory. "
            "Defaults to gs://<artifacts-bucket>/models/bert-bbc-finetuned/"
        ),
    )
    parser.add_argument(
        "--display-name",
        default="bert-bbc-news-classifier",
        help="Display name in Model Registry (default: bert-bbc-news-classifier)",
    )
    parser.add_argument(
        "--version-description",
        default=None,
        help=(
            "Human-readable description for this model version. "
            "Defaults to a timestamped string."
        ),
    )
    parser.add_argument(
        "--serving-container-uri",
        default=None,
        help=(
            "Serving container image URI. "
            "Defaults to the environment's trainer image from _ENV_CONFIG."
        ),
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    version_description = args.version_description or (
        f"Registered on {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
    )

    register_model(
        environment=args.environment,
        gcs_model_uri=args.gcs_model_uri,
        display_name=args.display_name,
        version_description=version_description,
        serving_container_uri=args.serving_container_uri,
    )


if __name__ == "__main__":
    main()
