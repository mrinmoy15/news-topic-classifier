import os

from kfp import dsl

_TRAINER_IMAGE = os.environ.get(
    "TRAINER_IMAGE",
    "us-central1-docker.pkg.dev/cs-cdwp-data-dev2188/news-topic-classifier/trainer:latest",
)


@dsl.component(base_image=_TRAINER_IMAGE)
def register_model_component(
    gcp_project: str,
    gcp_region: str,
    gcs_model_uri: str,
    environment_name: str,
    display_name: str = "bert-bbc-news-classifier",
    serving_container_uri: str = (
        "us-docker.pkg.dev/vertex-ai/prediction/pytorch-cpu.2-2:latest"
    ),
) -> str:
    """
    KFP component — register the fine-tuned model to Vertex AI Model Registry.

    Runs after evaluate_component so the model is only registered once the
    full pipeline (training + evaluation) has completed successfully.

    Parameters
    ----------
    gcp_project : str
        GCP project ID.
    gcp_region : str
        GCP region e.g. ``us-central1``.
    gcs_model_uri : str
        GCS URI of the fine-tuned model directory from train_component.
    environment_name : str
        Environment label (dev / pp / prd) — stored as a model label.
    display_name : str
        Display name in Model Registry. Each pipeline run creates a new
        version under this name; Vertex AI manages version history.
    serving_container_uri : str
        Docker image URI for online prediction. Defaults to the pre-built
        Vertex AI PyTorch CPU container. Override with a custom API image
        once api/Dockerfile is built and pushed.

    Returns
    -------
    str
        Fully-qualified Vertex AI Model resource name:
        ``projects/<num>/locations/<region>/models/<id>``.
    """
    from datetime import datetime, timezone

    from google.cloud import aiplatform

    aiplatform.init(project=gcp_project, location=gcp_region)

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    model = aiplatform.Model.upload(
        display_name=display_name,
        artifact_uri=gcs_model_uri,
        serving_container_image_uri=serving_container_uri,
        serving_container_predict_route="/predict",
        serving_container_health_route="/health",
        serving_container_ports=[8080],
        description=(
            "BERT-base-uncased fine-tuned on BBC News for 5-class topic "
            "classification (business / entertainment / politics / sport / tech)."
        ),
        version_description=f"Pipeline run {ts} | env={environment_name}",
        labels={
            "environment":  environment_name,
            "base_model":   "bert-base-uncased",
            "task":         "text-classification",
            "dataset":      "bbc-news",
        },
        sync=True,
    )

    print(f"Model registered: {model.resource_name}")
    print(f"Version ID      : {model.version_id}")

    return model.resource_name
