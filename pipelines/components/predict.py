from __future__ import annotations

from kfp import dsl


@dsl.component(
    base_image="us-central1-docker.pkg.dev/cs-cdwp-data-dev2188/news-topic-classifier/trainer:latest",
)
def predict_component(
    gcp_project: str,
    gcs_bucket_data: str,
    gcs_splits_dir: str,
    gcs_model_uri: str,
    mlflow_tracking_uri: str,
    project_name: str,
    max_seq_length: int,
    batch_size: int,
) -> str:
    """
    KFP component — run inference on the test split and save predictions to GCS.

    Thin wrapper around news_topic_classifier.modeling.predict.predict().
    Downloads the fine-tuned model and test split from GCS, runs inference,
    computes metrics, and writes a predictions Parquet to GCS.

    Parameters
    ----------
    gcp_project : str
        GCP project ID.
    gcs_bucket_data : str
        GCS bucket name for predictions Parquet output.
    gcs_splits_dir : str
        GCS directory URI from preprocess_component containing test.parquet.
    gcs_model_uri : str
        GCS URI of the fine-tuned model directory from train_component.
    mlflow_tracking_uri : str
        MLflow tracking URI — GCS path in dev/pp/prd.
    project_name : str
        MLflow experiment name.
    max_seq_length : int
        Maximum tokeniser sequence length (512).
    batch_size : int
        DataLoader batch size for inference.

    Returns
    -------
    str
        GCS URI of the predictions Parquet file — passed to evaluate_component.
    """
    import torch
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader
    from transformers import BertTokenizerFast

    from news_topic_classifier.dataset import BBCNewsDataset
    from news_topic_classifier.modeling.predict import predict
    from news_topic_classifier.modeling.train import download_base_model, download_splits

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Build minimal config ──────────────────────────────────────────────────
    cfg = OmegaConf.create({
        "project":     {"name": project_name},
        "environment": {
            "gcp_project":     gcp_project,
            "gcs_bucket_data": gcs_bucket_data,
            "mlflow":          {"tracking_uri": mlflow_tracking_uri},
        },
    })

    # ── Download fine-tuned model from GCS ────────────────────────────────────
    local_model_path = download_base_model(
        gcs_model_uri=gcs_model_uri,
        local_dir="/tmp/finetuned-model",
        gcp_project=gcp_project,
    )

    # ── Download test split from GCS ──────────────────────────────────────────
    _, _, test_path = download_splits(
        gcs_processed_dir=gcs_splits_dir,
        local_dir="/tmp/",
        gcp_project=gcp_project,
    )

    # ── Test DataLoader ───────────────────────────────────────────────────────
    tokenizer = BertTokenizerFast.from_pretrained(
        local_model_path, local_files_only=True
    )
    test_dataset = BBCNewsDataset(
        local_parquet_path=test_path,
        tokenizer=tokenizer,
        max_length=max_seq_length,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    # ── Run predict pipeline ──────────────────────────────────────────────────
    gcs_output_dir = f"gs://{gcs_bucket_data}/data/predictions/"

    _, gcs_predictions_uri = predict(
        cfg=cfg,
        model_path=local_model_path,
        test_loader=test_loader,
        device=device,
        gcs_output_dir=gcs_output_dir,
    )

    return gcs_predictions_uri
