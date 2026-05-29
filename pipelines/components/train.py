import os
from typing import NamedTuple

from kfp import dsl

_TRAINER_IMAGE = os.environ.get(
    "TRAINER_IMAGE",
    "us-central1-docker.pkg.dev/cs-cdwp-data-dev2188/news-topic-classifier/trainer:latest",
)


class TrainOutputs(NamedTuple):
    gcs_model_uri: str
    mlflow_run_id: str


@dsl.component(base_image=_TRAINER_IMAGE)
def train_component(
    gcp_project: str,
    gcs_bucket_artifacts: str,
    gcs_splits_dir: str,
    mlflow_tracking_uri: str,
    project_name: str,
    bert_base_model: str,
    num_labels: int,
    max_seq_length: int,
    epochs: int,
    batch_size: int,
    lr: float,
    warmup_steps: int,
    weight_decay: float,
    early_stopping_patience: int,
) -> TrainOutputs:
    """
    KFP component — fine-tune BERT on the BBC News train/val splits.

    Thin wrapper around news_topic_classifier.modeling.train.train().
    Downloads splits from GCS, builds model and dataloaders, runs the
    training loop with MLflow tracking, uploads the best checkpoint to GCS.

    Parameters
    ----------
    gcp_project : str
        GCP project ID.
    gcs_bucket_artifacts : str
        GCS bucket name for model artefact upload.
    gcs_splits_dir : str
        GCS directory URI from preprocess_component containing
        train.parquet, val.parquet, test.parquet.
    mlflow_tracking_uri : str
        MLflow tracking URI — GCS path in dev/pp/prd.
    project_name : str
        MLflow experiment name.
    bert_base_model : str
        Base model name e.g. ``bert-base-uncased``.
    num_labels : int
        Number of output classes (5 for BBC News).
    max_seq_length : int
        Maximum tokeniser sequence length (512).
    epochs : int
        Maximum number of training epochs.
    batch_size : int
        DataLoader batch size for train and val.
    lr : float
        AdamW peak learning rate.
    warmup_steps : int
        Number of linear warmup steps for the scheduler.
    weight_decay : float
        AdamW weight decay.
    early_stopping_patience : int
        Epochs without val_acc improvement before stopping.

    Returns
    -------
    str
        GCS URI of the uploaded fine-tuned model directory — passed to
        predict_component as gcs_model_uri.
    """
    import os
    from pathlib import Path
    import torch
    from omegaconf import OmegaConf

    from news_topic_classifier.config import ID2LABEL, LABEL2ID
    from news_topic_classifier.modeling.bert_classifier import build_model
    from news_topic_classifier.modeling.train import (
        build_dataloaders,
        download_base_model,
        download_splits,
        train,
    )
    from transformers import BertTokenizerFast

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build a minimal DictConfig from component parameters
    cfg = OmegaConf.create({
        "project":     {"name": project_name},
        "environment": {
            "gcp_project": gcp_project,
            "gcs_bucket_artifacts": gcs_bucket_artifacts,
            "mlflow": {"tracking_uri": mlflow_tracking_uri},
        },
        "model": {
            "bert_base_model": bert_base_model,
            "num_labels": num_labels,
            "max_seq_length":  max_seq_length,
        },
        "training": {
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "warmup_steps": warmup_steps,
            "weight_decay": weight_decay,
            "early_stopping_patience": early_stopping_patience,
        },
    })

    # Download base model from GCS
    gcs_base_model_uri = f"gs://{gcs_bucket_artifacts}/models/base-models/{bert_base_model}/"
    local_model_path = download_base_model(
        gcs_model_uri=gcs_base_model_uri,
        local_dir="/tmp/base-model",
        gcp_project=gcp_project,
    )

    # Download splits from GCS
    train_path, val_path, _ = download_splits(
        gcs_processed_dir=gcs_splits_dir,
        local_dir="/tmp/",
        gcp_project=gcp_project,
    )

    # Tokenizer + DataLoaders
    tokenizer = BertTokenizerFast.from_pretrained(
        local_model_path, local_files_only=True
    )
    train_loader, val_loader, _ = build_dataloaders(
        train_path, val_path, val_path, tokenizer, cfg
    )

    # Model
    model = build_model(
        model_path=local_model_path,
        num_labels=num_labels,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    ).to(device)

    # Train
    from collections import namedtuple

    _, gcs_model_uri, mlflow_run_id = train(
        cfg=cfg,
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        save_path="/tmp/bert-bbc-finetuned",
    )

    TrainOutputs = namedtuple("TrainOutputs", ["gcs_model_uri", "mlflow_run_id"])
    return TrainOutputs(gcs_model_uri=gcs_model_uri, mlflow_run_id=mlflow_run_id)
