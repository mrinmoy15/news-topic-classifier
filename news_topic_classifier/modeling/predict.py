from __future__ import annotations

import logging
from pathlib import Path
import os
from datetime import datetime, timezone
import mlflow
import numpy as np
from google.cloud import storage
from sklearn.metrics import accuracy_score, classification_report
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification
from transformers import BertTokenizerFast

from news_topic_classifier.dataset import BBCNewsDataset
from news_topic_classifier.modeling.train import download_splits, _setup_mlflow_tracking

logger = logging.getLogger(__name__)


# =============================================================================
# SECTION 1 — Model & tokenizer loader
# =============================================================================
def load_model_tokenizer(
    model_path: str,
    device: torch.device,
) -> tuple[BertForSequenceClassification, BertTokenizerFast]:
    """
    Load the fine-tuned model and tokenizer from a local checkpoint directory.

    Parameters
    ----------
    model_path : str
        Local path to the fine-tuned checkpoint directory
    device : torch.device
        Target device (cuda / cpu). Model is moved to device and set to
        eval mode before returning.

    Returns
    -------
    tuple[BertForSequenceClassification, BertTokenizerFast]
        (model, tokenizer) — model is on device and in eval mode.
    """
    model = BertForSequenceClassification.from_pretrained(
        model_path,
        local_files_only=True,
    )
    model = model.to(device)
    model.eval()

    tokenizer = BertTokenizerFast.from_pretrained(
        model_path,
        local_files_only=True,
    )

    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Fine-tuned model loaded from %s", model_path)
    logger.info("Total params    : %d", total_params)
    logger.info("id2label        : %s", model.config.id2label)

    return model, tokenizer


# =============================================================================
# SECTION 2 — Inference
# =============================================================================
def run_inference(
    model: BertForSequenceClassification,
    loader: DataLoader,
    device: torch.device,
    has_labels: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Run forward-pass inference over a DataLoader with no gradient computation.

    Parameters
    ----------
    model : BertForSequenceClassification
        Fine-tuned model in eval mode on the correct device.
    loader : DataLoader
        DataLoader for the split to run inference on (typically test).
    device : torch.device
        Target device (cuda / cpu).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray | None]
        (preds, probs, labels) — shapes (N,), (N, num_labels), (N,) or None.
        preds  : predicted class index per sample.
        probs  : softmax probabilities over all classes per sample.
        labels : ground-truth class indices, or None if not present in loader.
    """
    all_preds  = []
    all_probs  = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            if has_labels :
                all_labels.extend(batch["label"].numpy())

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            probs = torch.softmax(outputs.logits, dim=1)
            preds = torch.argmax(outputs.logits,  dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return (
        np.array(all_preds),
        np.array(all_probs),
        np.array(all_labels) if has_labels else None,
    )


# =============================================================================
# SECTION 4 — Metrics
# =============================================================================
def compute_metrics(
    preds: np.ndarray,
    labels: np.ndarray,
    id2label: dict[int, str],
) -> dict:
    """
    Compute overall accuracy and per-class precision/recall/f1.

    Parameters
    ----------
    preds : np.ndarray, shape (N,)
        Predicted class indices from run_inference().
    labels : np.ndarray, shape (N,)
        Ground-truth class indices.
    id2label : dict[int, str]
        Mapping from class index to class name — used as target_names
        in the classification report.

    Returns
    -------
    dict
        {
            "accuracy": float,
            "report":   dict   # sklearn classification_report output_dict=True
        }
        The report dict has per-class and macro/weighted averages,
        ready for per-class MLflow logging.
    """
    all_label_ids = sorted(id2label)
    target_names  = [id2label[i] for i in all_label_ids]

    acc    = accuracy_score(labels, preds)
    # Pass labels= so sklearn always reports all 5 classes even when
    # the sample doesn't contain every class (e.g. small test splits).
    report = classification_report(
        labels,
        preds,
        labels=all_label_ids,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )

    logger.info("accuracy: %.4f", acc)
    logger.info(
        "\n%s",
        classification_report(
            labels, preds,
            labels=all_label_ids,
            target_names=target_names,
            zero_division=0,
        ),
    )

    return {"accuracy": acc, "report": report}


# =============================================================================
# SECTION 5 — Save predictions
# =============================================================================
def save_predictions(
    preds: np.ndarray,
    probs: np.ndarray,
    labels: np.ndarray | None,
    id2label: dict[int, str],
    gcs_output_dir: str,
    gcp_project: str,
) -> str:
    """
    Write predictions to a Parquet file and upload to GCS.

    Parameters
    ----------
    preds : np.ndarray, shape (N,)
        Predicted class indices.
    probs : np.ndarray, shape (N, num_labels)
        Softmax probabilities over all classes.
    labels : np.ndarray, shape (N,)
        Ground-truth class indices.
    id2label : dict[int, str]
        Mapping from class index to class name.
    gcs_output_dir : str
        GCS URI directory to write the predictions Parquet file into
    gcp_project : str
        GCP project ID used to authenticate the storage client.

    Returns
    -------
    str
        GCS URI of the written predictions Parquet file.
    """
    label_names = [id2label[i] for i in sorted(id2label)]

    # Build PyArrow table 
    pred_labels = [id2label[int(p)] for p in preds]
    confidence  = [float(probs[i][int(preds[i])]) for i in range(len(preds))]

    columns: dict = {
        "predicted":  pa.array(pred_labels, type=pa.string()),
        "confidence": pa.array(confidence,  type=pa.float32()),
    }
    if labels is not None:
        true_labels = [id2label[int(l)] for l in labels]
        columns["true_label"] = pa.array(true_labels, type=pa.string())
        columns["correct"]    = pa.array(
            [t == p for t, p in zip(true_labels, pred_labels)], type=pa.bool_()
        )
    for idx, name in enumerate(label_names):
        columns[f"prob_{name}"] = pa.array(
            [float(probs[i][idx]) for i in range(len(preds))],
            type=pa.float32(),
        )

    table = pa.table(columns)

    # Write locally then upload to GCS
    local_dir  = "/tmp" if os.getenv("CLOUD_ML_PROJECT_ID") else "data/processed"
    local_path = Path(local_dir) / "predictions.parquet"
    pq.write_table(table, str(local_path))

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    gcs_dir = gcs_output_dir.rstrip("/")
    gcs_uri = f"{gcs_dir}/{ts}/predictions.parquet"

    path_no_scheme = gcs_uri.replace("gs://", "")
    bucket_name = path_no_scheme.split("/")[0]
    blob_name = "/".join(path_no_scheme.split("/")[1:])

    client = storage.Client(project=gcp_project)
    client.bucket(bucket_name).blob(blob_name).upload_from_filename(str(local_path))

    local_path.unlink()
    logger.info("Predictions saved to %s (%d rows)", gcs_uri, len(table))

    return gcs_uri


# =============================================================================
# SECTION 6 — Predict orchestrator
# =============================================================================
def predict(
    cfg: DictConfig,
    model_path: str,
    test_loader: DataLoader,
    device: torch.device,
    gcs_output_dir: str,
) -> tuple[dict, str]:
    """
    Full inference pipeline with MLflow tracking.

    Flow
    ----
    1. Load fine-tuned model + tokenizer from model_path.
    2. Run inference on test_loader.
    3. Compute accuracy and per-class metrics.
    4. Save predictions Parquet to GCS.
    5. Log all metrics and the predictions URI to MLflow.

    Parameters
    ----------
    cfg : DictConfig
        Full Hydra config — reads project.name, environment.mlflow.tracking_uri,
        environment.gcp_project.
    model_path : str
        Local path to the fine-tuned checkpoint directory. GCS download
        is handled upstream before calling this function.
    test_loader : DataLoader
        Test DataLoader — built from the test split Parquet file.
    device : torch.device
        Target device (cuda / cpu).
    gcs_output_dir : str
        GCS URI directory for predictions output

    Returns
    -------
    tuple[dict, str]
        (metrics, gcs_predictions_uri)
        metrics              : {"accuracy": float, "report": dict}
        gcs_predictions_uri  : GCS URI of the saved predictions Parquet.
    """
    # MLflow setup
    tracking_uri = OmegaConf.select(
        cfg, "environment.mlflow.tracking_uri",
        default="sqlite:///mlflow.db"
    )
    _setup_mlflow_tracking(tracking_uri)
    mlflow.set_experiment(cfg.project.name)

    # Load model
    model, _ = load_model_tokenizer(model_path, device)
    id2label  = model.config.id2label

    with mlflow.start_run():
        mlflow.log_params({
            "model_path":    model_path,
            "num_test_samples": len(test_loader.dataset),
        })

        # Inference 
        preds, probs, labels = run_inference(model, test_loader, device)

        # Metrics (only when ground-truth labels are available) 
        metrics: dict = {}
        if labels is not None:
            metrics = compute_metrics(preds, labels, id2label)
            mlflow.log_metric("test_accuracy", metrics["accuracy"])
            for class_name, scores in metrics["report"].items():
                if isinstance(scores, dict):
                    mlflow.log_metrics({
                        f"{class_name}_precision": scores["precision"],
                        f"{class_name}_recall":    scores["recall"],
                        f"{class_name}_f1":        scores["f1-score"],
                    })
        else:
            logger.info("No labels in loader — skipping metrics computation.")

        # Save predictions
        gcs_predictions_uri = save_predictions(
            preds=preds,
            probs=probs,
            labels=labels,
            id2label=id2label,
            gcs_output_dir=gcs_output_dir,
            gcp_project=cfg.environment.gcp_project,
        )

        mlflow.log_param("gcs_predictions_uri", gcs_predictions_uri)
        logger.info("Predict complete. Predictions at %s", gcs_predictions_uri)

    return metrics, gcs_predictions_uri


# =============================================================================
# SECTION 7 — Smoke test
# =============================================================================
if __name__ == "__main__":

    import os

    import hydra
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader

    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

    @hydra.main(
        config_path=str(PROJECT_ROOT / "conf"),
        config_name="config",
        version_base=None,
    )
    def main(cfg: DictConfig) -> None:

        print("\n" + "=" * 80)
        print("predict.py — smoke test")
        print("=" * 80)

        # Redirect MLflow to SQLite — GCS tracking URI requires a hosted server
        mlflow_db = "/tmp/mlflow.db" if os.getenv("CLOUD_ML_PROJECT_ID") else str(PROJECT_ROOT / "mlflow.db")
        cfg = OmegaConf.merge(
            cfg,
            {"environment": {"mlflow": {"tracking_uri": f"sqlite:///{mlflow_db}"}}},
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")

        local_dir = (
            "/tmp/"
            if os.getenv("CLOUD_ML_PROJECT_ID")
            else str(PROJECT_ROOT / "data" / "processed")
        )

        # Test split
        if cfg.data.gcs_splits_dir:
            _, _, test_path = download_splits(
                gcs_processed_dir=cfg.data.gcs_splits_dir,
                local_dir=local_dir,
                gcp_project=cfg.environment.gcp_project,
            )
        else:
            test_path = str(Path(local_dir) / "test.parquet")

        # Fine-tuned model & tokenizer
        if os.getenv("CLOUD_ML_PROJECT_ID"):
            from news_topic_classifier.modeling.train import download_base_model
            model_path = download_base_model(
                gcs_model_uri=f"gs://{cfg.environment.gcs_bucket_artifacts}/models/bert-bbc-finetuned/",
                local_dir="/tmp/finetuned-model",
                gcp_project=cfg.environment.gcp_project,
            )
        else:
            model_path = str(PROJECT_ROOT / "models" / "bert-bbc-finetuned")

        tokenizer = BertTokenizerFast.from_pretrained(model_path, local_files_only=True)

        # Test DataLoader
        test_dataset = BBCNewsDataset(
            local_parquet_path=test_path,
            tokenizer=tokenizer,
            max_length=cfg.model.max_seq_length,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=False,
            num_workers=0,
        )
        print(f"Test batches  : {len(test_loader)}")
        print(f"Test samples  : {len(test_dataset)}")

        # Predictions output dir
        gcs_output_dir = (
            f"gs://{cfg.environment.gcs_bucket_data}/data/predictions/"
        )

        # Run predict pipeline 
        metrics, gcs_predictions_uri = predict(
            cfg=cfg,
            model_path=model_path,
            test_loader=test_loader,
            device=device,
            gcs_output_dir=gcs_output_dir,
        )

        print(f"\nTest accuracy : {metrics['accuracy']:.4f}")
        print(f"Predictions saved to  : {gcs_predictions_uri}")
        print("Smoke test passed.")

    main()
