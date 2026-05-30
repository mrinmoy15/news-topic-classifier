from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import mlflow
import torch
from tqdm import tqdm
from google.cloud import storage
from omegaconf import DictConfig, OmegaConf
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast, get_linear_schedule_with_warmup

from news_topic_classifier.dataset import BBCNewsDataset
from news_topic_classifier.modeling.bert_classifier import build_model

logger = logging.getLogger(__name__)


def _setup_mlflow_tracking(tracking_uri: str) -> None:
    """Set MLflow tracking URI, fetching a GCP OIDC token for Cloud Run endpoints."""
    if ".run.app" in tracking_uri:
        import time
        import google.auth.transport.requests
        import google.oauth2.id_token
        auth_req = google.auth.transport.requests.Request()
        for attempt in range(4):
            try:
                token = google.oauth2.id_token.fetch_id_token(auth_req, tracking_uri)
                os.environ["MLFLOW_TRACKING_TOKEN"] = token
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
    mlflow.set_tracking_uri(tracking_uri)


# =============================================================================
# SECTION 1 — GCS download
# =============================================================================
def download_splits(
    gcs_processed_dir: str,
    local_dir: str,
    gcp_project: str,
) -> tuple[str, str, str]:
    """
    Download train/val/test Parquet splits from a GCS directory to local disk.

    On Vertex AI pipelines, pass local_dir="/tmp/" — that is the writable
    scratch space available to each component container.

    Parameters
    ----------
    gcs_processed_dir : str
        GCS URI of the directory produced by features.py
        Expected blobs: train.parquet, val.parquet, test.parquet
    local_dir : str
        Local directory to write the downloaded files into.
    gcp_project : str
        GCP project ID used to authenticate the storage client.

    Returns
    -------
    tuple[str, str, str]
        (train_path, val_path, test_path) — absolute local file paths,
        ready to pass directly into build_dataloaders().
    """
    local_dir_path = Path(local_dir)
    local_dir_path.mkdir(parents=True, exist_ok=True)

    gcs_dir = gcs_processed_dir.rstrip("/")
    client  = storage.Client(project=gcp_project)

    local_paths = []
    for split in ("train", "val", "test"):
        gcs_uri    = f"{gcs_dir}/{split}.parquet"
        local_path = local_dir_path / f"{split}.parquet"

        path_no_scheme = gcs_uri.replace("gs://", "")
        bucket_name    = path_no_scheme.split("/")[0]
        blob_name      = "/".join(path_no_scheme.split("/")[1:])

        bucket = client.bucket(bucket_name)
        blob   = bucket.blob(blob_name)
        blob.download_to_filename(str(local_path))

        logger.info("Downloaded %s -> %s", gcs_uri, local_path)
        local_paths.append(str(local_path))

    return local_paths[0], local_paths[1], local_paths[2]


def download_base_model(
    gcs_model_uri: str,
    local_dir: str,
    gcp_project: str,
) -> str:
    """
    Download a HuggingFace model directory from GCS to local disk.

    Lists all blobs under the GCS prefix and downloads each file,
    preserving the directory structure under local_dir.

    Parameters
    ----------
    gcs_model_uri : str
        GCS URI prefix of the model directory
        Expected blobs: config.json, model.safetensors, tokenizer.json,
        tokenizer_config.json, vocab.txt, etc.
    local_dir : str
        Local directory to download the model files into.
    gcp_project : str
        GCP project ID used to authenticate the storage client.

    Returns
    -------
    str
        Absolute local path of the downloaded model directory,
        ready to pass into BertForSequenceClassification.from_pretrained().
    """
    local_dir_path = Path(local_dir)
    local_dir_path.mkdir(parents=True, exist_ok=True)

    gcs_prefix     = gcs_model_uri.replace("gs://", "").rstrip("/")
    bucket_name    = gcs_prefix.split("/")[0]
    prefix         = "/".join(gcs_prefix.split("/")[1:]) + "/"

    client = storage.Client(project=gcp_project)
    blobs  = list(client.list_blobs(bucket_name, prefix=prefix))

    for blob in blobs:
        # Strip the GCS prefix to get the relative filename
        relative_path = blob.name[len(prefix):]
        if not relative_path:
            continue

        local_file = local_dir_path / relative_path
        local_file.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(local_file))
        logger.info("Downloaded %s -> %s", blob.name, local_file)

    logger.info("Base model downloaded to %s (%d files)", local_dir_path, len(blobs))

    # If gsutil cp -r created a single nested subdirectory (e.g. bert-base-uncased/),
    # return that subdirectory so from_pretrained() finds the model files directly.
    subdirs = [p for p in local_dir_path.iterdir() if p.is_dir()]
    if len(subdirs) == 1:
        return str(subdirs[0])
    return str(local_dir_path)


def upload_model_to_gcs(
    local_dir: str,
    gcs_model_uri: str,
    gcp_project: str,
) -> str:
    """
    Upload a local HuggingFace model directory to GCS.

    Walks all files under local_dir and uploads each one under the
    gcs_model_uri prefix, preserving relative paths.

    Parameters
    ----------
    local_dir : str
        Local directory containing the saved model files
        (config.json, model.safetensors, tokenizer files, etc.).
    gcs_model_uri : str
        Destination GCS URI prefix,
        or the value of the ``AIP_MODEL_DIR`` env var on Vertex AI.
    gcp_project : str
        GCP project ID used to authenticate the storage client.

    Returns
    -------
    str
        The gcs_model_uri that was written to — passed to downstream
        pipeline components as the fine-tuned model artifact path.
    """
    local_dir_path = Path(local_dir)

    gcs_prefix  = gcs_model_uri.replace("gs://", "").rstrip("/")
    bucket_name = gcs_prefix.split("/")[0]
    prefix      = "/".join(gcs_prefix.split("/")[1:])

    client = storage.Client(project=gcp_project)
    bucket = client.bucket(bucket_name)

    uploaded = 0
    for local_file in local_dir_path.rglob("*"):
        if not local_file.is_file():
            continue
        relative  = local_file.relative_to(local_dir_path)
        blob_name = f"{prefix}/{relative}".lstrip("/")
        bucket.blob(blob_name).upload_from_filename(str(local_file))
        logger.info("Uploaded %s -> gs://%s/%s", local_file.name, bucket_name, blob_name)
        uploaded += 1

    logger.info("Model uploaded to %s (%d files)", gcs_model_uri, uploaded)
    return gcs_model_uri


# =============================================================================
# SECTION 2 — DataLoader factory
# =============================================================================
def build_dataloaders(
    train_path: str,
    val_path: str,
    test_path: str,
    tokenizer,
    cfg: DictConfig,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Wrap train/val/test Parquet files in BBCNewsDatasets and DataLoaders.

    The GCS → local download is handled upstream (pipeline component).
    This function only constructs the in-process objects.

    Parameters
    ----------
    train_path : str
        Local path to the training split Parquet file.
    val_path : str
        Local path to the validation split Parquet file.
    test_path : str
        Local path to the test split Parquet file.
    tokenizer : PreTrainedTokenizer
        Loaded tokenizer — shared across all three splits.
    cfg : DictConfig
        Hydra config — reads model.max_seq_length, training.batch_size.

    Returns
    -------
    tuple[DataLoader, DataLoader, DataLoader]
        (train_loader, val_loader, test_loader)
    """
    ds_kwargs = dict(
        tokenizer=tokenizer,
        max_length=cfg.model.max_seq_length,
    )

    train_dataset = BBCNewsDataset(local_parquet_path=train_path, **ds_kwargs)
    val_dataset   = BBCNewsDataset(local_parquet_path=val_path,   **ds_kwargs)
    test_dataset  = BBCNewsDataset(local_parquet_path=test_path,  **ds_kwargs)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=0,
    )

    logger.info(
        "DataLoaders ready — train=%d batches, val=%d batches, test=%d batches",
        len(train_loader), len(val_loader), len(test_loader),
    )

    return train_loader, val_loader, test_loader


# =============================================================================
# SECTION 3 — Optimizer & scheduler
# =============================================================================
def build_optimizer_scheduler(
    model,
    total_steps: int,
    cfg: DictConfig,
) -> tuple[AdamW, object]:
    """
    Build AdamW optimizer and linear warmup-decay scheduler.

    Parameters
    ----------
    model : BertForSequenceClassification
        The model whose parameters will be optimized.
    total_steps : int
        Total training steps = len(train_loader) * epochs.
        Computed externally so the caller controls the epoch budget.
    cfg : DictConfig
        Reads training.lr, training.weight_decay, training.warmup_steps.

    Returns
    -------
    tuple[AdamW, LambdaLR]
    """
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.training.warmup_steps,
        num_training_steps=total_steps,
    )

    logger.info(
        "Optimizer: AdamW(lr=%s, weight_decay=%s)",
        cfg.training.lr, cfg.training.weight_decay,
    )
    logger.info(
        "Scheduler: linear warmup %d steps -> decay over %d total steps",
        cfg.training.warmup_steps, total_steps,
    )

    return optimizer, scheduler


# =============================================================================
# SECTION 4 — Training step
# =============================================================================
def train_epoch(
    model,
    loader: DataLoader,
    optimizer: AdamW,
    scheduler,
    device: torch.device,
) -> tuple[float, float]:
    """
    Run one full training epoch.

    Parameters
    ----------
    model : BertForSequenceClassification
        Model in training mode — dropout is active.
    loader : DataLoader
        Training DataLoader (shuffled).
    optimizer : AdamW
        AdamW optimizer instance.
    scheduler : LambdaLR
        Linear warmup-decay scheduler.
    device : torch.device
        Target device (cuda / cpu).

    Returns
    -------
    tuple[float, float]
        (avg_loss, accuracy) over the full epoch.
    """
    model.train()

    total_loss = 0.0
    correct = 0
    total_samples = 0

    for batch in tqdm(loader, desc="train", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        loss   = outputs.loss
        logits = outputs.logits

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    return total_loss / len(loader), correct / total_samples


# =============================================================================
# SECTION 5 — Evaluation step
# =============================================================================
def eval_epoch(
    model,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """
    Run one full evaluation pass with no gradient updates.

    Parameters
    ----------
    model : BertForSequenceClassification
        Model in eval mode — dropout is disabled.
    loader : DataLoader
        Validation or test DataLoader (not shuffled).
    device : torch.device
        Target device (cuda / cpu).

    Returns
    -------
    tuple[float, float]
        (avg_loss, accuracy) over the full loader.
    """
    model.eval()

    total_loss = 0.0
    correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(
                input_ids = input_ids,
                attention_mask = attention_mask,
                labels = labels,
            )

            total_loss += outputs.loss.item()
            preds = torch.argmax(outputs.logits, dim=1)
            correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

    return total_loss / len(loader), correct / total_samples


# =============================================================================
# SECTION 6 — Training loop with MLflow
# =============================================================================
def train(
    cfg: DictConfig,
    model,
    tokenizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    save_path: str,
) -> float:
    """
    Full training loop with MLflow experiment tracking.

    Flow
    ----
    1. Set MLflow tracking URI + experiment from config.
    2. Log all hyperparameters as MLflow params.
    3. Train for cfg.training.epochs with per-epoch val evaluation.
    4. Track best val_acc; deepcopy best weights; apply early stopping.
    5. Restore best weights → save model locally + log artifacts to MLflow.

    Parameters
    ----------
    cfg : DictConfig
        Full Hydra config — reads training.*, model.*, project.name,
        and environment.mlflow.tracking_uri.
    model : BertForSequenceClassification
        Model already moved to the correct device.
    tokenizer : PreTrainedTokenizer
        Tokenizer saved alongside the model checkpoint so the saved
        directory is self-contained for inference.
    train_loader : DataLoader
        Training DataLoader.
    val_loader : DataLoader
        Validation DataLoader.
    device : torch.device
        Target device (cuda / cpu).
    save_path : str
        Local directory where best model weights are written via
        model.save_pretrained() before being logged to MLflow.

    Returns
    -------
    tuple[float, str]
        (best_val_acc, gcs_model_uri) — accuracy and the GCS path where
        the fine-tuned model was uploaded, ready to pass to predict.py.
    """
    # MLflow setup
    tracking_uri = OmegaConf.select(
        cfg, "environment.mlflow.tracking_uri",
        default="sqlite:///mlflow.db"
    )
    _setup_mlflow_tracking(tracking_uri)
    mlflow.set_experiment(cfg.project.name)

    # Optimizer & scheduler
    total_steps = len(train_loader) * cfg.training.epochs
    optimizer, scheduler = build_optimizer_scheduler(model, total_steps, cfg)

    # Training state
    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = save_dir / "best_checkpoint.pt"

    best_val_acc = 0.0
    no_improve   = 0

    with mlflow.start_run() as run:
        mlflow.log_params({
            "bert_base_model": cfg.model.bert_base_model,
            "num_labels": cfg.model.num_labels,
            "max_seq_length": cfg.model.max_seq_length,
            "epochs": cfg.training.epochs,
            "batch_size": cfg.training.batch_size,
            "lr": cfg.training.lr,
            "warmup_steps": cfg.training.warmup_steps,
            "weight_decay": cfg.training.weight_decay,
            "early_stopping_patience": cfg.training.early_stopping_patience,
        })

        logger.info("Starting training on %s", device)
        logger.info(
            "%-8s %-14s %-14s %-14s %-14s %s",
            "Epoch", "Train Loss", "Train Acc", "Val Loss", "Val Acc", "Time",
        )
        logger.info("-" * 70)

        for epoch in range(cfg.training.epochs):
            start = time.time()

            train_loss, train_acc = train_epoch(
                model, train_loader, optimizer, scheduler, device
            )
            val_loss, val_acc = eval_epoch(model, val_loader, device)

            elapsed = time.time() - start

            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                },
                step = epoch + 1,
            )

            tag = ""
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), checkpoint_path)
                no_improve = 0
                tag = "  <- best"
            else:
                no_improve += 1

            logger.info(
                "%-8d %-14.4f %-14.4f %-14.4f %-14.4f %.1fs%s",
                epoch + 1, train_loss, train_acc, val_loss, val_acc, elapsed, tag,
            )

            if no_improve >= cfg.training.early_stopping_patience:
                logger.info("Early stopping triggered at epoch %d.", epoch + 1)
                break

        logger.info("-" * 70)
        logger.info("Best val accuracy: %.4f", best_val_acc)

        # Restore best weights
        model.load_state_dict(
            torch.load(checkpoint_path, map_location=device, weights_only=True)
        )
        mlflow.log_metric("best_val_acc", best_val_acc)

        # Save model locally + upload to GCS + log to MLflow
        model.save_pretrained(str(save_dir))
        tokenizer.save_pretrained(str(save_dir))

        # AIP_MODEL_DIR is set by Vertex AI; fall back to a fixed GCS path locally.
        gcs_model_uri = os.getenv("AIP_MODEL_DIR") or (
            f"gs://{cfg.environment.gcs_bucket_artifacts}/models/bert-bbc-finetuned/"
        )
        upload_model_to_gcs(str(save_dir), gcs_model_uri, cfg.environment.gcp_project)
        mlflow.log_artifacts(str(save_dir), artifact_path="model")
        mlflow.log_param("gcs_model_uri", gcs_model_uri)

        checkpoint_path.unlink()
        logger.info("Model saved to %s and uploaded to %s.", save_dir, gcs_model_uri)

    return best_val_acc, gcs_model_uri, run.info.run_id


# =============================================================================
# SECTION 7 — Smoke test
# =============================================================================
if __name__ == "__main__":

    import hydra
    from omegaconf import OmegaConf

    from news_topic_classifier.config import ID2LABEL, LABEL2ID

    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

    @hydra.main(
        config_path=str(PROJECT_ROOT / "conf"),
        config_name="config",
        version_base=None,
    )
    def main(cfg: DictConfig) -> None:

        print("\n" + "=" * 80)
        print("train.py — smoke test")
        print("=" * 80)

        # Redirect MLflow to local SQLite for smoke test — Hydra configs are
        # read-only so merge produces a new unfrozen config. Use an absolute path
        # so the DB lands in the project root regardless of Hydra's CWD change.
        mlflow_db = "/tmp/mlflow.db" if os.getenv("CLOUD_ML_PROJECT_ID") else str(PROJECT_ROOT / "mlflow.db")
        cfg = OmegaConf.merge(
            cfg,
            {"environment": {"mlflow": {"tracking_uri": f"sqlite:///{mlflow_db}"}}},
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")

        if os.getenv("CLOUD_ML_PROJECT_ID"):
            model_path = download_base_model(
                gcs_model_uri=f"gs://{cfg.environment.gcs_bucket_artifacts}/models/base-models/{cfg.model.bert_base_model}/",
                local_dir="/tmp/base-model",
                gcp_project=cfg.environment.gcp_project,
            )
        else:
            model_path = str(PROJECT_ROOT / "models" / "base-models" / cfg.model.bert_base_model)

        tokenizer = BertTokenizerFast.from_pretrained(model_path, local_files_only=True)

        # DataLoaders
        local_dir = (
            "/tmp/"
            if os.getenv("CLOUD_ML_PROJECT_ID")
            else str(PROJECT_ROOT / "data" / "processed")
        )

        if cfg.data.gcs_splits_dir:
            train_path, val_path, test_path = download_splits(
                gcs_processed_dir=cfg.data.gcs_splits_dir,
                local_dir=local_dir,
                gcp_project=cfg.environment.gcp_project,
            )
        else:
            train_path = str(Path(local_dir) / "train.parquet")
            val_path   = str(Path(local_dir) / "val.parquet")
            test_path  = str(Path(local_dir) / "test.parquet")

        # test_loader is kept for predict.py; train() only needs train + val
        train_loader, val_loader, _ = build_dataloaders(
            train_path, val_path, test_path, tokenizer, cfg
        )
        print(f"Train batches : {len(train_loader)}")
        print(f"Val batches   : {len(val_loader)}")

        # Model
        model = build_model(
            model_path=model_path,
            num_labels=cfg.model.num_labels,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        ).to(device)

        # Train
        save_path = (
            "/tmp/bert-bbc-finetuned"
            if os.getenv("CLOUD_ML_PROJECT_ID")
            else str(PROJECT_ROOT / "models" / "bert-bbc-finetuned")
        )

        best_val_acc, gcs_model_uri, run_id = train(
            cfg=cfg,
            model=model,
            tokenizer=tokenizer,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            save_path=save_path,
        )

        print(f"\nBest val accuracy : {best_val_acc:.4f}")
        print(f"Model uploaded to : {gcs_model_uri}")
        print(f"MLflow run ID     : {run_id}")
        print("Smoke test passed.")

    main()
