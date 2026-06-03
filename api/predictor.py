"""
Core inference logic — shared by the FastAPI service and the batch predict job.

Responsibilities
----------------
- Download the fine-tuned model from GCS on first call.
- Load the HuggingFace model + tokenizer into memory.
- Run BERT inference on a list of texts in mini-batches (no full-dataset load).
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import torch
from transformers import BertForSequenceClassification, BertTokenizerFast

from news_topic_classifier.modeling.predict import load_model_tokenizer
from news_topic_classifier.modeling.train import download_base_model

logger = logging.getLogger(__name__)

_MODEL_LOCAL_DIR    = "/tmp/bert-bbc-model"
_MAX_SEQ_LENGTH     = int(os.environ.get("MAX_SEQ_LENGTH",     "512"))
_PREDICT_BATCH_SIZE = int(os.environ.get("PREDICT_BATCH_SIZE", "32"))


@dataclass
class Prediction:
    label: str
    confidence: float
    scores: dict[str, float]


def download_and_load(
    gcs_uri: str,
    gcp_project: str,
    local_dir: str = _MODEL_LOCAL_DIR,
) -> tuple[BertForSequenceClassification, BertTokenizerFast, torch.device]:
    """Download model artifacts from GCS and load into memory.

    Uses the trainer's existing download_base_model utility so the same
    GCS path convention is shared across training and serving.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Downloading model from %s -> %s", gcs_uri, local_dir)
    model_path = download_base_model(
        gcs_model_uri=gcs_uri,
        local_dir=local_dir,
        gcp_project=gcp_project,
    )
    model, tokenizer = load_model_tokenizer(model_path, device)
    logger.info("Model loaded on %s  |  id2label: %s", device, model.config.id2label)
    return model, tokenizer, device


def predict_texts(
    texts: list[str],
    model: BertForSequenceClassification,
    tokenizer: BertTokenizerFast,
    device: torch.device,
    batch_size: int = _PREDICT_BATCH_SIZE,
) -> list[Prediction]:
    """Run inference on a list of texts in mini-batches.

    Never loads the full text list into a single tensor — processes
    `batch_size` rows at a time to keep memory bounded.
    """
    id2label = model.config.id2label
    results: list[Prediction] = []

    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]

        inputs = tokenizer(
            chunk,
            padding=True,
            truncation=True,
            max_length=_MAX_SEQ_LENGTH,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits

        probs    = torch.softmax(logits, dim=-1).cpu().numpy()
        pred_ids = probs.argmax(axis=-1)

        for i in range(len(chunk)):
            label      = id2label[int(pred_ids[i])]
            confidence = float(probs[i][int(pred_ids[i])])
            scores     = {id2label[j]: float(probs[i][j]) for j in range(len(id2label))}
            results.append(Prediction(label=label, confidence=confidence, scores=scores))

    return results
