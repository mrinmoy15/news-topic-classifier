"""
BBC News Topic Classifier — FastAPI serving container.

Endpoints
---------
GET  /health     Health check — 503 while model is loading, 200 once ready.
POST /predict    Classify one or more news texts.

Environment variables
---------------------
MODEL_GCS_URI   GCS URI of the fine-tuned model directory (required).
GCP_PROJECT     GCP project ID used for GCS auth (required).
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from api.predictor import Prediction, download_and_load, predict_texts

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_MODEL_GCS_URI = os.environ.get("MODEL_GCS_URI", "")
_GCP_PROJECT   = os.environ.get("GCP_PROJECT",   "")

_model     = None
_tokenizer = None
_device: torch.device | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _tokenizer, _device
    if not _MODEL_GCS_URI:
        raise RuntimeError("MODEL_GCS_URI env var is required")
    if not _GCP_PROJECT:
        raise RuntimeError("GCP_PROJECT env var is required")
    logger.info("Loading model from %s", _MODEL_GCS_URI)
    _model, _tokenizer, _device = download_and_load(_MODEL_GCS_URI, _GCP_PROJECT)
    logger.info("Model ready — serving on %s", _device)
    yield


app = FastAPI(
    title="BBC News Topic Classifier",
    description="BERT-base-uncased fine-tuned on BBC News for 5-class topic classification.",
    version="1.0.0",
    lifespan=lifespan,
)


# ─── Request / Response schemas ──────────────────────────────────────────────

class Instance(BaseModel):
    text: str


class PredictRequest(BaseModel):
    instances: list[Instance]


class PredictionResult(BaseModel):
    label: str
    confidence: float
    scores: dict[str, float]


class PredictResponse(BaseModel):
    predictions: list[PredictionResult]


# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    return {"status": "ok", "device": str(_device)}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    if not req.instances:
        raise HTTPException(status_code=422, detail="instances list is empty")

    texts: list[str] = [inst.text for inst in req.instances]
    preds: list[Prediction] = predict_texts(texts, _model, _tokenizer, _device)

    return PredictResponse(
        predictions=[
            PredictionResult(
                label=p.label,
                confidence=round(p.confidence, 4),
                scores={k: round(v, 4) for k, v in p.scores.items()},
            )
            for p in preds
        ]
    )
