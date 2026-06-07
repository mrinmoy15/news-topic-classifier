import os

from kfp import dsl

_TRAINER_IMAGE = os.environ.get(
    "TRAINER_IMAGE",
    "us-central1-docker.pkg.dev/cs-cdwp-data-dev2188/news-topic-classifier/trainer:latest",
)


@dsl.component(base_image=_TRAINER_IMAGE)
def run_batch_inference_component(
    gcp_project: str,
    gcs_model_uri: str,
    gcs_input_uri: str,
    gcs_bucket_data: str,
    day: int = -1,
    batch_size: int = 32,
    max_seq_length: int = 512,
) -> str:
    """
    KFP component — run BERT inference on an input Parquet and write predictions to GCS.

    Downloads the fine-tuned model from GCS, reads the input Parquet produced
    by fetch_inference_data_component, runs classification in mini-batches,
    and writes a predictions Parquet back to GCS.

    Parameters
    ----------
    gcp_project : str
        GCP project ID.
    gcs_model_uri : str
        GCS URI of the fine-tuned model directory (must contain config.json,
        model.safetensors, tokenizer files).
    gcs_input_uri : str
        GCS URI of the input Parquet from fetch_inference_data_component.
    gcs_bucket_data : str
        GCS bucket to write the predictions Parquet.
    day : int
        Partition index for output path naming. -1 = auto-compute from UTC date.
    batch_size : int
        Number of texts to tokenise and infer in a single forward pass.
    max_seq_length : int
        Maximum tokeniser sequence length.

    Returns
    -------
    str
        GCS URI of the predictions Parquet — passed to write_inference_results_component.
    """
    import os
    import tempfile
    from datetime import datetime, timezone

    import pyarrow.parquet as pq
    import torch
    from google.cloud import storage

    from news_topic_classifier.modeling.predict import load_model_tokenizer
    from news_topic_classifier.modeling.train import download_base_model

    if day < 0:
        day = (datetime.now(timezone.utc).day - 1) % 30

    # ── Download input Parquet from GCS ───────────────────────────────────────
    gcs_path = gcs_input_uri.replace("gs://", "")
    bucket_name, blob_name = gcs_path.split("/", 1)
    gcs_client = storage.Client(project=gcp_project)

    import io
    fd, tmp_path = tempfile.mkstemp(suffix=".parquet")
    os.close(fd)
    try:
        gcs_client.bucket(bucket_name).blob(blob_name).download_to_filename(tmp_path)
        with open(tmp_path, "rb") as _fh:
            _buf = io.BytesIO(_fh.read())
    finally:
        os.unlink(tmp_path)
    input_table = pq.read_table(_buf)

    rows = input_table.to_pylist()
    print(f"Loaded {len(rows)} articles from {gcs_input_uri}")

    # ── Load model ────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_local = download_base_model(
        gcs_model_uri=gcs_model_uri,
        local_dir="/tmp/bert-bbc-finetuned",
        gcp_project=gcp_project,
    )
    model, tokenizer = load_model_tokenizer(model_local, device)
    id2label = model.config.id2label
    print(f"Model loaded on {device}  |  labels: {list(id2label.values())}")

    # ── Mini-batch inference ──────────────────────────────────────────────────
    def _label_idx(id2label: dict, label: str) -> int:
        for idx, name in id2label.items():
            if name == label:
                return int(idx)
        return 0

    now       = datetime.now(timezone.utc)
    pred_date = now.date().isoformat()
    run_ts    = now.isoformat()

    output_rows = []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        texts = [r["body"] for r in chunk]

        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_seq_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits

        probs    = torch.softmax(logits, dim=-1).cpu().numpy()
        pred_ids = probs.argmax(axis=-1)

        for i, row in enumerate(chunk):
            label      = id2label[int(pred_ids[i])]
            confidence = float(probs[i][int(pred_ids[i])])
            output_rows.append({
                "prediction_date":     pred_date,
                "run_timestamp":       run_ts,
                "title":               row["title"],
                "body":                row["body"],
                "true_label":          row["true_label"],
                "predicted_label":     label,
                "confidence":          confidence,
                "score_business":      float(probs[i][_label_idx(id2label, "business")]),
                "score_entertainment": float(probs[i][_label_idx(id2label, "entertainment")]),
                "score_politics":      float(probs[i][_label_idx(id2label, "politics")]),
                "score_sport":         float(probs[i][_label_idx(id2label, "sport")]),
                "score_tech":          float(probs[i][_label_idx(id2label, "tech")]),
                "day_partition":       day,
            })

    print(f"Inference complete — {len(output_rows)} predictions")

    # ── Write predictions Parquet to GCS ─────────────────────────────────────
    import pyarrow as pa
    pred_table = pa.Table.from_pylist(output_rows)

    gcs_out_uri = f"gs://{gcs_bucket_data}/inference/day={day}/predictions.parquet"
    out_path    = gcs_out_uri.replace("gs://", "")
    out_bucket, out_blob = out_path.split("/", 1)

    fd, tmp_path = tempfile.mkstemp(suffix=".parquet")
    os.close(fd)
    try:
        pq.write_table(pred_table, tmp_path)
        gcs_client.bucket(out_bucket).blob(out_blob).upload_from_filename(tmp_path)
    finally:
        os.unlink(tmp_path)

    print(f"Predictions written to {gcs_out_uri}")
    return gcs_out_uri
