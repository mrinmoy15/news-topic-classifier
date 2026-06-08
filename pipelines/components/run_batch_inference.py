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
    batch_size: int = 32,
    max_seq_length: int = 512,
) -> str:
    """
    KFP component — run BERT inference on the input Parquet and write
    original rows + a predicted_label column back to GCS.

    Parameters
    ----------
    gcp_project : str
        GCP project ID.
    gcs_model_uri : str
        GCS URI of the fine-tuned model directory.
    gcs_input_uri : str
        GCS URI of the input Parquet from fetch_inference_data_component.
    gcs_bucket_data : str
        GCS bucket to write the predictions Parquet.
    batch_size : int
        Number of texts per forward pass.
    max_seq_length : int
        Maximum tokeniser sequence length.

    Returns
    -------
    str
        GCS URI of the predictions Parquet — passed to write_inference_results_component.
    """
    import io
    import os
    import tempfile

    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch
    from google.cloud import storage

    from news_topic_classifier.modeling.predict import load_model_tokenizer
    from news_topic_classifier.modeling.train import download_base_model

    # ── Download input Parquet from GCS ───────────────────────────────────────
    gcs_path = gcs_input_uri.replace("gs://", "")
    bucket_name, blob_name = gcs_path.split("/", 1)
    gcs_client = storage.Client(project=gcp_project)

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

        pred_ids = torch.softmax(logits, dim=-1).cpu().numpy().argmax(axis=-1)

        for i, row in enumerate(chunk):
            output_rows.append({**row, "predicted_label": id2label[int(pred_ids[i])]})

    print(f"Inference complete — {len(output_rows)} predictions")

    # ── Write predictions Parquet to GCS ─────────────────────────────────────
    pred_table = pa.Table.from_pylist(output_rows)
    gcs_out_uri = f"gs://{gcs_bucket_data}/inference/samples/predictions.parquet"
    out_path = gcs_out_uri.replace("gs://", "")
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
