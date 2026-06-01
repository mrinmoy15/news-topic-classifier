# News Topic Classifier

A production-ready ML system that fine-tunes BERT to classify BBC news articles into 5 categories: **business**, **entertainment**, **politics**, **sport**, and **tech**.

Built on Google Cloud Platform with Vertex AI pipelines, MLflow experiment tracking, and a fully automated GitHub Actions CI/CD workflow.

---

## Architecture

```
BigQuery (BBC News)
       │
       ▼
┌─────────────────────────────────────────────────────┐
│              Vertex AI Training Pipeline            │
│                                                     │
│  Extract ──► Preprocess ──► Train ──► Predict ──► Report │
│   (BQ)        (Parquet)    (BERT)   (Inference)  (Metrics) │
└─────────────────────────────────────────────────────┘
       │                        │
       ▼                        ▼
  GCS Buckets             MLflow Tracking
  (data + models)         (Cloud Run)
       │
       ▼
   FastAPI (inference) + Streamlit (dashboard)
```

![Vertex AI Pipeline DAG](docs/images/pipeline_dag.png)

**Five-step Kubeflow Pipelines v2 workflow:**

| Step | Component | Description |
|------|-----------|-------------|
| 1 | `extract.py` | Pull articles from BigQuery public dataset |
| 2 | `preprocess.py` | Clean text, stratified 80/10/10 split |
| 3 | `train.py` | Fine-tune `bert-base-uncased` with AdamW |
| 4 | `predict.py` | Run inference on held-out test set |
| 5 | `evaluate.py` | Generate classification report |

---

## Model

| Parameter | Value |
|-----------|-------|
| Base model | `bert-base-uncased` (110M params) |
| Max sequence length | 512 tokens |
| Tokenization strategy | Head (340) + Tail (170) tokens |
| Epochs | 5 (early stopping patience: 3) |
| Batch size | 8 |
| Learning rate | 2e-5 (linear warmup + decay) |
| Warmup steps | 100 |
| Optimizer | AdamW (weight decay: 0.01) |
| Dropout | 0.3 |

Labels: `business`, `entertainment`, `politics`, `sport`, `tech`

Dataset: [`bigquery-public-data.bbc_news.fulltext`](https://console.cloud.google.com/marketplace/product/bbc/bbc-news)

---

## Project Structure

```
news-topic-classifier/
├── api/                        # FastAPI inference service
├── dashboard/                  # Streamlit monitoring dashboard
├── pipelines/                  # Vertex AI pipeline definitions (KFP v2)
│   └── components/             # Extract, preprocess, train, predict, evaluate
├── news_topic_classifier/      # Core Python package
│   └── modeling/               # BERT classifier, training loop, inference, reports
├── conf/                       # Hydra configuration
│   ├── model/                  # Model hyperparameters
│   ├── training/               # Training hyperparameters
│   ├── data/                   # Data pipeline paths
│   └── environment/            # dev / pp / prd GCP settings
├── docker/                     # Base and trainer Dockerfiles
├── tests/                      # Unit and integration tests
├── .github/workflows/          # CI/CD (build, run_pipeline, promote)
└── requirements/               # Layered dependency files
```

---

## Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- Docker
- GCP project with the following APIs enabled:
  - BigQuery, Cloud Storage, Vertex AI, Artifact Registry
- [Workload Identity Federation](https://cloud.google.com/iam/docs/workload-identity-federation) configured for GitHub Actions

---

## Installation

```bash
# Core dependencies
make install

# With development extras (Jupyter, plotting)
make install-dev

# With test extras
make install-test
```

Or install specific groups directly:

```bash
pip install -e . -r requirements/base.txt
pip install -r requirements/dev.txt
```

---

## Configuration

All configuration is managed with [Hydra](https://hydra.cc/). The main entrypoint is `conf/config.yaml`, which composes from:

```
conf/
├── config.yaml            # root config
├── environment/
│   ├── dev.yaml           # cs-cdwp-data-dev2188
│   ├── pp.yaml            # cs-cdwp-data-pp2188
│   └── prd.yaml           # cs-cdwp-data-prd2188
├── model/default.yaml     # BERT hyperparameters
├── training/default.yaml  # optimizer, scheduler, epochs
└── data/default.yaml      # BigQuery table, GCS paths
```

Copy `.env.example` to `.env` and fill in your GCP credentials for local runs:

```bash
cp .env.example .env
```

---

## Running Locally

### Smoke-test individual pipeline components

```bash
# Build Docker images locally
make docker-build

# Test each component in isolation
make docker-test-extract
make docker-test-preprocess RAW_GCS_URI=gs://your-bucket/data/raw/
make docker-test-train      GCS_SPLITS_DIR=gs://your-bucket/data/processed/
make docker-test-predict    GCS_SPLITS_DIR=gs://your-bucket/data/processed/
make docker-test-report     RUN_ID=<mlflow-run-id> GCS_PREDICTIONS_URI=gs://...

# Or run all steps end-to-end
make docker-test-all
```

### Start local services

```bash
# MLflow UI at http://localhost:5000
docker compose up

# Streamlit dashboard at http://localhost:8501
docker compose up dashboard
```

### Run Python modules directly

```bash
python -m news_topic_classifier.dataset          # Extract from BigQuery
python -m news_topic_classifier.features         # Preprocess & split
python -m news_topic_classifier.modeling.train   # Fine-tune BERT
python -m news_topic_classifier.modeling.predict # Run inference
python -m news_topic_classifier.modeling.report  # Generate report
```

---

## Submitting a Vertex AI Pipeline

```bash
# Default: dev environment
python pipelines/run_pipeline.py

# Target a specific environment
python pipelines/run_pipeline.py environment=pp
python pipelines/run_pipeline.py environment=prd

# Override hyperparameters at submission time
python pipelines/run_pipeline.py environment=dev training.epochs=10 training.lr=3e-5
```

The pipeline is compiled to `pipelines/compiled/training_pipeline.yaml` before submission.

---

## CI/CD (GitHub Actions)

| Workflow | Trigger | Description |
|----------|---------|-------------|
| `test.yml` | PR to `develop`/`main`, push to `develop` | Run unit test suite (no GCP) |
| `build.yml` | Push to `develop` | Build and push `base` + `trainer` images to Artifact Registry |
| `run_pipeline.yml` | After `build.yml` succeeds, or manual | Submit Vertex AI training pipeline |
| `promote.yml` | Manual | Promote Docker images dev → pp → prd |
| `integration_test.yml` | Push to `main`, or manual | Run integration tests against dev GCP environment |

Authentication uses [Workload Identity Federation](https://cloud.google.com/iam/docs/workload-identity-federation) — no long-lived service account keys are stored in GitHub secrets.

---

## MLflow Tracking

Experiments are tracked on MLflow servers hosted on Cloud Run:

| Environment | Endpoint |
|-------------|----------|
| dev | https://mlflow-tracking-server-eeh43tst7q-uc.a.run.app |
| pre-prod | https://mlflow-tracking-server-nityigrfzq-uc.a.run.app |
| prod | https://mlflow-tracking-server-wngg5g6m6q-uc.a.run.app |

Each training run logs: hyperparameters, per-epoch train/val accuracy, best model checkpoint, and a final classification report.

---

## Testing

```bash
# Install test dependencies
make install-test

# Run all unit tests
make test

# Run with coverage report
make test-cov

# Run a specific file
pytest tests/unit/test_dataset.py -v
```

Tests are split into `tests/unit/` (fast, no GCP) and `tests/integration/` (requires real GCP credentials).

### Integration tests

Integration tests hit real GCP services (BigQuery, GCS, MLflow) in the dev environment. They are **skipped automatically** unless `INTEGRATION_TESTS=true` is set.

**PowerShell (Windows):**
```powershell
$env:INTEGRATION_TESTS = "true"
make integration-test        # Tier 1 — data pipeline only (~3 min)
make integration-test-full   # Tier 2 — full pipeline including training (~25 min)
```

**bash / Linux / macOS:**
```bash
INTEGRATION_TESTS=true make integration-test
INTEGRATION_TESTS=true make integration-test-full
```

| Test | Tier | What it does |
|------|------|-------------|
| `test_01_extract` | 1 | Pull 50 rows from BigQuery → GCS Parquet |
| `test_02_preprocess` | 1 | Clean + stratified split → 3 GCS Parquet files |
| `test_03_train` | 2 (`slow`) | Download splits → 1-epoch BERT fine-tune → GCS model |
| `test_04_predict` | 2 (`slow`) | Load model → test-set inference → GCS predictions |
| `test_05_report` | 2 (`slow`) | MLflow data + predictions → plots + Word doc on GCS |

Tests share state via a module-scoped `pipeline_artifacts` dict so each step feeds the next. All GCS objects are written under a timestamped `integration-tests/<timestamp>/` prefix and **cleaned up automatically** after the session.

### CI trigger

Unit tests run automatically via [`.github/workflows/test.yml`](.github/workflows/test.yml):

| Event | Branches |
|-------|----------|
| Pull request (opened / updated) | `develop`, `main` |
| Push | `develop` (when source files or tests change) |

The workflow installs `requirements/base.txt` + `requirements/test.txt`, runs the full unit suite, and uploads a `coverage.xml` artifact.

### Unit test coverage

All tests mock GCP clients and avoid loading real BERT weights. A shared `_FakeModel` (tiny `nn.Linear`) and `_FakeTokenizer` in [tests/conftest.py](tests/conftest.py) stand in for the full model, keeping the suite fast.

| File | Tests | What's covered |
|------|-------|----------------|
| [test_dataset.py](tests/unit/test_dataset.py) | 17 | `_gcs_output_path` URI format, `_build_extraction_query` SQL (CASE labels, NULL filters, FARM_FINGERPRINT sampling), `BBCNewsDataset` length / tensor shapes / dtypes / `use_title` concatenation |
| [test_features.py](tests/unit/test_features.py) | 16 | `_build_preprocessing_query` SQL (split labels, pct boundaries, FARM_FINGERPRINT, NFKC normalisation, HTML stripping), `_gcs_split_output_paths` shared timestamp and bucket |
| [test_model.py](tests/unit/test_model.py) | 18 | `build_model` id2label/label2id config (mocked), `build_optimizer_scheduler` AdamW type/lr/weight_decay, `build_dataloaders` batch counts, `train_epoch` and `eval_epoch` return types / loss bounds / training-mode side-effects |
| [test_predictor.py](tests/unit/test_predictor.py) | 19 | `compute_metrics` accuracy values and report structure, `run_inference` output shapes / softmax probabilities / `has_labels=False` path, `save_predictions` Parquet column schema / row count / GCS URI format (GCS upload mocked) |
| [test_report.py](tests/unit/test_report.py) | 12 | `plot_training_curves` (file created, raises on empty history), `plot_confusion_matrix`, `plot_per_class_metrics` — all verify the PNG is written to disk. Skipped automatically if `matplotlib`/`seaborn` are not installed. |

---

## Development

Install linting tools:

```bash
make install-lint   # installs ruff and mypy
```

Then run them directly:

```bash
ruff check .        # lint
ruff format .       # auto-format
mypy news_topic_classifier/
```

Code style is enforced by [Ruff](https://github.com/astral-sh/ruff) and [mypy](https://mypy.readthedocs.io/).

---

## GCP Resource Summary

| Resource | Dev | Notes |
|----------|-----|-------|
| Project | `cs-cdwp-data-dev2188` | Separate projects per environment |
| Region | `us-central1` | All resources |
| GCS data bucket | `cs-cdwp-data-dev-model-data` | Raw, interim, processed splits |
| GCS artifacts bucket | `cs-cdwp-data-dev-model-artifacts` | Models, pipeline root |
| Artifact Registry | `us-central1-docker.pkg.dev/{project}/news-topic-classifier` | Docker images |
| Vertex AI SA | `vertex-ai-sa@cs-cdwp-data-dev.iam.gserviceaccount.com` | Pipeline runner |
| Training resource | 8 vCPU / 32 GB RAM | Vertex AI training component |

---

## License

[MIT](LICENSE)
