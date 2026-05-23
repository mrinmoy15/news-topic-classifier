# =============================================================================
# setup_structure.ps1
# News Topic Classifier — CCDS v2 + GCP Extensions
# Run from the root of your cloned GitHub repo:
#   cd news-topic-classifier
#   .\setup_structure.ps1
# =============================================================================

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  News Topic Classifier — Repo Scaffold   " -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

function New-Dir {
    param([string]$Path)
    if (-not (Test-Path $Path)) {
        New-Item -ItemType Directory -Path $Path -Force | Out-Null
    }
}

function New-File {
    param([string]$Path)
    if (-not (Test-Path $Path)) {
        New-Item -ItemType File -Path $Path -Force | Out-Null
    }
}

function New-Gitkeep {
    param([string]$Dir)
    New-Dir $Dir
    New-File "$Dir/.gitkeep"
}

# =============================================================================
# ROOT FILES
# =============================================================================

Write-Host "[ 1/9 ] Creating root files..." -ForegroundColor Yellow

New-File "LICENSE"
New-File "README.md"
New-File "Makefile"
New-File "pyproject.toml"
New-File "setup.cfg"
New-File ".gitignore"
New-File ".env.example"

# =============================================================================
# HYDRA CONFIGURATION
# =============================================================================

Write-Host "[ 2/11] Creating conf/..." -ForegroundColor Yellow

New-Dir "conf"
New-File "conf/config.yaml"

New-Dir "conf/environment"
New-File "conf/environment/local.yaml"
New-File "conf/environment/dev.yaml"
New-File "conf/environment/pp.yaml"
New-File "conf/environment/prd.yaml"

New-Dir "conf/model"
New-File "conf/model/default.yaml"

New-Dir "conf/training"
New-File "conf/training/default.yaml"

New-Dir "conf/data"
New-File "conf/data/default.yaml"

# =============================================================================
# REQUIREMENTS  (split by environment)
# =============================================================================
 
Write-Host "[ 2/10] Creating requirements/..." -ForegroundColor Yellow
 
New-Dir "requirements"
New-File "requirements/base.txt"        # core ML + GCP — installed everywhere
New-File "requirements/dev.txt"         # jupyter, matplotlib — local only
New-File "requirements/api.txt"         # fastapi, uvicorn — Cloud Run API
New-File "requirements/dashboard.txt"   # streamlit, plotly — Cloud Run dashboard
New-File "requirements/test.txt"        # pytest, pytest-cov — CI only
New-File "requirements/lint.txt"        # ruff, mypy — standalone

# =============================================================================
# DATA  (local only — gitignored except .gitkeep)
# =============================================================================

Write-Host "[ 2/9 ] Creating data/ directories..." -ForegroundColor Yellow

New-Gitkeep "data/raw"           # BQ sample extracts (local dev only)
New-Gitkeep "data/interim"       # Tokenized intermediate data
New-Gitkeep "data/processed"     # Final train / val / test splits
New-Gitkeep "data/external"      # BERT vocab, pretrained configs

# =============================================================================
# MODELS  (local only — gitignored except .gitkeep)
# =============================================================================

Write-Host "[ 3/9 ] Creating models/ directory..." -ForegroundColor Yellow

New-Gitkeep "models"             # Local model artifacts; real ones live in GCS

# =============================================================================
# NOTEBOOKS
# =============================================================================

Write-Host "[ 4/9 ] Creating notebooks/..." -ForegroundColor Yellow

New-Dir "notebooks"
New-File "notebooks/1.0-eda-bbc-dataset.ipynb"
New-File "notebooks/2.0-bert-baseline-experiment.ipynb"
New-File "notebooks/3.0-error-analysis.ipynb"

# =============================================================================
# REFERENCES
# =============================================================================

Write-Host "[ 5/9 ] Creating references/..." -ForegroundColor Yellow

New-Dir "references"
New-File "references/label_map.md"
New-File "references/bq_schema.md"
New-File "references/architecture.md"

# =============================================================================
# REPORTS
# =============================================================================

Write-Host "[ 6/9 ] Creating reports/..." -ForegroundColor Yellow

New-Gitkeep "reports/figures"    # Confusion matrices, metric plots

# =============================================================================
# DOCS
# =============================================================================

Write-Host "[ 7/9 ] Creating docs/..." -ForegroundColor Yellow

New-Dir "docs"
New-File "docs/index.md"
New-File "docs/pipeline.md"
New-File "docs/api.md"
New-File "docs/dashboard.md"
New-File "mkdocs.yml"

# =============================================================================
# CORE PYTHON MODULE  (news_topic_classifier/)
# CCDS v2 standard source directory
# =============================================================================

Write-Host "[ 8/9 ] Creating news_topic_classifier/ module..." -ForegroundColor Yellow

New-Dir "news_topic_classifier"
New-File "news_topic_classifier/__init__.py"
New-File "news_topic_classifier/config.py"        # env-aware settings (local/dev/pp/prd)
New-File "news_topic_classifier/dataset.py"       # BQ client + PyTorch Dataset class
New-File "news_topic_classifier/features.py"      # tokenizer logic
New-File "news_topic_classifier/plots.py"         # metrics visualisation helpers

New-Dir  "news_topic_classifier/modeling"
New-File "news_topic_classifier/modeling/__init__.py"
New-File "news_topic_classifier/modeling/bert_classifier.py"  # model architecture
New-File "news_topic_classifier/modeling/train.py"            # training loop
New-File "news_topic_classifier/modeling/predict.py"          # inference logic

# =============================================================================
# GCP EXTENSION — PIPELINES  (Vertex AI / KFP)
# =============================================================================

Write-Host "[ 9/9 ] Creating GCP extension directories..." -ForegroundColor Yellow

New-Dir "pipelines"
New-File "pipelines/training_pipeline.py"         # KFP DAG — training
New-File "pipelines/inference_pipeline.py"        # KFP DAG — batch inference

New-Dir "pipelines/components"
New-File "pipelines/components/__init__.py"
New-File "pipelines/components/extract.py"        # BQ → GCS
New-File "pipelines/components/preprocess.py"     # tokenize + split
New-File "pipelines/components/train.py"          # BERT fine-tune (wraps modeling/train.py)
New-File "pipelines/components/evaluate.py"       # metrics (wraps modeling/predict.py)
New-File "pipelines/components/predict.py"        # batch inference (wraps modeling/predict.py)
New-File "pipelines/components/write_results.py"  # GCS → BQ results

# -----------------------------------------------------------------------------
# GCP EXTENSION — API  (Cloud Run / FastAPI)
# -----------------------------------------------------------------------------

New-Dir "api"
New-File "api/main.py"                            # FastAPI application
New-File "api/predictor.py"                       # model load + inference wrapper
New-File "api/Dockerfile"
New-File "api/requirements.txt"
New-File "api/.dockerignore"

# -----------------------------------------------------------------------------
# GCP EXTENSION — DASHBOARD  (Cloud Run / Streamlit)
# -----------------------------------------------------------------------------

New-Dir "dashboard"
New-File "dashboard/app.py"                       # Streamlit app
New-File "dashboard/bq_queries.py"                # BQ query helpers
New-File "dashboard/Dockerfile"
New-File "dashboard/requirements.txt"
New-File "dashboard/.dockerignore"

# -----------------------------------------------------------------------------
# GCP EXTENSION — DOCKER  (Pipeline container images)
# -----------------------------------------------------------------------------

New-Dir "docker/base"
New-File "docker/base/Dockerfile"                 # shared base (torch, transformers, BQ)

New-Dir "docker/trainer"
New-File "docker/trainer/Dockerfile"              # extends base + GPU libs

# -----------------------------------------------------------------------------
# TESTS
# -----------------------------------------------------------------------------

New-Dir "tests/unit"
New-File "tests/__init__.py"
New-File "tests/unit/__init__.py"
New-File "tests/unit/test_dataset.py"
New-File "tests/unit/test_features.py"
New-File "tests/unit/test_model.py"
New-File "tests/unit/test_predictor.py"

New-Dir "tests/integration"
New-File "tests/integration/__init__.py"
New-File "tests/integration/test_pipeline.py"

New-File "tests/conftest.py"                      # shared pytest fixtures

# -----------------------------------------------------------------------------
# GITHUB ACTIONS  (CI/CD + scheduling)
# -----------------------------------------------------------------------------

New-Dir ".github/workflows"
New-File ".github/workflows/ci.yml"               # lint, test, build image
New-File ".github/workflows/training.yml"         # trigger training pipeline
New-File ".github/workflows/inference.yml"        # scheduled batch inference

# =============================================================================
# SUMMARY
# =============================================================================

Write-Host ""
Write-Host "==========================================" -ForegroundColor Green
Write-Host "  Scaffold complete!                       " -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Directory structure created:" -ForegroundColor White
Write-Host ""

$items = Get-ChildItem -Recurse -Force | Where-Object { $_.Name -ne ".gitkeep" }
$dirs  = ($items | Where-Object { $_.PSIsContainer }).Count
$files = ($items | Where-Object { -not $_.PSIsContainer }).Count

Write-Host "  Directories : $dirs" -ForegroundColor Cyan
Write-Host "  Files       : $files" -ForegroundColor Cyan
Write-Host ""