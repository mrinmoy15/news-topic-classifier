"""Unit tests for model registration — KFP component and manual script."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# ─── KFP component ───────────────────────────────────────────────────────────

from pipelines.components.register_model import register_model_component

# The underlying Python function is accessible via .python_func on KFP components.
_fn = register_model_component.python_func


def _mock_aiplatform(resource_name="projects/123/locations/us-central1/models/456",
                     version_id="1"):
    """Return a (mock_module, mock_model) pair for patching google.cloud.aiplatform."""
    mock_model = MagicMock()
    mock_model.resource_name = resource_name
    mock_model.version_id    = version_id

    mock_aip = MagicMock()
    mock_aip.Model.upload.return_value = mock_model
    return mock_aip, mock_model


# ─── component: return value ─────────────────────────────────────────────────

def test_component_returns_resource_name():
    mock_aip, _ = _mock_aiplatform()
    with patch("google.cloud.aiplatform", mock_aip):
        result = _fn(
            gcp_project="test-proj",
            gcp_region="us-central1",
            gcs_model_uri="gs://bucket/model/",
            environment_name="dev",
        )
    assert result == "projects/123/locations/us-central1/models/456"


def test_component_calls_aiplatform_init():
    mock_aip, _ = _mock_aiplatform()
    with patch("google.cloud.aiplatform", mock_aip):
        _fn(
            gcp_project="my-project",
            gcp_region="eu-west1",
            gcs_model_uri="gs://bucket/model/",
            environment_name="dev",
        )
    mock_aip.init.assert_called_once_with(project="my-project", location="eu-west1")


def test_component_calls_model_upload_with_artifact_uri():
    mock_aip, _ = _mock_aiplatform()
    with patch("google.cloud.aiplatform", mock_aip):
        _fn(
            gcp_project="proj",
            gcp_region="us-central1",
            gcs_model_uri="gs://bucket/models/bert/",
            environment_name="dev",
        )
    call_kwargs = mock_aip.Model.upload.call_args.kwargs
    assert call_kwargs["artifact_uri"] == "gs://bucket/models/bert/"


def test_component_uses_custom_display_name():
    mock_aip, _ = _mock_aiplatform()
    with patch("google.cloud.aiplatform", mock_aip):
        _fn(
            gcp_project="proj",
            gcp_region="us-central1",
            gcs_model_uri="gs://bucket/model/",
            environment_name="dev",
            display_name="my-custom-model",
        )
    call_kwargs = mock_aip.Model.upload.call_args.kwargs
    assert call_kwargs["display_name"] == "my-custom-model"


def test_component_labels_include_environment():
    mock_aip, _ = _mock_aiplatform()
    with patch("google.cloud.aiplatform", mock_aip):
        _fn(
            gcp_project="proj",
            gcp_region="us-central1",
            gcs_model_uri="gs://bucket/model/",
            environment_name="prd",
        )
    labels = mock_aip.Model.upload.call_args.kwargs["labels"]
    assert labels["environment"] == "prd"


def test_component_version_description_contains_environment():
    mock_aip, _ = _mock_aiplatform()
    with patch("google.cloud.aiplatform", mock_aip):
        _fn(
            gcp_project="proj",
            gcp_region="us-central1",
            gcs_model_uri="gs://bucket/model/",
            environment_name="pp",
        )
    version_desc = mock_aip.Model.upload.call_args.kwargs["version_description"]
    assert "pp" in version_desc


def test_component_uses_custom_serving_container():
    mock_aip, _ = _mock_aiplatform()
    custom_image = "us-central1-docker.pkg.dev/my-proj/repo/api:latest"
    with patch("google.cloud.aiplatform", mock_aip):
        _fn(
            gcp_project="proj",
            gcp_region="us-central1",
            gcs_model_uri="gs://bucket/model/",
            environment_name="dev",
            serving_container_uri=custom_image,
        )
    call_kwargs = mock_aip.Model.upload.call_args.kwargs
    assert call_kwargs["serving_container_image_uri"] == custom_image


def test_component_calls_upload_with_sync_true():
    mock_aip, _ = _mock_aiplatform()
    with patch("google.cloud.aiplatform", mock_aip):
        _fn(
            gcp_project="proj",
            gcp_region="us-central1",
            gcs_model_uri="gs://bucket/model/",
            environment_name="dev",
        )
    assert mock_aip.Model.upload.call_args.kwargs["sync"] is True


def test_component_default_display_name():
    mock_aip, _ = _mock_aiplatform()
    with patch("google.cloud.aiplatform", mock_aip):
        _fn(
            gcp_project="proj",
            gcp_region="us-central1",
            gcs_model_uri="gs://bucket/model/",
            environment_name="dev",
        )
    assert mock_aip.Model.upload.call_args.kwargs["display_name"] == "bert-bbc-news-classifier"


def test_component_serving_routes_and_port():
    mock_aip, _ = _mock_aiplatform()
    with patch("google.cloud.aiplatform", mock_aip):
        _fn(
            gcp_project="proj",
            gcp_region="us-central1",
            gcs_model_uri="gs://bucket/model/",
            environment_name="dev",
        )
    kw = mock_aip.Model.upload.call_args.kwargs
    assert kw["serving_container_predict_route"] == "/predict"
    assert kw["serving_container_health_route"] == "/health"
    assert kw["serving_container_ports"] == [8080]


def test_component_labels_include_all_static_keys():
    mock_aip, _ = _mock_aiplatform()
    with patch("google.cloud.aiplatform", mock_aip):
        _fn(
            gcp_project="proj",
            gcp_region="us-central1",
            gcs_model_uri="gs://bucket/model/",
            environment_name="dev",
        )
    labels = mock_aip.Model.upload.call_args.kwargs["labels"]
    assert labels["base_model"] == "bert-base-uncased"
    assert labels["task"] == "text-classification"
    assert labels["dataset"] == "bbc-news"


# ─── script: register_model() ────────────────────────────────────────────────

from scripts.register_model import _ENV_CONFIG, register_model


def _mock_script_aiplatform(resource_name="projects/1/locations/us-central1/models/2",
                             version_id="3"):
    mock_model = MagicMock()
    mock_model.resource_name = resource_name
    mock_model.version_id    = version_id

    mock_aip = MagicMock()
    mock_aip.Model.upload.return_value = mock_model
    return mock_aip


def test_script_calls_aiplatform_init_with_correct_project():
    mock_aip = _mock_script_aiplatform()
    with patch("scripts.register_model.aiplatform", mock_aip):
        register_model("dev", None, "bert-bbc-news-classifier", "test run", "container:latest")
    mock_aip.init.assert_called_once_with(
        project=_ENV_CONFIG["dev"]["project"],
        location=_ENV_CONFIG["dev"]["region"],
    )


def test_script_infers_default_gcs_uri_from_environment():
    mock_aip = _mock_script_aiplatform()
    with patch("scripts.register_model.aiplatform", mock_aip):
        register_model("dev", None, "bert-bbc-news-classifier", "v1", "container:latest")
    artifact_uri = mock_aip.Model.upload.call_args.kwargs["artifact_uri"]
    expected_bucket = _ENV_CONFIG["dev"]["artifacts_bucket"]
    assert expected_bucket in artifact_uri


def test_script_uses_explicit_gcs_uri_when_provided():
    mock_aip = _mock_script_aiplatform()
    explicit = "gs://custom-bucket/models/my-run/"
    with patch("scripts.register_model.aiplatform", mock_aip):
        register_model("dev", explicit, "bert-bbc-news-classifier", "v1", "container:latest")
    assert mock_aip.Model.upload.call_args.kwargs["artifact_uri"] == explicit


def test_script_returns_model_object():
    mock_aip = _mock_script_aiplatform()
    with patch("scripts.register_model.aiplatform", mock_aip):
        result = register_model("dev", None, "bert-bbc-news-classifier", "v1", "container:latest")
    assert result is mock_aip.Model.upload.return_value


def test_script_all_envs_have_config():
    for env in ("dev", "pp", "prd"):
        cfg = _ENV_CONFIG[env]
        assert "project" in cfg
        assert "region" in cfg
        assert "artifacts_bucket" in cfg


# ─── script: CLI argument parsing ────────────────────────────────────────────

from scripts.register_model import _parse_args


def test_parse_args_default_environment():
    args = _parse_args([])
    assert args.environment == "dev"


def test_parse_args_explicit_environment():
    args = _parse_args(["--environment", "prd"])
    assert args.environment == "prd"


def test_parse_args_default_display_name():
    args = _parse_args([])
    assert args.display_name == "bert-bbc-news-classifier"


def test_parse_args_gcs_model_uri_defaults_to_none():
    args = _parse_args([])
    assert args.gcs_model_uri is None


def test_parse_args_explicit_gcs_model_uri():
    args = _parse_args(["--gcs-model-uri", "gs://bucket/model/"])
    assert args.gcs_model_uri == "gs://bucket/model/"
