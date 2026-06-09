"""Unit tests for dashboard/app.py helper functions.

Streamlit is mocked at module level (it runs rendering code on import).
All other dependencies (httpx, vertexai, google-auth) are mocked per-test
so they don't pollute sys.modules for other test files.
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

# ─── Mock streamlit BEFORE importing dashboard.app ────────────────────────────
# streamlit executes rendering calls at import time; we replace the whole
# module with a MagicMock so those calls are silently absorbed.
# cache_data / cache_resource are made into pass-through decorators so the
# wrapped functions remain callable as normal Python functions in tests.

def _passthrough_decorator(*args, **kwargs):
    def decorator(fn):
        return fn
    if len(args) == 1 and callable(args[0]):
        return args[0]
    return decorator


_mock_st = MagicMock()
_mock_st.cache_data.side_effect    = _passthrough_decorator
_mock_st.cache_resource.side_effect = _passthrough_decorator
_mock_st.tabs.return_value = tuple(MagicMock() for _ in range(6))
_mock_st.columns.return_value = (MagicMock(), MagicMock())
_mock_st.button.return_value = False  # buttons not clicked by default
_mock_st.sidebar = MagicMock()

sys.modules["streamlit"] = _mock_st

# plotly is a dashboard-only dependency not installed in the test environment.
sys.modules["plotly"] = MagicMock()
sys.modules["plotly.express"] = MagicMock()
sys.modules["plotly.graph_objects"] = MagicMock()

# Now safe to import the app module.
import dashboard.app as app  # noqa: E402


# ═══════════════════════════════════════════════════════════════════════════════
# _auth_headers
# ═══════════════════════════════════════════════════════════════════════════════

class TestAuthHeaders:

    def test_returns_empty_dict_for_localhost(self, monkeypatch):
        monkeypatch.setattr(app, "API_URL", "http://localhost:8080")
        assert app._auth_headers() == {}

    def test_returns_bearer_header_for_cloud_run(self, monkeypatch):
        monkeypatch.setattr(app, "API_URL", "https://my-service-xyz.run.app")
        with patch.object(app, "_api_id_token", return_value="tok123"):
            headers = app._auth_headers()
        assert headers == {"Authorization": "Bearer tok123"}

    def test_no_bearer_for_non_cloud_run_url(self, monkeypatch):
        monkeypatch.setattr(app, "API_URL", "http://internal-service:8080")
        assert "Authorization" not in app._auth_headers()


# ═══════════════════════════════════════════════════════════════════════════════
# _api_id_token
# ═══════════════════════════════════════════════════════════════════════════════

class TestApiIdToken:

    def test_fetches_oidc_token(self):
        mock_fetch = MagicMock(return_value="oidc-token-abc")
        with patch("google.oauth2.id_token.fetch_id_token", mock_fetch), \
             patch("google.auth.transport.requests.Request"):
            token = app._api_id_token("https://my-service.run.app")
        assert token == "oidc-token-abc"

    def test_passes_audience_to_fetch(self):
        mock_fetch = MagicMock(return_value="tok")
        with patch("google.oauth2.id_token.fetch_id_token", mock_fetch), \
             patch("google.auth.transport.requests.Request"):
            app._api_id_token("https://audience.run.app")
        call_args = mock_fetch.call_args[0]
        assert "https://audience.run.app" in call_args


# ═══════════════════════════════════════════════════════════════════════════════
# _call_api
# ═══════════════════════════════════════════════════════════════════════════════

class TestCallApi:

    def test_returns_prediction_on_success(self, monkeypatch):
        monkeypatch.setattr(app, "API_URL", "http://localhost:8080")
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "predictions": [{"label": "tech", "confidence": 0.95, "scores": {}}]
        }
        import httpx
        with patch.object(httpx, "post", return_value=mock_resp):
            result = app._call_api("AI article text")
        assert result["label"] == "tech"
        assert result["confidence"] == 0.95

    def test_returns_none_on_connect_error(self, monkeypatch):
        monkeypatch.setattr(app, "API_URL", "http://localhost:8080")
        import httpx
        with patch.object(httpx, "post", side_effect=httpx.ConnectError("refused")):
            result = app._call_api("some text")
        assert result is None

    def test_returns_none_on_generic_error(self, monkeypatch):
        monkeypatch.setattr(app, "API_URL", "http://localhost:8080")
        import httpx
        with patch.object(httpx, "post", side_effect=RuntimeError("unexpected")):
            result = app._call_api("some text")
        assert result is None


# ═══════════════════════════════════════════════════════════════════════════════
# _gemini_classify
# ═══════════════════════════════════════════════════════════════════════════════

class TestGeminiClassify:

    def _mock_vertexai(self, response_text: str):
        mock_model    = MagicMock()
        mock_response = MagicMock()
        mock_response.text = response_text
        mock_model.generate_content.return_value = mock_response
        mock_gm = MagicMock()
        mock_gm.GenerativeModel.return_value = mock_model
        mock_v  = MagicMock()
        return mock_v, mock_gm

    def test_returns_valid_label(self):
        mock_v, mock_gm = self._mock_vertexai("tech")
        with patch.dict(sys.modules, {"vertexai": mock_v, "vertexai.generative_models": mock_gm}):
            assert app._gemini_classify("AI article") == "tech"

    def test_returns_unknown_for_invalid_label(self):
        mock_v, mock_gm = self._mock_vertexai("sports")  # should be 'sport'
        with patch.dict(sys.modules, {"vertexai": mock_v, "vertexai.generative_models": mock_gm}):
            assert app._gemini_classify("article") == "unknown"

    def test_returns_error_string_on_exception(self):
        mock_v  = MagicMock()
        mock_gm = MagicMock()
        mock_gm.GenerativeModel.side_effect = RuntimeError("quota exceeded")
        with patch.dict(sys.modules, {"vertexai": mock_v, "vertexai.generative_models": mock_gm}):
            label = app._gemini_classify("article")
        assert label.startswith("error:")

    @pytest.mark.parametrize("label", ["business", "entertainment", "politics", "sport", "tech"])
    def test_accepts_all_valid_labels(self, label):
        mock_v, mock_gm = self._mock_vertexai(label)
        with patch.dict(sys.modules, {"vertexai": mock_v, "vertexai.generative_models": mock_gm}):
            assert app._gemini_classify("text") == label

    def test_strips_trailing_whitespace(self):
        mock_v, mock_gm = self._mock_vertexai("politics\n")
        with patch.dict(sys.modules, {"vertexai": mock_v, "vertexai.generative_models": mock_gm}):
            assert app._gemini_classify("political article") == "politics"


# ═══════════════════════════════════════════════════════════════════════════════
# Module-level constants
# ═══════════════════════════════════════════════════════════════════════════════

def test_label_colours_covers_all_five_categories():
    assert set(app.LABEL_COLOURS.keys()) == {"business", "entertainment", "politics", "sport", "tech"}


def test_labels_list_has_five_entries():
    assert len(app.LABELS) == 5
