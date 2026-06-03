"""Unit tests for news_topic_classifier.config."""
from __future__ import annotations

from omegaconf import OmegaConf

from news_topic_classifier.config import (
    ID2LABEL,
    LABEL2ID,
    NUM_LABELS,
    get_artifact_registry_uri,
    get_bq_results_full_table,
    get_vertex_pipeline_root,
)


# ─── Constants ───────────────────────────────────────────────────────────────

def test_label2id_has_five_classes():
    assert len(LABEL2ID) == 5


def test_id2label_is_inverse_of_label2id():
    for label, idx in LABEL2ID.items():
        assert ID2LABEL[idx] == label


def test_num_labels_matches_mapping_length():
    assert NUM_LABELS == len(LABEL2ID)


def test_all_expected_categories_present():
    expected = {"business", "entertainment", "politics", "sport", "tech"}
    assert set(LABEL2ID.keys()) == expected


def test_label_ids_are_zero_indexed_contiguous():
    assert set(LABEL2ID.values()) == set(range(NUM_LABELS))


# ─── Config helpers ──────────────────────────────────────────────────────────

@staticmethod
def _make_cfg(**env_overrides):
    base = {
        "environment": {
            "gcp_project": "my-project",
            "gcp_region": "us-central1",
            "gcs_bucket_artifacts": "my-artifacts",
            "artifact_registry_repo": "my-repo",
            "bq_dataset": "MY_DATASET",
            "bq_results_table": "results",
        }
    }
    base["environment"].update(env_overrides)
    return OmegaConf.create(base)


def test_get_vertex_pipeline_root_contains_bucket():
    cfg = _make_cfg()
    assert "my-artifacts" in get_vertex_pipeline_root(cfg)


def test_get_vertex_pipeline_root_format():
    cfg = _make_cfg()
    assert get_vertex_pipeline_root(cfg) == "gs://my-artifacts/pipeline-root"


def test_get_artifact_registry_uri_contains_project():
    cfg = _make_cfg()
    assert "my-project" in get_artifact_registry_uri(cfg)


def test_get_artifact_registry_uri_contains_region():
    cfg = _make_cfg()
    assert "us-central1" in get_artifact_registry_uri(cfg)


def test_get_artifact_registry_uri_contains_repo():
    cfg = _make_cfg()
    assert "my-repo" in get_artifact_registry_uri(cfg)


def test_get_bq_results_full_table_format():
    cfg = _make_cfg()
    result = get_bq_results_full_table(cfg)
    assert result == "my-project.MY_DATASET.results"
