import json

import pytest

from yutnori.training import resolve_model_observation_mode


def test_resolve_model_observation_mode_prefers_explicit_value(tmp_path):
    model_path = tmp_path / "model.zip"
    (tmp_path / "config.json").write_text(
        json.dumps({"observation_mode": "base"}),
    )

    assert resolve_model_observation_mode(model_path, "tactical") == "tactical"


def test_resolve_model_observation_mode_reads_training_config(tmp_path):
    model_path = tmp_path / "model.zip"
    (tmp_path / "config.json").write_text(
        json.dumps({"observation_mode": "tactical"}),
    )

    assert resolve_model_observation_mode(model_path) == "tactical"


def test_resolve_model_observation_mode_defaults_to_base_without_config(tmp_path):
    assert resolve_model_observation_mode(tmp_path / "model.zip") == "base"


def test_resolve_model_observation_mode_rejects_invalid_config_value(tmp_path):
    model_path = tmp_path / "model.zip"
    (tmp_path / "config.json").write_text(
        json.dumps({"observation_mode": "unknown"}),
    )

    with pytest.raises(ValueError, match="observation_mode"):
        resolve_model_observation_mode(model_path)
