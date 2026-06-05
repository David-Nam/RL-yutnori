"""Helpers for reading training metadata saved beside PPO models."""

from __future__ import annotations

import json
from pathlib import Path

from yutnori.env import OBSERVATION_MODE_BASE, OBSERVATION_MODES


def resolve_model_observation_mode(
    model_path: Path,
    requested_observation_mode: str | None = None,
) -> str:
    """Resolve the observation mode to use when evaluating a saved model."""

    if requested_observation_mode is not None:
        return _validate_observation_mode(
            requested_observation_mode,
            source="--observation-mode",
        )

    config_path = model_path.parent / "config.json"
    if not config_path.exists():
        return OBSERVATION_MODE_BASE

    try:
        config = json.loads(config_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON config: {config_path}") from exc

    observation_mode = config.get("observation_mode", OBSERVATION_MODE_BASE)
    return _validate_observation_mode(
        observation_mode,
        source=f"{config_path} observation_mode",
    )


def _validate_observation_mode(observation_mode: object, *, source: str) -> str:
    if isinstance(observation_mode, str) and observation_mode in OBSERVATION_MODES:
        return observation_mode
    expected = ", ".join(OBSERVATION_MODES)
    raise ValueError(f"{source} must be one of {expected}")
