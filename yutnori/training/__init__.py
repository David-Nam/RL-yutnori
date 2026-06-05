"""Training helpers for learned Yutnori agents."""

from yutnori.training.env_factory import (
    OPPONENT_NAMES,
    make_opponent,
    make_yutnori_env,
    make_yutnori_vec_env,
)
from yutnori.training.model_config import resolve_model_observation_mode
from yutnori.training.ppo_evaluation import (
    PolicyEvaluationResult,
    evaluate_maskable_policy,
)
from yutnori.training.reward_shaping import (
    RF_SHAPING_CAPTURE_WEIGHT,
    RF_SHAPING_FINISH_WEIGHT,
    RF_SHAPING_SHORTCUT_BONUS,
    project_rf_event_shaping_reward,
    project_rf_events_shaping_reward,
)

__all__ = [
    "OPPONENT_NAMES",
    "PolicyEvaluationResult",
    "RF_SHAPING_CAPTURE_WEIGHT",
    "RF_SHAPING_FINISH_WEIGHT",
    "RF_SHAPING_SHORTCUT_BONUS",
    "evaluate_maskable_policy",
    "make_opponent",
    "make_yutnori_env",
    "make_yutnori_vec_env",
    "project_rf_event_shaping_reward",
    "project_rf_events_shaping_reward",
    "resolve_model_observation_mode",
]
