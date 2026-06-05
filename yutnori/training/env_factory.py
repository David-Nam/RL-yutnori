"""Factories for PPO-compatible Yutnori training environments."""

from __future__ import annotations

from collections.abc import Callable

from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv

from yutnori.agents import (
    Agent,
    CaptureFirstAgent,
    GreedyFinishAgent,
    ProjectRFRuleBasedAgent,
    RandomAgent,
)
from yutnori.env import OBSERVATION_MODE_BASE, REWARD_MODE_TERMINAL, YutnoriEnv

OPPONENT_NAMES = ("random", "capture_first", "greedy_finish", "project_rf_rule")


def make_opponent(name: str, *, seed: int | None = None) -> Agent:
    """Create one of the supported baseline opponents."""

    if name == "random":
        return RandomAgent(seed=seed)
    if name == "capture_first":
        return CaptureFirstAgent()
    if name == "greedy_finish":
        return GreedyFinishAgent()
    if name == "project_rf_rule":
        return ProjectRFRuleBasedAgent()
    raise ValueError(
        f"unknown opponent {name!r}; expected one of {', '.join(OPPONENT_NAMES)}"
    )


def make_yutnori_env(
    *,
    opponent: str = "random",
    seed: int | None = None,
    learner_player: int = 0,
    starting_player: int | None = None,
    observation_mode: str = OBSERVATION_MODE_BASE,
    reward_mode: str = REWARD_MODE_TERMINAL,
) -> YutnoriEnv:
    """Create a single Gymnasium env with a seeded baseline opponent."""

    opponent_agent = make_opponent(opponent, seed=seed)
    env = YutnoriEnv(
        learner_player=learner_player,
        starting_player=starting_player,
        opponent_policy=opponent_agent.select_action,
        observation_mode=observation_mode,
        reward_mode=reward_mode,
    )
    if seed is not None:
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    return env


def make_yutnori_vec_env(
    *,
    opponent: str = "random",
    n_envs: int = 1,
    seed: int | None = None,
    learner_player: int = 0,
    starting_player: int | None = None,
    observation_mode: str = OBSERVATION_MODE_BASE,
    reward_mode: str = REWARD_MODE_TERMINAL,
) -> VecEnv:
    """Create a DummyVecEnv whose child envs expose ``action_masks()``."""

    if n_envs <= 0:
        raise ValueError("n_envs must be positive")

    env_fns: list[Callable[[], YutnoriEnv]] = []
    for rank in range(n_envs):
        env_seed = _rank_seed(seed, rank)

        def _init(env_seed: int | None = env_seed) -> YutnoriEnv:
            return make_yutnori_env(
                opponent=opponent,
                seed=env_seed,
                learner_player=learner_player,
                starting_player=starting_player,
                observation_mode=observation_mode,
                reward_mode=reward_mode,
            )

        env_fns.append(_init)

    vec_env = DummyVecEnv(env_fns)
    if seed is not None:
        vec_env.seed(seed)
    return vec_env


def _rank_seed(seed: int | None, rank: int) -> int | None:
    if seed is None:
        return None
    return seed + rank
