"""Mask-aware evaluation helpers for PPO policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from yutnori.training.env_factory import make_yutnori_env


class MaskablePredictor(Protocol):
    def predict(
        self,
        observation: np.ndarray,
        state: tuple[np.ndarray, ...] | None = None,
        episode_start: np.ndarray | None = None,
        deterministic: bool = False,
        action_masks: np.ndarray | None = None,
    ) -> tuple[np.ndarray, tuple[np.ndarray, ...] | None]:
        ...


@dataclass(frozen=True)
class PolicyEvaluationResult:
    opponent: str
    episodes: int
    learner_player: int
    wins: int
    losses: int
    win_rate: float
    average_turns: float
    average_decisions: float
    illegal_action_count: int
    starting_player_counts: dict[int, int]

    def to_dict(self) -> dict[str, object]:
        return {
            "opponent": self.opponent,
            "episodes": self.episodes,
            "learner_player": self.learner_player,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": self.win_rate,
            "average_turns": self.average_turns,
            "average_decisions": self.average_decisions,
            "illegal_action_count": self.illegal_action_count,
            "starting_player_counts": {
                str(player): count
                for player, count in sorted(self.starting_player_counts.items())
            },
        }


def evaluate_maskable_policy(
    model: MaskablePredictor,
    *,
    opponent: str,
    episodes: int,
    seed: int,
    learner_player: int = 0,
    deterministic: bool = True,
    max_decisions: int = 10_000,
) -> PolicyEvaluationResult:
    """Evaluate a MaskablePPO-style model while passing masks to predict()."""

    if episodes < 0:
        raise ValueError("episodes must be non-negative")
    if max_decisions <= 0:
        raise ValueError("max_decisions must be positive")

    env = make_yutnori_env(
        opponent=opponent,
        seed=seed,
        learner_player=learner_player,
    )
    wins = 0
    total_turns = 0
    total_decisions = 0
    illegal_action_count = 0
    starting_player_counts = {0: 0, 1: 0}

    try:
        for episode in range(episodes):
            obs, info = env.reset(seed=seed + episode)
            starting_player = int(info["starting_player"])
            starting_player_counts[starting_player] += 1
            terminated = False
            truncated = False

            while not (terminated or truncated):
                mask = env.action_masks()
                if not mask.any():
                    raise RuntimeError("non-terminal learner state has no legal actions")
                action, _state = model.predict(
                    obs,
                    deterministic=deterministic,
                    action_masks=mask,
                )
                action_int = int(np.asarray(action).item())
                if action_int < 0 or action_int >= mask.shape[0] or not mask[action_int]:
                    illegal_action_count += 1
                    raise ValueError(
                        f"model selected illegal action {action_int}; "
                        f"legal_actions={np.flatnonzero(mask).tolist()}"
                    )

                obs, _reward, terminated, truncated, info = env.step(action_int)
                if int(info["decision_count"]) > max_decisions:
                    raise RuntimeError(
                        f"evaluation game exceeded max_decisions={max_decisions}"
                    )

            winner = info["winner"]
            if winner == learner_player:
                wins += 1
            total_turns += int(info["turn_count"])
            total_decisions += int(info["decision_count"])
    finally:
        env.close()

    losses = episodes - wins
    return PolicyEvaluationResult(
        opponent=opponent,
        episodes=episodes,
        learner_player=learner_player,
        wins=wins,
        losses=losses,
        win_rate=0.0 if episodes == 0 else wins / episodes,
        average_turns=0.0 if episodes == 0 else total_turns / episodes,
        average_decisions=0.0 if episodes == 0 else total_decisions / episodes,
        illegal_action_count=illegal_action_count,
        starting_player_counts=starting_player_counts,
    )
