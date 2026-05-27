"""Gymnasium-compatible wrapper for the Yutnori game state."""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from yutnori.core import (
    ACTION_SIZE,
    PIECES_PER_PLAYER,
    PLAYER_COUNT,
    GameEvent,
    GameState,
    PieceStatus,
    Position,
    YUT_ORDER,
    YutSampler,
)
from yutnori.core.game import Sampler

POSITION_WAITING = 29
POSITION_FINISHED = 30
OBSERVATION_SIZE = (4 + 4 + 16) * 2 + len(YUT_ORDER)

OpponentPolicy = Callable[[GameState, list[int]], int]
YutSamplerFactory = Callable[[random.Random], Sampler]


class YutnoriEnv(gym.Env[np.ndarray, int]):
    """Single-learner Gymnasium view over a two-player Yutnori game.

    The environment only returns decision states for ``learner_player``.
    Opponent turns are advanced internally with ``opponent_policy``.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        learner_player: int = 0,
        starting_player: int | None = None,
        opponent_policy: OpponentPolicy | None = None,
        yut_sampler_factory: YutSamplerFactory | None = None,
    ) -> None:
        if learner_player < 0 or learner_player >= PLAYER_COUNT:
            raise ValueError(f"learner_player must be in [0, {PLAYER_COUNT})")
        if starting_player is not None and (
            starting_player < 0 or starting_player >= PLAYER_COUNT
        ):
            raise ValueError(f"starting_player must be in [0, {PLAYER_COUNT})")

        self.learner_player = learner_player
        self._fixed_starting_player = starting_player
        self._opponent_policy = opponent_policy
        self._yut_sampler_factory = yut_sampler_factory
        self._rng = random.Random()
        self.state: GameState | None = None

        self.action_space = spaces.Discrete(ACTION_SIZE)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1_000_000.0,
            shape=(OBSERVATION_SIZE,),
            dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._rng = random.Random(seed)

        starting_player = self._resolve_starting_player(options)
        sampler = self._create_yut_sampler()
        self.state = GameState(
            starting_player=starting_player,
            yut_sampler=sampler,
        )
        initial_rolls = self.state.start_turn()
        opponent_events = self._advance_opponent_turns()

        info = self._base_info()
        info.update(
            {
                "starting_player": starting_player,
                "initial_rolls": [result.value for result in initial_rolls],
                "opponent_events": [self._event_to_dict(event) for event in opponent_events],
            }
        )
        return self._get_obs(), info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        state = self._require_state()
        if state.current_player != self.learner_player:
            raise RuntimeError("step called while learner is not the current player")
        if not state.is_legal_action(int(action), self.learner_player):
            raise ValueError(f"illegal learner action: {action}")

        learner_event = state.apply_action(int(action))
        opponent_events = self._advance_opponent_turns()
        terminated = state.winner is not None
        reward = self._reward()
        info = self._base_info()
        info.update(
            {
                "learner_event": self._event_to_dict(learner_event),
                "opponent_events": [
                    self._event_to_dict(event) for event in opponent_events
                ],
            }
        )
        return self._get_obs(), reward, terminated, False, info

    def action_masks(self) -> np.ndarray:
        state = self.state
        if state is None or state.winner is not None:
            return np.zeros(ACTION_SIZE, dtype=np.bool_)
        if state.current_player != self.learner_player:
            return np.zeros(ACTION_SIZE, dtype=np.bool_)
        return np.array(
            [state.is_legal_action(action, self.learner_player) for action in range(ACTION_SIZE)],
            dtype=np.bool_,
        )

    def render(self) -> None:
        return None

    def _resolve_starting_player(self, options: dict[str, Any] | None) -> int:
        if options is not None and "starting_player" in options:
            starting_player = int(options["starting_player"])
            if starting_player < 0 or starting_player >= PLAYER_COUNT:
                raise ValueError(f"starting_player must be in [0, {PLAYER_COUNT})")
            return starting_player
        if self._fixed_starting_player is not None:
            return self._fixed_starting_player
        return self._rng.randrange(PLAYER_COUNT)

    def _create_yut_sampler(self) -> Sampler:
        if self._yut_sampler_factory is not None:
            return self._yut_sampler_factory(self._rng)
        return YutSampler(rng=self._rng)

    def _advance_opponent_turns(self) -> list[GameEvent]:
        state = self._require_state()
        events: list[GameEvent] = []
        while state.winner is None and state.current_player != self.learner_player:
            legal_actions = state.get_legal_actions()
            if not legal_actions:
                raise RuntimeError("opponent has no legal actions during its turn")
            action = self._select_opponent_action(state, legal_actions)
            if action not in legal_actions:
                raise ValueError(f"opponent selected illegal action: {action}")
            events.append(state.apply_action(action))
        return events

    def _select_opponent_action(self, state: GameState, legal_actions: list[int]) -> int:
        if self._opponent_policy is not None:
            return int(self._opponent_policy(state, legal_actions))
        return self._rng.choice(legal_actions)

    def _get_obs(self) -> np.ndarray:
        state = self._require_state()
        return encode_observation(state, self.learner_player)

    def _reward(self) -> float:
        state = self._require_state()
        if state.winner is None:
            return 0.0
        return 1.0 if state.winner == self.learner_player else -1.0

    def _base_info(self) -> dict[str, Any]:
        state = self._require_state()
        return {
            "learner_player": self.learner_player,
            "current_player": state.current_player,
            "winner": state.winner,
            "turn_count": state.turn_count,
            "decision_count": state.decision_count,
            "action_mask": self.action_masks(),
        }

    def _require_state(self) -> GameState:
        if self.state is None:
            raise RuntimeError("environment must be reset before use")
        return self.state

    def _event_to_dict(self, event: GameEvent) -> dict[str, Any]:
        return {
            "actor": event.actor,
            "action": event.action,
            "piece_id": event.piece_id,
            "yut_result": event.yut_result.value,
            "moved_piece_ids": event.moved_piece_ids,
            "captured": event.captured,
            "captured_count": event.captured_count,
            "captured_piece_ids": event.captured_piece_ids,
            "stacked": event.stacked,
            "stack_size": event.stack_size,
            "finished_count": event.finished_count,
            "entered_shortcut": event.entered_shortcut,
            "landed_on_home": event.landed_on_home,
            "passed_home": event.passed_home,
            "bonus_rolls": [result.value for result in event.bonus_rolls],
            "turn_changed": event.turn_changed,
            "winner": event.winner,
            "pool_counts": {
                result.value: count for result, count in event.pool_counts.items()
            },
        }


def encode_observation(state: GameState, player: int) -> np.ndarray:
    opponent = 1 - player
    values: list[float] = []
    values.extend(_position_values(state, player))
    values.extend(_status_values(state, player))
    values.extend(_stack_matrix_values(state, player))
    values.extend(_position_values(state, opponent))
    values.extend(_status_values(state, opponent))
    values.extend(_stack_matrix_values(state, opponent))
    values.extend(float(state.pool_counts[result]) for result in YUT_ORDER)
    return np.array(values, dtype=np.float32)


def _position_values(state: GameState, player: int) -> list[float]:
    return [float(_position_value(position)) for position in state.pieces[player]]


def _position_value(position: Position) -> int:
    if position.status == PieceStatus.WAITING:
        return POSITION_WAITING
    if position.status == PieceStatus.FINISHED:
        return POSITION_FINISHED
    if position.physical_cell is None:
        raise ValueError("on-board position requires physical_cell")
    return int(position.physical_cell)


def _status_values(state: GameState, player: int) -> list[float]:
    return [float(_status_value(position.status)) for position in state.pieces[player]]


def _status_value(status: PieceStatus) -> int:
    if status == PieceStatus.WAITING:
        return 0
    if status == PieceStatus.ON_BOARD:
        return 1
    if status == PieceStatus.FINISHED:
        return 2
    raise ValueError(f"unknown piece status: {status}")


def _stack_matrix_values(state: GameState, player: int) -> list[float]:
    values: list[float] = []
    for left in state.pieces[player]:
        for right in state.pieces[player]:
            values.append(float(_same_stack(left, right)))
    return values


def _same_stack(left: Position, right: Position) -> bool:
    return (
        left.status == PieceStatus.ON_BOARD
        and right.status == PieceStatus.ON_BOARD
        and left.physical_cell is not None
        and left.physical_cell == right.physical_cell
    )
