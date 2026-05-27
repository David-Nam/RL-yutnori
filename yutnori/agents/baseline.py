"""Baseline agents for rule validation and evaluation."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Protocol

from yutnori.core import GameState, PieceStatus, YutResult, decode_action
from yutnori.core.yut import steps_for


class Agent(Protocol):
    name: str

    def select_action(self, state: GameState, legal_actions: list[int]) -> int:
        ...


@dataclass(frozen=True)
class ActionEvaluation:
    action: int
    piece_id: int
    yut_result: YutResult
    moved_count: int
    captured_count: int
    finished_count: int
    entered_shortcut: bool
    steps: int


class RandomAgent:
    name = "random"

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def select_action(self, _state: GameState, legal_actions: list[int]) -> int:
        if not legal_actions:
            raise ValueError("legal_actions must not be empty")
        return self._rng.choice(legal_actions)


class CaptureFirstAgent:
    name = "capture_first"

    def select_action(self, state: GameState, legal_actions: list[int]) -> int:
        if not legal_actions:
            raise ValueError("legal_actions must not be empty")
        evaluations = [evaluate_action(state, action) for action in legal_actions]
        capture_candidates = [
            evaluation
            for evaluation in evaluations
            if evaluation.captured_count > 0
        ]
        if capture_candidates:
            return max(capture_candidates, key=_capture_score).action
        return max(evaluations, key=_greedy_score).action


class GreedyFinishAgent:
    name = "greedy_finish"

    def select_action(self, state: GameState, legal_actions: list[int]) -> int:
        if not legal_actions:
            raise ValueError("legal_actions must not be empty")
        evaluations = [evaluate_action(state, action) for action in legal_actions]
        return max(evaluations, key=_greedy_score).action


def evaluate_action(state: GameState, action: int) -> ActionEvaluation:
    if not state.is_legal_action(action):
        raise ValueError(f"illegal action cannot be evaluated: {action}")

    actor = state.current_player
    opponent = 1 - actor
    piece_id, yut_result = decode_action(action)
    moving_piece_ids = state.stack_piece_ids(actor, piece_id)
    move_result = state.board.move(
        state.pieces[actor][piece_id],
        steps_for(yut_result),
    )

    captured_count = 0
    if (
        move_result.status == PieceStatus.ON_BOARD
        and move_result.physical_cell is not None
    ):
        captured_count = len(
            state.piece_ids_at_cell(opponent, move_result.physical_cell)
        )

    finished_count = (
        len(moving_piece_ids)
        if move_result.status == PieceStatus.FINISHED
        else 0
    )

    return ActionEvaluation(
        action=action,
        piece_id=piece_id,
        yut_result=yut_result,
        moved_count=len(moving_piece_ids),
        captured_count=captured_count,
        finished_count=finished_count,
        entered_shortcut=move_result.entered_shortcut,
        steps=steps_for(yut_result),
    )


def _capture_score(evaluation: ActionEvaluation) -> tuple[int, int, int, int, int]:
    return (
        evaluation.captured_count,
        evaluation.finished_count,
        evaluation.moved_count,
        evaluation.steps,
        -evaluation.action,
    )


def _greedy_score(evaluation: ActionEvaluation) -> tuple[int, int, int, int, int]:
    return (
        evaluation.finished_count,
        evaluation.moved_count,
        int(evaluation.entered_shortcut),
        evaluation.steps,
        -evaluation.action,
    )
