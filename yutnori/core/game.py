"""Game-state rules for two-player Yutnori."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from yutnori.core.board import Board, Cell, PieceStatus, Position
from yutnori.core.yut import YUT_ORDER, YutResult, YutSampler, is_bonus_result, steps_for

PLAYER_COUNT = 2
PIECES_PER_PLAYER = 4
ACTION_SIZE = PIECES_PER_PLAYER * len(YUT_ORDER)

YUT_TO_ACTION_ID: dict[YutResult, int] = {
    result: index for index, result in enumerate(YUT_ORDER)
}
ACTION_ID_TO_YUT: dict[int, YutResult] = {
    index: result for result, index in YUT_TO_ACTION_ID.items()
}


class Sampler(Protocol):
    def sample(self) -> YutResult:
        ...


def encode_action(piece_id: int, yut_result: YutResult) -> int:
    if piece_id < 0 or piece_id >= PIECES_PER_PLAYER:
        raise ValueError(f"piece_id must be in [0, {PIECES_PER_PLAYER})")
    return piece_id * len(YUT_ORDER) + YUT_TO_ACTION_ID[yut_result]


def decode_action(action: int) -> tuple[int, YutResult]:
    if action < 0 or action >= ACTION_SIZE:
        raise ValueError(f"action must be in [0, {ACTION_SIZE})")
    piece_id = action // len(YUT_ORDER)
    yut_type_id = action % len(YUT_ORDER)
    return piece_id, ACTION_ID_TO_YUT[yut_type_id]


def empty_pool() -> dict[YutResult, int]:
    return {result: 0 for result in YUT_ORDER}


@dataclass
class GameEvent:
    actor: int
    action: int
    piece_id: int
    yut_result: YutResult
    moved_piece_ids: list[int]
    captured: bool = False
    captured_count: int = 0
    captured_piece_ids: list[int] = field(default_factory=list)
    stacked: bool = False
    stack_size: int = 1
    finished_count: int = 0
    entered_shortcut: bool = False
    landed_on_home: bool = False
    passed_home: bool = False
    bonus_rolls: list[YutResult] = field(default_factory=list)
    turn_changed: bool = False
    winner: int | None = None
    pool_counts: dict[YutResult, int] = field(default_factory=empty_pool)


class GameState:
    """Mutable game-state engine for the confirmed project rules."""

    def __init__(
        self,
        *,
        starting_player: int = 0,
        board: Board | None = None,
        yut_sampler: Sampler | None = None,
    ) -> None:
        if starting_player < 0 or starting_player >= PLAYER_COUNT:
            raise ValueError(f"starting_player must be in [0, {PLAYER_COUNT})")
        self.board = board if board is not None else Board()
        self.yut_sampler = yut_sampler if yut_sampler is not None else YutSampler()
        self.current_player = starting_player
        self.pieces: list[list[Position]] = [
            [Position.waiting() for _ in range(PIECES_PER_PLAYER)]
            for _ in range(PLAYER_COUNT)
        ]
        self.pool_counts: dict[YutResult, int] = empty_pool()
        self.winner: int | None = None
        self.turn_count = 0
        self.decision_count = 0

    def start_turn(self) -> list[YutResult]:
        if self.winner is not None:
            raise ValueError("cannot start a turn after the game is finished")
        self.pool_counts = empty_pool()
        self.turn_count += 1
        return self._roll_until_non_bonus()

    def get_legal_actions(self, player: int | None = None) -> list[int]:
        target_player = self.current_player if player is None else player
        return [
            action
            for action in range(ACTION_SIZE)
            if self.is_legal_action(action, target_player)
        ]

    def is_legal_action(self, action: int, player: int | None = None) -> bool:
        if self.winner is not None:
            return False
        target_player = self.current_player if player is None else player
        if target_player != self.current_player:
            return False
        try:
            piece_id, yut_result = decode_action(action)
        except ValueError:
            return False
        if self.pool_counts[yut_result] <= 0:
            return False
        position = self.pieces[target_player][piece_id]
        if position.status == PieceStatus.FINISHED:
            return False
        return self.stack_representative(target_player, piece_id) == piece_id

    def apply_action(self, action: int) -> GameEvent:
        if not self.is_legal_action(action):
            raise ValueError(f"illegal action: {action}")

        actor = self.current_player
        opponent = 1 - actor
        piece_id, yut_result = decode_action(action)
        self.pool_counts[yut_result] -= 1
        self.decision_count += 1

        moving_piece_ids = self.stack_piece_ids(actor, piece_id)
        move_result = self.board.move(
            self.pieces[actor][piece_id],
            steps_for(yut_result),
        )
        for moving_piece_id in moving_piece_ids:
            self.pieces[actor][moving_piece_id] = move_result.position

        captured_piece_ids = self._capture_at_destination(opponent, move_result.position)
        destination_stack_ids = self._unify_own_stack(actor, move_result.position)
        finished_count = sum(
            1
            for moving_piece_id in moving_piece_ids
            if self.pieces[actor][moving_piece_id].status == PieceStatus.FINISHED
        )

        if self._all_finished(actor):
            self.winner = actor

        bonus_rolls: list[YutResult] = []
        if (
            self.winner is None
            and captured_piece_ids
            and not is_bonus_result(yut_result)
        ):
            bonus_rolls = self._roll_until_non_bonus()

        turn_changed = False
        if self.winner is None and (
            self._pool_total() == 0 or len(self.get_legal_actions(actor)) == 0
        ):
            self.pool_counts = empty_pool()
            self.current_player = opponent
            turn_changed = True
            self.start_turn()

        event = GameEvent(
            actor=actor,
            action=action,
            piece_id=piece_id,
            yut_result=yut_result,
            moved_piece_ids=moving_piece_ids,
            captured=bool(captured_piece_ids),
            captured_count=len(captured_piece_ids),
            captured_piece_ids=captured_piece_ids,
            stacked=len(destination_stack_ids) > len(moving_piece_ids),
            stack_size=max(1, len(destination_stack_ids)),
            finished_count=finished_count,
            entered_shortcut=move_result.entered_shortcut,
            landed_on_home=move_result.landed_on_home,
            passed_home=move_result.passed_home,
            bonus_rolls=bonus_rolls,
            turn_changed=turn_changed,
            winner=self.winner,
            pool_counts=self.pool_counts.copy(),
        )
        return event

    def stack_piece_ids(self, player: int, piece_id: int) -> list[int]:
        position = self.pieces[player][piece_id]
        if position.status != PieceStatus.ON_BOARD or position.physical_cell is None:
            return [piece_id]
        return self.piece_ids_at_cell(player, position.physical_cell)

    def stack_representative(self, player: int, piece_id: int) -> int:
        position = self.pieces[player][piece_id]
        if position.status != PieceStatus.ON_BOARD or position.physical_cell is None:
            return piece_id
        return min(self.piece_ids_at_cell(player, position.physical_cell))

    def piece_ids_at_cell(self, player: int, cell: Cell) -> list[int]:
        return [
            piece_id
            for piece_id, position in enumerate(self.pieces[player])
            if position.status == PieceStatus.ON_BOARD and position.physical_cell == cell
        ]

    def set_pool(self, *results: YutResult) -> None:
        self.pool_counts = empty_pool()
        for result in results:
            self.pool_counts[result] += 1

    def _roll_until_non_bonus(self) -> list[YutResult]:
        rolled: list[YutResult] = []
        while True:
            result = self.yut_sampler.sample()
            rolled.append(result)
            self.pool_counts[result] += 1
            if not is_bonus_result(result):
                return rolled

    def _capture_at_destination(self, opponent: int, position: Position) -> list[int]:
        if position.status != PieceStatus.ON_BOARD or position.physical_cell is None:
            return []
        captured_piece_ids = self.piece_ids_at_cell(opponent, position.physical_cell)
        for captured_piece_id in captured_piece_ids:
            self.pieces[opponent][captured_piece_id] = Position.waiting()
        return captured_piece_ids

    def _unify_own_stack(self, player: int, position: Position) -> list[int]:
        if position.status != PieceStatus.ON_BOARD or position.physical_cell is None:
            return []
        stack_piece_ids = self.piece_ids_at_cell(player, position.physical_cell)
        for stack_piece_id in stack_piece_ids:
            self.pieces[player][stack_piece_id] = position
        return stack_piece_ids

    def _pool_total(self) -> int:
        return sum(self.pool_counts.values())

    def _all_finished(self, player: int) -> bool:
        return all(
            position.status == PieceStatus.FINISHED for position in self.pieces[player]
        )
