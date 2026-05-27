"""Board coordinates and movement for the project Yutnori rules."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum


class PieceStatus(str, Enum):
    WAITING = "WAITING"
    ON_BOARD = "ON_BOARD"
    FINISHED = "FINISHED"


class Route(str, Enum):
    OUTER = "OUTER"
    C1_DIAGONAL = "C1_DIAGONAL"
    C2_DIAGONAL = "C2_DIAGONAL"
    CENTER_TO_HOME = "CENTER_TO_HOME"


class Cell(IntEnum):
    HOME = 0
    O1 = 1
    O2 = 2
    O3 = 3
    O4 = 4
    C1 = 5
    O6 = 6
    O7 = 7
    O8 = 8
    O9 = 9
    C2 = 10
    O11 = 11
    O12 = 12
    O13 = 13
    O14 = 14
    C3 = 15
    O16 = 16
    O17 = 17
    O18 = 18
    O19 = 19
    A1 = 20
    A2 = 21
    CENTER = 22
    A3 = 23
    A4 = 24
    B1 = 25
    B2 = 26
    B3 = 27
    B4 = 28


ROUTES: dict[Route, tuple[Cell, ...]] = {
    Route.OUTER: (
        Cell.HOME,
        Cell.O1,
        Cell.O2,
        Cell.O3,
        Cell.O4,
        Cell.C1,
        Cell.O6,
        Cell.O7,
        Cell.O8,
        Cell.O9,
        Cell.C2,
        Cell.O11,
        Cell.O12,
        Cell.O13,
        Cell.O14,
        Cell.C3,
        Cell.O16,
        Cell.O17,
        Cell.O18,
        Cell.O19,
        Cell.HOME,
    ),
    Route.C1_DIAGONAL: (
        Cell.C1,
        Cell.A1,
        Cell.A2,
        Cell.CENTER,
        Cell.A3,
        Cell.A4,
        Cell.C3,
        Cell.O16,
        Cell.O17,
        Cell.O18,
        Cell.O19,
        Cell.HOME,
    ),
    Route.C2_DIAGONAL: (
        Cell.C2,
        Cell.B1,
        Cell.B2,
        Cell.CENTER,
        Cell.B3,
        Cell.B4,
        Cell.HOME,
    ),
    Route.CENTER_TO_HOME: (
        Cell.CENTER,
        Cell.B3,
        Cell.B4,
        Cell.HOME,
    ),
}


@dataclass(frozen=True)
class Position:
    status: PieceStatus
    route: Route | None = None
    index: int | None = None
    physical_cell: Cell | None = None

    @classmethod
    def waiting(cls) -> "Position":
        return cls(status=PieceStatus.WAITING)

    @classmethod
    def finished(cls) -> "Position":
        return cls(status=PieceStatus.FINISHED)

    @classmethod
    def home(cls) -> "Position":
        return cls(
            status=PieceStatus.ON_BOARD,
            route=Route.OUTER,
            index=0,
            physical_cell=Cell.HOME,
        )

    @classmethod
    def at(cls, route: Route, index: int) -> "Position":
        cells = ROUTES[route]
        if index < 0 or index >= len(cells):
            raise ValueError(f"invalid index {index} for route {route.value}")
        return cls(
            status=PieceStatus.ON_BOARD,
            route=route,
            index=index,
            physical_cell=cells[index],
        )


@dataclass(frozen=True)
class MoveResult:
    position: Position
    entered_shortcut: bool = False
    landed_on_home: bool = False
    passed_home: bool = False

    @property
    def status(self) -> PieceStatus:
        return self.position.status

    @property
    def route(self) -> Route | None:
        return self.position.route

    @property
    def index(self) -> int | None:
        return self.position.index

    @property
    def physical_cell(self) -> Cell | None:
        return self.position.physical_cell


class Board:
    """Move pieces on the confirmed 29-cell board."""

    def move(self, position: Position, steps: int) -> MoveResult:
        if steps <= 0:
            raise ValueError("steps must be positive")
        if position.status == PieceStatus.FINISHED:
            raise ValueError("finished pieces cannot move")
        if position.status == PieceStatus.WAITING:
            return self._move_from(Route.OUTER, 0, steps)
        if position.status != PieceStatus.ON_BOARD:
            raise ValueError(f"unknown piece status: {position.status}")
        if position.physical_cell == Cell.HOME:
            return MoveResult(position=Position.finished(), passed_home=True)
        if position.route is None or position.index is None:
            raise ValueError("on-board position requires route and index")
        return self._move_from(position.route, position.index, steps)

    def _move_from(self, route: Route, index: int, steps: int) -> MoveResult:
        route_cells = ROUTES[route]
        target_index = index + steps
        home_index = len(route_cells) - 1

        if target_index > home_index:
            return MoveResult(position=Position.finished(), passed_home=True)

        target_cell = route_cells[target_index]
        if target_index == home_index and target_cell == Cell.HOME:
            return MoveResult(position=Position.home(), landed_on_home=True)

        new_route = route
        new_index = target_index
        entered_shortcut = False

        if route == Route.OUTER and target_cell == Cell.C1:
            new_route = Route.C1_DIAGONAL
            new_index = 0
            entered_shortcut = True
        elif route == Route.OUTER and target_cell == Cell.C2:
            new_route = Route.C2_DIAGONAL
            new_index = 0
            entered_shortcut = True
        elif target_cell == Cell.CENTER:
            new_route = Route.CENTER_TO_HOME
            new_index = 0
            entered_shortcut = True

        return MoveResult(
            position=Position.at(new_route, new_index),
            entered_shortcut=entered_shortcut,
        )
