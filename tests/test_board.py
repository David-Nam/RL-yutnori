import pytest

from yutnori.core import Board, Cell, PieceStatus, Position, Route


def test_waiting_piece_enters_outer_route_by_steps():
    board = Board()

    assert board.move(Position.waiting(), 1).physical_cell == Cell.O1
    assert board.move(Position.waiting(), 2).physical_cell == Cell.O2
    assert board.move(Position.waiting(), 3).physical_cell == Cell.O3
    assert board.move(Position.waiting(), 4).physical_cell == Cell.O4

    mo_result = board.move(Position.waiting(), 5)
    assert mo_result.physical_cell == Cell.C1
    assert mo_result.route == Route.C1_DIAGONAL
    assert mo_result.index == 0
    assert mo_result.entered_shortcut


def test_exact_c1_and_c2_landing_enters_shortcut_next_route():
    board = Board()

    c1 = board.move(Position.at(Route.OUTER, 4), 1)
    assert c1.physical_cell == Cell.C1
    assert c1.route == Route.C1_DIAGONAL
    assert c1.index == 0
    assert c1.entered_shortcut

    c2 = board.move(Position.at(Route.OUTER, 9), 1)
    assert c2.physical_cell == Cell.C2
    assert c2.route == Route.C2_DIAGONAL
    assert c2.index == 0
    assert c2.entered_shortcut


def test_passing_outer_branch_points_keeps_outer_route():
    board = Board()

    passed_c1 = board.move(Position.at(Route.OUTER, 4), 2)
    assert passed_c1.physical_cell == Cell.O6
    assert passed_c1.route == Route.OUTER
    assert not passed_c1.entered_shortcut

    passed_c2 = board.move(Position.at(Route.OUTER, 9), 2)
    assert passed_c2.physical_cell == Cell.O11
    assert passed_c2.route == Route.OUTER
    assert not passed_c2.entered_shortcut


def test_exact_center_landing_switches_to_center_to_home_route():
    board = Board()

    center_from_c1 = board.move(Position.at(Route.C1_DIAGONAL, 2), 1)
    assert center_from_c1.physical_cell == Cell.CENTER
    assert center_from_c1.route == Route.CENTER_TO_HOME
    assert center_from_c1.index == 0
    assert center_from_c1.entered_shortcut

    center_from_c2 = board.move(Position.at(Route.C2_DIAGONAL, 2), 1)
    assert center_from_c2.physical_cell == Cell.CENTER
    assert center_from_c2.route == Route.CENTER_TO_HOME
    assert center_from_c2.index == 0
    assert center_from_c2.entered_shortcut


def test_passing_center_keeps_current_route():
    board = Board()

    passed_center = board.move(Position.at(Route.C1_DIAGONAL, 2), 2)
    assert passed_center.physical_cell == Cell.A3
    assert passed_center.route == Route.C1_DIAGONAL
    assert passed_center.index == 4
    assert not passed_center.entered_shortcut


def test_c3_is_not_a_shortcut_entry_from_outer_route():
    board = Board()

    c3 = board.move(Position.at(Route.OUTER, 14), 1)
    assert c3.physical_cell == Cell.C3
    assert c3.route == Route.OUTER
    assert c3.index == 15
    assert not c3.entered_shortcut


def test_exact_home_landing_is_not_finished():
    board = Board()

    from_o18 = board.move(Position.at(Route.OUTER, 18), 2)
    assert from_o18.status == PieceStatus.ON_BOARD
    assert from_o18.physical_cell == Cell.HOME
    assert from_o18.landed_on_home
    assert not from_o18.passed_home

    from_o19 = board.move(Position.at(Route.OUTER, 19), 1)
    assert from_o19.status == PieceStatus.ON_BOARD
    assert from_o19.physical_cell == Cell.HOME
    assert from_o19.landed_on_home
    assert not from_o19.passed_home


def test_passing_home_finishes_piece():
    board = Board()

    from_o18 = board.move(Position.at(Route.OUTER, 18), 3)
    assert from_o18.status == PieceStatus.FINISHED
    assert from_o18.physical_cell is None
    assert from_o18.passed_home

    from_o19 = board.move(Position.at(Route.OUTER, 19), 2)
    assert from_o19.status == PieceStatus.FINISHED
    assert from_o19.physical_cell is None
    assert from_o19.passed_home


def test_home_on_board_finishes_with_any_positive_move():
    result = Board().move(Position.home(), 1)

    assert result.status == PieceStatus.FINISHED
    assert result.physical_cell is None
    assert result.passed_home


def test_invalid_board_moves_raise_errors():
    board = Board()

    with pytest.raises(ValueError):
        board.move(Position.waiting(), 0)

    with pytest.raises(ValueError):
        board.move(Position.finished(), 1)
