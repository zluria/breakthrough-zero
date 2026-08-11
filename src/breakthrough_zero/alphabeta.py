"""A small iterative-deepening alpha-beta baseline with absolute values."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter

from .game import BOARD_SIZE, PLAYER_1, GameState, Move


Clock = Callable[[], float]


class SearchTimeout(Exception):
    """Internal control flow used to discard an incomplete search depth."""


@dataclass(frozen=True, slots=True)
class AlphaBetaConfig:
    max_depth: int = 64
    material_weight: float = 0.65

    def __post_init__(self) -> None:
        if self.max_depth < 1:
            raise ValueError("max_depth must be positive")
        if not 0 <= self.material_weight <= 1:
            raise ValueError("material_weight must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class AlphaBetaResult:
    move: Move
    value: float
    depth: int
    nodes: int
    elapsed_seconds: float


class AlphaBetaAgent:
    """Choose moves by completing as many alpha-beta depths as time permits."""

    def __init__(
        self,
        config: AlphaBetaConfig = AlphaBetaConfig(),
        *,
        clock: Clock = perf_counter,
    ) -> None:
        self.config = config
        self.clock = clock
        self._nodes = 0

    def choose_move(self, state: GameState, time_limit_seconds: float) -> Move:
        return self.search(state, time_limit_seconds).move

    def search(
        self, state: GameState, time_limit_seconds: float
    ) -> AlphaBetaResult:
        """Return the last fully completed iterative-deepening result."""

        if time_limit_seconds <= 0:
            raise ValueError("time limit must be positive")
        if state.outcome is not None:
            raise ValueError("a terminal state has no move")

        moves = ordered_moves(state)
        if not moves:
            raise RuntimeError("a non-terminal state has no legal move")
        original = state.clone()
        best_move = moves[0]
        best_value = heuristic_value(state, self.config.material_weight)
        completed_depth = 0
        self._nodes = 0
        start = self.clock()
        deadline = start + time_limit_seconds

        for depth in range(1, self.config.max_depth + 1):
            try:
                move, value = self._search_root(
                    state, depth, deadline, preferred=best_move
                )
            except SearchTimeout:
                break
            best_move, best_value = move, value
            completed_depth = depth
            if value == (1.0 if state.to_move == PLAYER_1 else -1.0):
                break

        elapsed = self.clock() - start
        if state != original:
            raise RuntimeError("alpha-beta failed to restore the root state")
        return AlphaBetaResult(
            move=best_move,
            value=best_value,
            depth=completed_depth,
            nodes=self._nodes,
            elapsed_seconds=elapsed,
        )

    def _search_root(
        self,
        state: GameState,
        depth: int,
        deadline: float,
        *,
        preferred: Move,
    ) -> tuple[Move, float]:
        self._check_deadline(deadline)
        self._nodes += 1
        maximizing = state.to_move == PLAYER_1
        best_value = -1.0 if maximizing else 1.0
        best_move: Move | None = None
        alpha, beta = -1.0, 1.0

        for move in ordered_moves(state, preferred=preferred):
            undo = state.make_move(move, validate=False)
            try:
                value = self._alphabeta(state, depth - 1, alpha, beta, deadline)
            finally:
                state.unmake_move(move, undo)

            if best_move is None or (maximizing and value > best_value) or (
                not maximizing and value < best_value
            ):
                best_move, best_value = move, value
            if maximizing:
                alpha = max(alpha, best_value)
            else:
                beta = min(beta, best_value)
            if alpha >= beta:
                break

        if best_move is None:
            raise AssertionError("root search visited no move")
        return best_move, best_value

    def _alphabeta(
        self,
        state: GameState,
        depth: int,
        alpha: float,
        beta: float,
        deadline: float,
    ) -> float:
        self._check_deadline(deadline)
        self._nodes += 1
        if state.outcome is not None:
            return float(state.outcome)
        if depth == 0:
            return heuristic_value(state, self.config.material_weight)

        maximizing = state.to_move == PLAYER_1
        value = -1.0 if maximizing else 1.0
        for move in ordered_moves(state):
            undo = state.make_move(move, validate=False)
            try:
                child = self._alphabeta(state, depth - 1, alpha, beta, deadline)
            finally:
                state.unmake_move(move, undo)

            if maximizing:
                value = max(value, child)
                alpha = max(alpha, value)
            else:
                value = min(value, child)
                beta = min(beta, value)
            if alpha >= beta:
                break
        return value

    def _check_deadline(self, deadline: float) -> None:
        if self.clock() >= deadline:
            raise SearchTimeout


def heuristic_value(state: GameState, material_weight: float = 0.65) -> float:
    """Return a simple non-terminal score from absolute Player 1's view."""

    if state.outcome is not None:
        return float(state.outcome)
    if not 0 <= material_weight <= 1:
        raise ValueError("material_weight must be in [0, 1]")

    starting_pieces = state.rules.active_size * state.rules.starting_rows
    material = (state.p1.bit_count() - state.p2.bit_count()) / starting_pieces
    p1_progress = _progress(state.p1, state.rules.active_size, PLAYER_1)
    p2_progress = _progress(state.p2, state.rules.active_size, -PLAYER_1)
    advancement = (p1_progress - p2_progress) / starting_pieces
    value = material_weight * material + (1 - material_weight) * advancement
    return max(-0.99, min(0.99, value))


def ordered_moves(state: GameState, preferred: Move | None = None) -> list[Move]:
    """Order one principal move, goal moves, and captures before quiet moves."""

    opponent = state.pieces(-state.to_move)
    goal = state.rules.goal(state.to_move)

    def key(move: Move) -> tuple[int, int, int, int, int]:
        target = 1 << move.target
        return (
            -int(move == preferred),
            -int(bool(target & goal)),
            -int(bool(target & opponent)),
            move.source,
            move.target,
        )

    return sorted(state.legal_moves(), key=key)


def _progress(bitboard: int, size: int, player: int) -> float:
    total = 0.0
    while bitboard:
        bit = bitboard & -bitboard
        row = (bit.bit_length() - 1) // BOARD_SIZE
        total += row / (size - 1) if player == PLAYER_1 else (size - 1 - row) / (
            size - 1
        )
        bitboard ^= bit
    return total
