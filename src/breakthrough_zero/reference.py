"""Obviously-correct, intentionally slow rules used to test the bitboards."""

from __future__ import annotations

from .game import BOARD_SIZE, PLAYER_1, GameState, Move


def reference_legal_moves(state: GameState) -> list[Move]:
    """Generate moves by scanning the board and spelling out every rule."""

    if state.winner:
        return []

    moves: list[Move] = []
    ours = state.pieces(state.to_move)
    occupied = state.p1 | state.p2

    size = state.rules.active_size
    for row in range(size):
        for col in range(size):
            source = row * BOARD_SIZE + col
            if not (ours & (1 << source)):
                continue

            target_row = row + (1 if state.to_move == PLAYER_1 else -1)
            if not 0 <= target_row < size:
                continue

            for column_step in (-1, 0, 1):
                target_col = col + column_step
                if not 0 <= target_col < size:
                    continue

                target = target_row * BOARD_SIZE + target_col
                target_bit = 1 << target
                if ours & target_bit:
                    continue
                if column_step == 0 and occupied & target_bit:
                    continue
                moves.append(Move(source, target))

    return moves
