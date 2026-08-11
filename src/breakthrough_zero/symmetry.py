"""The four exact symmetries of Breakthrough training examples.

The two independent operations are:

1. Reflect left and right.
2. Swap the players *and* reflect top and bottom.

The rank reflection in operation 2 is essential. Relabelling the players alone
would reverse which direction their pieces are allowed to move.
"""

from __future__ import annotations

from enum import Enum

from .game import BOARD_SIZE, GameState, Move


class Symmetry(Enum):
    """Two binary choices give four transformations."""

    IDENTITY = (False, False)
    MIRROR_LEFT_RIGHT = (True, False)
    SWAP_PLAYERS = (False, True)
    SWAP_AND_MIRROR = (True, True)

    @property
    def mirror_left_right(self) -> bool:
        return self.value[0]

    @property
    def swap_players(self) -> bool:
        return self.value[1]


def transform_square(square: int, symmetry: Symmetry) -> int:
    row, col = divmod(square, BOARD_SIZE)
    if symmetry.swap_players:
        row = BOARD_SIZE - 1 - row
    if symmetry.mirror_left_right:
        col = BOARD_SIZE - 1 - col
    return row * BOARD_SIZE + col


def transform_move(move: Move, symmetry: Symmetry) -> Move:
    return Move(
        transform_square(move.source, symmetry),
        transform_square(move.target, symmetry),
    )


def transform_outcome(outcome: int | None, symmetry: Symmetry) -> int | None:
    if outcome is None:
        return None
    return -outcome if symmetry.swap_players else outcome


def transform_state(state: GameState, symmetry: Symmetry) -> GameState:
    transformed_p1 = 0
    transformed_p2 = 0

    for player, bitboard in ((1, state.p1), (-1, state.p2)):
        transformed_player = -player if symmetry.swap_players else player
        while bitboard:
            bit = bitboard & -bitboard
            square = bit.bit_length() - 1
            transformed_bit = 1 << transform_square(square, symmetry)
            if transformed_player == 1:
                transformed_p1 |= transformed_bit
            else:
                transformed_p2 |= transformed_bit
            bitboard ^= bit

    return GameState(
        p1=transformed_p1,
        p2=transformed_p2,
        to_move=-state.to_move if symmetry.swap_players else state.to_move,
        winner=(
            -state.winner
            if symmetry.swap_players and state.winner
            else state.winner
        ),
        ply=state.ply,
    )

