"""Rules and action encoding for Breakthrough.

There are deliberately two coordinate systems:

* Game states and ``Move`` objects always use absolute Player-1 coordinates.
* The neural-network policy uses coordinates relative to the player to move.

Keeping that conversion in two small functions prevents orientation logic from
leaking into the rules or search code.
"""

from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np
from numpy.typing import NDArray

BOARD_SIZE = 8
NUM_SQUARES = BOARD_SIZE * BOARD_SIZE
POLICY_PLANES = 3
ACTION_SIZE = NUM_SQUARES * POLICY_PLANES

PLAYER_1 = 1
PLAYER_2 = -1
EMPTY = 0

FULL_BOARD = (1 << NUM_SQUARES) - 1
FILE_A = sum(1 << (row * BOARD_SIZE) for row in range(BOARD_SIZE))
FILE_H = FILE_A << (BOARD_SIZE - 1)

START_P1 = (1 << 16) - 1
START_P2 = START_P1 << 48
GOAL_P1 = ((1 << BOARD_SIZE) - 1) << (NUM_SQUARES - BOARD_SIZE)
GOAL_P2 = (1 << BOARD_SIZE) - 1


@dataclass(frozen=True, slots=True, order=True)
class Move:
    """A move between two absolute square indices in ``[0, 63]``."""

    source: int
    target: int

    def __post_init__(self) -> None:
        if not 0 <= self.source < NUM_SQUARES:
            raise ValueError(f"source square is outside the board: {self.source}")
        if not 0 <= self.target < NUM_SQUARES:
            raise ValueError(f"target square is outside the board: {self.target}")


@dataclass(frozen=True, slots=True)
class Undo:
    """The small amount of information needed to reverse one move."""

    mover: int
    captured: bool
    previous_winner: int


@dataclass(slots=True)
class GameState:
    """A mutable Breakthrough position backed by two Python-int bitboards.

    Player 1 starts on rows 0 and 1 and moves toward row 7. Player 2 starts
    on rows 6 and 7 and moves toward row 0. ``winner == 0`` means the game is
    still in progress; otherwise it is the winning player's number.
    """

    p1: int = START_P1
    p2: int = START_P2
    to_move: int = PLAYER_1
    winner: int = EMPTY
    ply: int = 0

    def __post_init__(self) -> None:
        if self.to_move not in (PLAYER_1, PLAYER_2):
            raise ValueError("to_move must be PLAYER_1 or PLAYER_2")
        if self.winner not in (EMPTY, PLAYER_1, PLAYER_2):
            raise ValueError("winner must be EMPTY, PLAYER_1, or PLAYER_2")
        if self.p1 & self.p2:
            raise ValueError("the two bitboards overlap")
        if (self.p1 | self.p2) & ~FULL_BOARD:
            raise ValueError("a bitboard contains squares outside the board")

    @property
    def outcome(self) -> int | None:
        """Return an absolute result: +1/-1 for the winner, or None."""

        return self.winner or None

    def clone(self) -> GameState:
        return GameState(self.p1, self.p2, self.to_move, self.winner, self.ply)

    def pieces(self, player: int) -> int:
        if player == PLAYER_1:
            return self.p1
        if player == PLAYER_2:
            return self.p2
        raise ValueError("player must be PLAYER_1 or PLAYER_2")

    def legal_moves(self) -> list[Move]:
        """Generate legal moves with bitboard shifts, without scanning 64 squares."""

        if self.winner:
            return []

        moves: list[Move] = []
        for targets, delta in self._target_bitboards():
            while targets:
                target_bit = targets & -targets
                target = target_bit.bit_length() - 1
                moves.append(Move(target - delta, target))
                targets ^= target_bit
        return moves

    def legal_action_indices(self) -> list[int]:
        return [self.policy_index(move) for move in self.legal_moves()]

    def has_legal_move(self) -> bool:
        return not self.winner and any(targets for targets, _ in self._target_bitboards())

    def random_legal_move(
        self, rng: random.Random, *, prefer_tactical: bool = False
    ) -> Move:
        """Sample a legal move without allocating the complete move list.

        With ``prefer_tactical``, immediate wins are sampled first, then
        captures, then all moves. Each chosen category is sampled uniformly.
        """

        if self.winner:
            raise ValueError("a terminal state has no legal move")

        targets = self._target_bitboards()
        if prefer_tactical:
            goal = GOAL_P1 if self.to_move == PLAYER_1 else GOAL_P2
            winning = tuple((bits & goal, delta) for bits, delta in targets)
            if any(bits for bits, _ in winning):
                return _sample_target(winning, rng)

            opponent = self.pieces(-self.to_move)
            captures = tuple((bits & opponent, delta) for bits, delta in targets)
            if any(bits for bits, _ in captures):
                return _sample_target(captures, rng)

        return _sample_target(targets, rng)

    def _target_bitboards(self) -> tuple[tuple[int, int], ...]:
        """Return ``(target_squares, signed_move_delta)`` for three directions."""

        ours = self.pieces(self.to_move)
        occupied = self.p1 | self.p2
        empty = FULL_BOARD ^ occupied
        not_ours = FULL_BOARD ^ ours

        if self.to_move == PLAYER_1:
            return (
                (((ours & ~FILE_A) << 7) & not_ours & FULL_BOARD, 7),
                (((ours << 8) & empty) & FULL_BOARD, 8),
                (((ours & ~FILE_H) << 9) & not_ours & FULL_BOARD, 9),
            )
        return (
            (((ours & ~FILE_A) >> 9) & not_ours, -9),
            (((ours >> 8) & empty), -8),
            (((ours & ~FILE_H) >> 7) & not_ours, -7),
        )

    def is_legal(self, move: Move) -> bool:
        """Check one move directly; this avoids constructing the legal-move list."""

        if self.winner:
            return False

        source_row, source_col = divmod(move.source, BOARD_SIZE)
        target_row, target_col = divmod(move.target, BOARD_SIZE)
        if target_row - source_row != self.to_move:
            return False

        column_step = target_col - source_col
        if abs(column_step) > 1:
            return False

        source_bit = 1 << move.source
        target_bit = 1 << move.target
        if not self.pieces(self.to_move) & source_bit:
            return False
        if self.pieces(self.to_move) & target_bit:
            return False
        if column_step == 0 and (self.p1 | self.p2) & target_bit:
            return False
        return True

    def make_move(self, move: Move, *, validate: bool = True) -> Undo:
        """Apply a move and return the information needed by ``unmake_move``."""

        if validate and not self.is_legal(move):
            raise ValueError(f"illegal move: {move}")

        mover = self.to_move
        source_bit = 1 << move.source
        target_bit = 1 << move.target
        captured = bool(self.pieces(-mover) & target_bit)
        undo = Undo(mover=mover, captured=captured, previous_winner=self.winner)

        if mover == PLAYER_1:
            self.p1 = (self.p1 ^ source_bit) | target_bit
            self.p2 &= ~target_bit
        else:
            self.p2 = (self.p2 ^ source_bit) | target_bit
            self.p1 &= ~target_bit

        target_row = move.target // BOARD_SIZE
        reached_goal = (mover == PLAYER_1 and target_row == 7) or (
            mover == PLAYER_2 and target_row == 0
        )

        self.ply += 1
        if reached_goal:
            self.winner = mover
            return undo

        # Change turns only while the game continues. We temporarily give the
        # opponent the turn to test the second terminal rule: no legal reply.
        self.to_move = -mover
        if not self.has_legal_move():
            self.winner = mover
            self.to_move = mover
        return undo

    def unmake_move(self, move: Move, undo: Undo) -> None:
        """Reverse the most recently applied move."""

        mover = undo.mover
        source_bit = 1 << move.source
        target_bit = 1 << move.target

        if mover == PLAYER_1:
            self.p1 = (self.p1 ^ target_bit) | source_bit
            if undo.captured:
                self.p2 |= target_bit
        else:
            self.p2 = (self.p2 ^ target_bit) | source_bit
            if undo.captured:
                self.p1 |= target_bit

        self.to_move = mover
        self.winner = undo.previous_winner
        self.ply -= 1

    def policy_index(self, move: Move) -> int:
        """Map an absolute move to ``source x relative direction``.

        Player 2's move is rotated 180 degrees first. In canonical coordinates
        every legal move therefore advances exactly one row.
        """

        source = _canonical_square(move.source, self.to_move)
        target = _canonical_square(move.target, self.to_move)
        source_row, source_col = divmod(source, BOARD_SIZE)
        target_row, target_col = divmod(target, BOARD_SIZE)
        row_step = target_row - source_row
        column_step = target_col - source_col
        if row_step != 1 or column_step not in (-1, 0, 1):
            raise ValueError(f"move cannot be represented by the policy head: {move}")
        return source * POLICY_PLANES + (column_step + 1)

    def decode_policy_index(self, action: int) -> Move:
        """Decode one policy index using the state's current player."""

        if not 0 <= action < ACTION_SIZE:
            raise ValueError(f"policy index is outside [0, {ACTION_SIZE}): {action}")

        source, plane = divmod(action, POLICY_PLANES)
        source_row, source_col = divmod(source, BOARD_SIZE)
        target_row = source_row + 1
        target_col = source_col + plane - 1
        if target_row >= BOARD_SIZE or not 0 <= target_col < BOARD_SIZE:
            raise ValueError(f"policy index points outside the board: {action}")

        target = target_row * BOARD_SIZE + target_col
        return Move(
            _canonical_square(source, self.to_move),
            _canonical_square(target, self.to_move),
        )

    def encode(self) -> NDArray[np.float32]:
        """Encode mover-oriented pieces while preserving absolute identity.

        Channels 0 and 1 are our pieces and their pieces after rotating a
        Player 2 position. Channel 2 is all ones when the mover is Player 1 and
        all zeros when the mover is Player 2. The policy therefore gets one
        spatial meaning for forward, while the value head can learn an absolute
        Player 1 value directly. No value sign conversion is needed.
        """

        planes = np.zeros((BOARD_SIZE, BOARD_SIZE, 3), dtype=np.float32)
        for channel, bitboard in enumerate(
            (self.pieces(self.to_move), self.pieces(-self.to_move))
        ):
            while bitboard:
                bit = bitboard & -bitboard
                square = bit.bit_length() - 1
                canonical = _canonical_square(square, self.to_move)
                row, col = divmod(canonical, BOARD_SIZE)
                planes[row, col, channel] = 1.0
                bitboard ^= bit
        if self.to_move == PLAYER_1:
            planes[:, :, 2] = 1.0
        return planes


def _canonical_square(square: int, player: int) -> int:
    """Rotate Player 2's view by 180 degrees; the transform is self-inverse."""

    return square if player == PLAYER_1 else (NUM_SQUARES - 1 - square)


def _sample_target(
    target_bitboards: tuple[tuple[int, int], ...], rng: random.Random
) -> Move:
    """Select one set bit uniformly across ``(targets, delta)`` groups."""

    move_count = sum(bits.bit_count() for bits, _ in target_bitboards)
    if move_count == 0:
        raise ValueError("a non-terminal state has no legal move")

    choice = rng.randrange(move_count)
    for bits, delta in target_bitboards:
        count = bits.bit_count()
        if choice >= count:
            choice -= count
            continue

        for _ in range(choice):
            bits &= bits - 1
        target_bit = bits & -bits
        target = target_bit.bit_length() - 1
        return Move(target - delta, target)

    raise AssertionError("unreachable random-move selection state")
