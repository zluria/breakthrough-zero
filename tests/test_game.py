from __future__ import annotations

import random
import unittest

import numpy as np

from breakthrough_zero.game import (
    ACTION_SIZE,
    MINI_RULES,
    PLAYER_1,
    PLAYER_2,
    GameState,
    Move,
)
from breakthrough_zero.reference import reference_legal_moves


class GameTests(unittest.TestCase):
    def test_initial_position_has_22_legal_moves(self) -> None:
        self.assertEqual(len(GameState().legal_moves()), 22)

    def test_mini_initial_position_has_one_row_and_13_legal_moves(self) -> None:
        state = GameState.initial(MINI_RULES)
        self.assertEqual(state.p1.bit_count(), 5)
        self.assertEqual(state.p2.bit_count(), 5)
        self.assertEqual(len(state.legal_moves()), 13)
        self.assertTrue(all(state.is_legal(move) for move in state.legal_moves()))

    def test_policy_round_trip_for_both_players(self) -> None:
        rng = random.Random(7)
        state = GameState()

        for _ in range(80):
            legal_moves = state.legal_moves()
            if not legal_moves:
                break
            indices = [state.policy_index(move) for move in legal_moves]
            self.assertEqual(len(indices), len(set(indices)))
            self.assertTrue(all(0 <= index < ACTION_SIZE for index in indices))
            self.assertEqual(
                [state.decode_policy_index(index) for index in indices], legal_moves
            )
            state.make_move(rng.choice(legal_moves))

    def test_player_2_policy_uses_forward_relative_coordinates(self) -> None:
        state = GameState(p1=1 << 8, p2=1 << 55, to_move=PLAYER_2)
        black_forward = Move(55, 47)
        canonical_source = 63 - black_forward.source
        self.assertEqual(
            state.policy_index(black_forward), canonical_source * 3 + 1
        )
        self.assertEqual(
            state.decode_policy_index(state.policy_index(black_forward)),
            black_forward,
        )

    def test_mini_policy_round_trip_rotates_only_the_active_board(self) -> None:
        state = GameState(
            p1=1 << 8,
            p2=1 << 34,
            to_move=PLAYER_2,
            rules=MINI_RULES,
        )
        black_forward = Move(34, 26)
        # Active-board rotation maps row 4, column 2 to row 0, column 2.
        self.assertEqual(state.policy_index(black_forward), 2 * 3 + 1)
        self.assertEqual(
            state.decode_policy_index(state.policy_index(black_forward)),
            black_forward,
        )

    def test_straight_move_cannot_capture_but_diagonal_move_can(self) -> None:
        state = GameState(
            p1=1 << 27, p2=(1 << 35) | (1 << 36), to_move=PLAYER_1
        )
        self.assertFalse(state.is_legal(Move(27, 35)))
        self.assertTrue(state.is_legal(Move(27, 36)))

    def test_make_and_unmake_restore_state_exactly(self) -> None:
        state = GameState()
        original = state.clone()
        move = state.legal_moves()[5]
        undo = state.make_move(move)
        state.unmake_move(move, undo)
        self.assertEqual(state, original)

    def test_goal_row_sets_absolute_winner(self) -> None:
        state = GameState(p1=1 << 48, p2=1 << 15, to_move=PLAYER_1)
        state.make_move(Move(48, 56))
        self.assertEqual(state.outcome, PLAYER_1)
        self.assertEqual(state.to_move, PLAYER_1)
        self.assertEqual(state.legal_moves(), [])

    def test_mini_goal_row_sets_absolute_winner_without_switching_turn(self) -> None:
        state = GameState(
            p1=1 << 24,
            p2=1 << 4,
            to_move=PLAYER_1,
            rules=MINI_RULES,
        )
        state.make_move(Move(24, 32))
        self.assertEqual(state.outcome, PLAYER_1)
        self.assertEqual(state.to_move, PLAYER_1)

    def test_terminal_move_does_not_switch_turn_for_either_player(self) -> None:
        positions = (
            (GameState(p1=1 << 48, p2=1 << 15, to_move=PLAYER_1), Move(48, 56)),
            (GameState(p1=1 << 48, p2=1 << 15, to_move=PLAYER_2), Move(15, 7)),
        )
        for state, move in positions:
            original = state.clone()
            mover = state.to_move
            undo = state.make_move(move)
            self.assertEqual(state.outcome, mover)
            self.assertEqual(state.to_move, mover)
            state.unmake_move(move, undo)
            self.assertEqual(state, original)

    def test_no_legal_reply_is_terminal_without_switching_turn(self) -> None:
        # Capturing Player 2's last piece leaves it with no legal reply.
        state = GameState(p1=1 << 8, p2=1 << 17, to_move=PLAYER_1)
        state.make_move(Move(8, 17))
        self.assertEqual(state.outcome, PLAYER_1)
        self.assertEqual(state.to_move, PLAYER_1)

    def test_encode_is_canonical_to_the_current_player(self) -> None:
        white = GameState(p1=1 << 8, p2=1 << 55, to_move=PLAYER_1)
        black = GameState(p1=1 << 8, p2=1 << 55, to_move=PLAYER_2)
        self.assertEqual(white.encode()[1, 0, 0], 1)
        self.assertEqual(black.encode()[1, 0, 0], 1)
        self.assertEqual(white.encode().shape, (8, 8, 3))
        self.assertTrue(np.all(white.encode()[:, :, 2] == 1))
        self.assertTrue(np.all(black.encode()[:, :, 2] == 0))

    def test_invalid_policy_slots_fail_loudly(self) -> None:
        state = GameState()
        with self.assertRaises(ValueError):
            state.decode_policy_index((7 * 8) * 3 + 1)

        mini = GameState.initial(MINI_RULES)
        with self.assertRaises(ValueError):
            mini.decode_policy_index((0 * 8 + 5) * 3 + 1)

    def test_bitboards_match_slow_reference_on_random_games(self) -> None:
        for initial in (GameState(), GameState.initial(MINI_RULES)):
            for seed in range(40):
                rng = random.Random(seed)
                state = initial.clone()
                for _ in range(100):
                    fast = sorted(state.legal_moves())
                    slow = sorted(reference_legal_moves(state))
                    self.assertEqual(fast, slow)
                    self.assertTrue(all(state.is_legal(move) for move in fast))
                    if not fast:
                        break
                    state.make_move(rng.choice(fast))


if __name__ == "__main__":
    unittest.main()
