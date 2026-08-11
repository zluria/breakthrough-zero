from __future__ import annotations

import unittest

from breakthrough_zero.alphabeta import (
    AlphaBetaAgent,
    AlphaBetaConfig,
    heuristic_value,
)
from breakthrough_zero.game import MINI_RULES, PLAYER_1, PLAYER_2, GameState, Move
from breakthrough_zero.symmetry import Symmetry, transform_state


class StepClock:
    def __init__(self, step: float) -> None:
        self.now = 0.0
        self.step = step

    def __call__(self) -> float:
        current = self.now
        self.now += self.step
        return current


class AlphaBetaTests(unittest.TestCase):
    def test_heuristic_is_absolute_and_player_swap_negates_it(self) -> None:
        state = GameState(
            p1=(1 << 8) | (1 << 18),
            p2=1 << 27,
            rules=MINI_RULES,
        )
        swapped = transform_state(state, Symmetry.SWAP_PLAYERS)
        self.assertAlmostEqual(heuristic_value(state), -heuristic_value(swapped))
        self.assertGreater(heuristic_value(state), 0)

    def test_heuristic_reserves_exact_endpoints_for_terminal_results(self) -> None:
        state = GameState.initial(MINI_RULES)
        self.assertLess(abs(heuristic_value(state)), 1)
        terminal = GameState(
            p1=1 << 32,
            p2=1 << 9,
            to_move=PLAYER_1,
            winner=PLAYER_1,
            rules=MINI_RULES,
        )
        self.assertEqual(heuristic_value(terminal), 1)

    def test_search_finds_immediate_win_for_both_players(self) -> None:
        positions = (
            (
                GameState(
                    p1=1 << 24,
                    p2=1 << 4,
                    to_move=PLAYER_1,
                    rules=MINI_RULES,
                ),
                Move(24, 32),
                1.0,
            ),
            (
                GameState(
                    p1=1 << 24,
                    p2=1 << 8,
                    to_move=PLAYER_2,
                    rules=MINI_RULES,
                ),
                Move(8, 0),
                -1.0,
            ),
        )
        for state, winning_move, value in positions:
            original = state.clone()
            result = AlphaBetaAgent(
                AlphaBetaConfig(max_depth=3)
            ).search(state, 0.5)
            self.assertEqual(result.move, winning_move)
            self.assertEqual(result.value, value)
            self.assertEqual(state, original)

    def test_timeout_discards_partial_depth_and_restores_every_move(self) -> None:
        state = GameState.initial(MINI_RULES)
        original = state.clone()
        result = AlphaBetaAgent(
            AlphaBetaConfig(max_depth=10), clock=StepClock(0.001)
        ).search(state, 0.003)
        self.assertIn(result.move, state.legal_moves())
        self.assertEqual(result.depth, 0)
        self.assertEqual(state, original)

    def test_player_2_search_returns_an_absolute_negative_value(self) -> None:
        state = GameState(
            p1=1 << 24,
            p2=1 << 8,
            to_move=PLAYER_2,
            rules=MINI_RULES,
        )
        result = AlphaBetaAgent(AlphaBetaConfig(max_depth=2)).search(state, 0.5)
        self.assertEqual(result.value, -1.0)
        self.assertEqual(state.to_move, PLAYER_2)


if __name__ == "__main__":
    unittest.main()
