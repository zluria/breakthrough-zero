from __future__ import annotations

import unittest

import numpy as np

from breakthrough_zero.data import ActionStatistics, PositionRecord
from breakthrough_zero.game import MINI_RULES, GameState
from breakthrough_zero.symmetry import Symmetry
from scripts.diagnose_model_calibration import calibration_table
from scripts.diagnose_model_symmetry import policy_comparison, sample_states
from scripts.summarize_selfplay import _position_diagnostic


class DiagnosticScriptTests(unittest.TestCase):
    def test_calibration_table_keeps_both_endpoint_values(self) -> None:
        rows = calibration_table(
            np.asarray([-1.0, -0.5, 0.5, 1.0]),
            np.asarray([-1, -1, 1, 1]),
            np.asarray([-0.9, -0.4, 0.4, 0.9]),
        )

        self.assertEqual(sum(row["positions"] for row in rows), 4)
        self.assertEqual(rows[0]["p1_win_fraction"], 0.0)
        self.assertEqual(rows[-1]["p1_win_fraction"], 1.0)

    def test_random_state_sampler_uses_requested_rules(self) -> None:
        states = sample_states(
            12,
            rules=MINI_RULES,
            max_plies=MINI_RULES.maximum_game_plies,
            seed=17,
        )

        self.assertTrue(all(state.rules == MINI_RULES for state in states))
        self.assertTrue(all(state.outcome is None for state in states))

    def test_policy_divergence_handles_exact_zero_probabilities(self) -> None:
        state = GameState.initial(MINI_RULES)
        policy = np.zeros(MINI_RULES.action_size, dtype=np.float32)
        policy[state.policy_index(state.legal_moves()[0])] = 1.0

        l1, js, top_matches = policy_comparison(
            state,
            policy,
            state,
            policy,
            Symmetry.IDENTITY,
        )

        self.assertEqual(l1, 0.0)
        self.assertEqual(js, 0.0)
        self.assertTrue(top_matches)

    def test_exploration_diagnostic_finds_low_prior_winning_move(self) -> None:
        state = GameState(
            p1=1 << 24,
            p2=1 << 12,
            rules=MINI_RULES,
        )
        moves = state.legal_moves()
        actions = tuple(
            ActionStatistics(
                move=move,
                prior=0.8 if index == 0 else 0.2,
                network_prior=0.1 if index == 0 else 0.9,
                visits=10 if index == 0 else 0,
                value_sum=10.0 if index == 0 else 0.0,
                value_square_sum=10.0 if index == 0 else 0.0,
            )
            for index, move in enumerate(moves)
        )
        position = PositionRecord(
            state=state,
            actions=actions,
            selected_move=moves[0],
            root_visits=11,
            root_value_sum=10.0,
            root_value_square_sum=10.0,
            root_evaluation=0.0,
            greedy_backup=1.0,
        )

        report = _position_diagnostic(
            position, np.asarray([1.0, 0.0], dtype=np.float64)
        )

        self.assertGreater(report["search_network_prior_kl"], 0.0)
        self.assertEqual(report["low_prior_actions"], 1)
        self.assertEqual(report["visited_low_prior_actions"], 1)
        self.assertEqual(report["low_prior_visit_mass"], 1.0)
        self.assertEqual(report["has_immediate_win"], 1)
        self.assertEqual(report["immediate_win_has_top_visit"], 1)
        self.assertEqual(report["immediate_win_selected"], 1)


if __name__ == "__main__":
    unittest.main()
