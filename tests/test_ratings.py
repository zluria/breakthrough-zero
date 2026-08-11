from __future__ import annotations

import unittest

from breakthrough_zero.arena import ArenaGame
from breakthrough_zero.ratings import fit_elo_table, summarize_paired_games


def result(pair: int, game: int, p1: str, p2: str, winner: int) -> ArenaGame:
    return ArenaGame(
        pair_id=pair,
        game_in_pair=game,
        opening_index=pair,
        opening_seed=pair,
        p1_agent=p1,
        p2_agent=p2,
        p1_seed=1,
        p2_seed=2,
        records=(),
        winner=winner,
        termination="terminal",
    )


class RatingTests(unittest.TestCase):
    def test_summary_uses_agent_identity_after_color_reversal(self) -> None:
        games = (
            result(0, 0, "a", "b", 1),
            result(0, 1, "b", "a", -1),
            result(1, 0, "a", "b", -1),
            result(1, 1, "b", "a", 1),
        )
        summary = summarize_paired_games(games, "a", "b")
        self.assertEqual((summary.wins, summary.losses), (2, 2))
        self.assertEqual(
            (summary.agent_a_pair_sweeps, summary.agent_b_pair_sweeps),
            (1, 1),
        )
        self.assertEqual(summary.color_split_pairs, 0)
        self.assertEqual(summary.score, 0.5)
        self.assertAlmostEqual(summary.elo_difference, 0.0)
        self.assertLess(summary.elo_95_low, 0)
        self.assertGreater(summary.elo_95_high, 0)

    def test_early_sweep_has_finite_regularized_elo(self) -> None:
        games = (
            result(0, 0, "a", "b", 1),
            result(0, 1, "b", "a", -1),
        )
        summary = summarize_paired_games(games, "a", "b")
        self.assertEqual(summary.score, 1.0)
        self.assertEqual(summary.regularized_score, 0.75)
        self.assertGreater(summary.elo_difference, 0)
        self.assertLess(summary.elo_difference, 1000)

    def test_unbalanced_pair_is_rejected(self) -> None:
        games = (
            result(0, 0, "a", "b", 1),
            result(0, 1, "a", "b", 1),
        )
        with self.assertRaisesRegex(ValueError, "reverse"):
            summarize_paired_games(games, "a", "b")

    def test_global_table_keeps_random_fixed_as_the_anchor(self) -> None:
        versus_random = summarize_paired_games(
            (
                result(0, 0, "a", "random", 1),
                result(0, 1, "random", "a", -1),
            ),
            "a",
            "random",
        )
        ratings = fit_elo_table((versus_random,), anchor="random")
        self.assertEqual(ratings["random"], 1000.0)
        self.assertGreater(ratings["a"], ratings["random"])


if __name__ == "__main__":
    unittest.main()
