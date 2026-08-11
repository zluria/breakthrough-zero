from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from scripts.run_mini_tournament import (
    _check_unique_names,
    agents,
    matchup_pairs,
    parse_model_specs,
    parse_tactical_puct_specs,
)


class TournamentScriptTests(unittest.TestCase):
    def test_plain_and_ensemble_models_are_distinct_specs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.keras"
            second = Path(directory) / "second.keras"
            first.touch()
            second.touch()

            specs = parse_model_specs(
                [f"plain={first}"], [f"ensemble={second}"]
            )

        self.assertEqual(
            specs,
            [("plain", first, False), ("ensemble", second, True)],
        )

    def test_names_cannot_repeat_across_model_types(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.keras"
            model.touch()
            with self.assertRaisesRegex(ValueError, "duplicated"):
                parse_model_specs([f"same={model}"], [f"same={model}"])

    def test_named_tactical_puct_agents_have_independent_constants(self) -> None:
        specs = parse_tactical_puct_specs(["low=0.75", "high=3.0"])
        self.assertEqual(specs, [("low", 0.75), ("high", 3.0)])
        _check_unique_names([], specs)

    def test_names_cannot_repeat_between_models_and_search_agents(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.keras"
            model.touch()
            model_specs = parse_model_specs([f"same={model}"], [])
            with self.assertRaisesRegex(ValueError, "unique"):
                _check_unique_names(model_specs, [("same", 1.5)])

    def test_strong_screen_runs_only_custom_agent_against_two_anchors(self) -> None:
        specs = agents(
            [],
            [("candidate", 1.5)],
            baseline_set="strong",
        )
        pairs = matchup_pairs(specs, {"candidate"}, "custom-vs-baselines")
        self.assertEqual(
            [(left.name, right.name) for left, right in pairs],
            [
                ("alpha-beta", "candidate"),
                ("puct-tactical", "candidate"),
            ],
        )

    def test_confirmation_adds_custom_head_to_head_without_anchor_repeat(self) -> None:
        specs = agents(
            [],
            [("small", 1.5), ("large", 1.5)],
            baseline_set="strong",
        )
        pairs = matchup_pairs(specs, {"small", "large"}, "custom-relevant")
        self.assertEqual(len(pairs), 5)
        self.assertNotIn(
            ("alpha-beta", "puct-tactical"),
            [(left.name, right.name) for left, right in pairs],
        )


if __name__ == "__main__":
    unittest.main()
