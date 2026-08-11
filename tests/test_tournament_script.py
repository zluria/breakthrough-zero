from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from scripts.run_mini_tournament import (
    _check_unique_names,
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


if __name__ == "__main__":
    unittest.main()
