from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from scripts.run_mini_tournament import parse_model_specs


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


if __name__ == "__main__":
    unittest.main()
