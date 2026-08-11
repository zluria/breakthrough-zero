from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from breakthrough_zero.game import MINI_RULES, Move
from breakthrough_zero.openings import (
    Opening,
    OpeningConfig,
    generate_opening_suite,
    load_opening_suite,
    save_opening_suite,
)


class OpeningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = OpeningConfig(count=2, plies=5, simulations=4)
        cls.suite = generate_opening_suite(cls.config, MINI_RULES, seed=101)

    def test_generation_is_reproducible_and_non_terminal(self) -> None:
        replay = generate_opening_suite(self.config, MINI_RULES, seed=101)
        self.assertEqual(replay, self.suite)
        self.assertTrue(
            all(opening.state.outcome is None for opening in replay.openings)
        )
        self.assertTrue(all(len(opening.moves) == 5 for opening in replay.openings))

    def test_opening_rejects_a_prefix_that_does_not_match_its_state(self) -> None:
        opening = self.suite.openings[0]
        wrong = (Move(0, 8),) + opening.moves[1:]
        with self.assertRaisesRegex(ValueError, "illegal|does not match"):
            Opening(opening.state, wrong, opening.seed)

    def test_atomic_json_round_trip_is_immutable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "openings.json"
            save_opening_suite(path, self.suite, metadata={"purpose": "test"})
            loaded, metadata = load_opening_suite(path)
            with self.assertRaises(FileExistsError):
                save_opening_suite(path, self.suite)

        self.assertEqual(loaded, self.suite)
        self.assertEqual(metadata, {"purpose": "test"})

    def test_opening_length_is_restricted_to_the_eval_noise_window(self) -> None:
        with self.assertRaisesRegex(ValueError, "5 to 10"):
            OpeningConfig(count=1, plies=4)
        with self.assertRaisesRegex(ValueError, "5 to 10"):
            OpeningConfig(count=1, plies=11)


if __name__ == "__main__":
    unittest.main()
