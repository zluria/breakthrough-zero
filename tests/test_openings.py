from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from breakthrough_zero.game import MINI_RULES, GameState, Move
from breakthrough_zero.openings import (
    LEGACY_GENERATOR,
    Opening,
    OpeningConfig,
    OpeningSuite,
    generate_opening_suite,
    load_opening_suite,
    save_opening_suite,
)


class OpeningTests(unittest.TestCase):
    def test_suite_rejects_an_immediate_win_when_configured(self) -> None:
        state = GameState.initial(MINI_RULES)
        moves = tuple(
            Move(source, target)
            for source, target in (
                (2, 11),
                (35, 26),
                (11, 19),
                (34, 27),
                (19, 28),
                (26, 19),
            )
        )
        for move in moves:
            state.make_move(move)
        opening = Opening(state, moves, 9)
        self.assertTrue(opening.state.has_immediate_winning_move())
        with self.assertRaisesRegex(ValueError, "immediate win"):
            OpeningSuite(
                MINI_RULES,
                OpeningConfig(count=1, plies=6),
                9,
                (opening,),
            )

    @classmethod
    def setUpClass(cls) -> None:
        cls.config = OpeningConfig(count=2, plies=4)
        cls.suite = generate_opening_suite(cls.config, MINI_RULES, seed=101)

    def test_generation_is_reproducible_and_non_terminal(self) -> None:
        replay = generate_opening_suite(self.config, MINI_RULES, seed=101)
        self.assertEqual(replay, self.suite)
        self.assertTrue(
            all(opening.state.outcome is None for opening in replay.openings)
        )
        self.assertTrue(all(len(opening.moves) == 4 for opening in replay.openings))

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

    def test_legacy_noisy_suite_remains_readable(self) -> None:
        legacy_suite = generate_opening_suite(
            OpeningConfig(count=1, plies=6), MINI_RULES, seed=81
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.json"
            save_opening_suite(path, legacy_suite)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["schema_version"] = 1
            payload["config"] = {
                "count": 1,
                "plies": 6,
                "simulations": 4,
                "c_puct": 1.5,
                "noise_fraction": 0.25,
                "noise_total_concentration": 10.0,
                "temperature": 1.0,
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
            loaded, _ = load_opening_suite(path)

        self.assertEqual(loaded.config.generator, LEGACY_GENERATOR)
        self.assertFalse(loaded.config.reject_immediate_wins)
        self.assertEqual(loaded.openings, legacy_suite.openings)

    def test_random_opening_gives_each_side_the_same_number_of_moves(self) -> None:
        with self.assertRaisesRegex(ValueError, "even"):
            OpeningConfig(count=1, plies=5)
        with self.assertRaisesRegex(ValueError, "even"):
            OpeningConfig(count=1, plies=11)


if __name__ == "__main__":
    unittest.main()
