from __future__ import annotations

from pathlib import Path
import unittest

from scripts.run_continuous_training import _fresh_window, _should_start_cycle


class ContinuousTrainingTests(unittest.TestCase):
    def test_first_cycle_needs_the_preregistered_minimum_time(self) -> None:
        self.assertTrue(
            _should_start_cycle(
                elapsed_seconds=0,
                duration_seconds=3600,
                previous_cycle_seconds=None,
                minimum_cycle_seconds=300,
            )
        )
        self.assertFalse(
            _should_start_cycle(
                elapsed_seconds=3301,
                duration_seconds=3600,
                previous_cycle_seconds=None,
                minimum_cycle_seconds=300,
            )
        )
        self.assertTrue(
            _should_start_cycle(
                elapsed_seconds=0.5,
                duration_seconds=90,
                previous_cycle_seconds=None,
                minimum_cycle_seconds=90,
            )
        )
        self.assertFalse(
            _should_start_cycle(
                elapsed_seconds=2,
                duration_seconds=90,
                previous_cycle_seconds=None,
                minimum_cycle_seconds=90,
            )
        )

    def test_later_cycle_reserves_fifteen_percent_runtime_margin(self) -> None:
        self.assertTrue(
            _should_start_cycle(
                elapsed_seconds=3000,
                duration_seconds=3600,
                previous_cycle_seconds=500,
                minimum_cycle_seconds=300,
            )
        )
        self.assertFalse(
            _should_start_cycle(
                elapsed_seconds=3026,
                duration_seconds=3600,
                previous_cycle_seconds=500,
                minimum_cycle_seconds=300,
            )
        )

    def test_fresh_window_keeps_only_complete_recent_archives(self) -> None:
        paths = [Path(str(index)) for index in range(6)]
        self.assertEqual(_fresh_window(paths, 4), tuple(paths[2:]))
        with self.assertRaises(ValueError):
            _fresh_window(paths, 0)


if __name__ == "__main__":
    unittest.main()
