from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from breakthrough_zero.data import save_chunk
from breakthrough_zero.game import MINI_RULES, GameState
from breakthrough_zero.search import SearchConfig
from breakthrough_zero.selfplay import SelfPlayConfig, play_dummy_game
from scripts.audit_selfplay import audit_corpus


def _game(seed: int):
    return play_dummy_game(
        SelfPlayConfig(
            search=SearchConfig(simulations=4, c_puct=0.75),
            sample_until_ply=4,
        ),
        seed=seed,
        prefer_tactical_rollouts=True,
        initial_state=GameState.initial(MINI_RULES),
    )


def _metadata(
    master_seed: int, *, c_puct: float = 0.75, noise_fraction: float = 0.0
):
    return {
        "run_config": {
            "schema_version": 3,
            "master_seed": master_seed,
            "rules": MINI_RULES.name,
            "chunk_games": 1,
            "batch_size": 8,
            "simulations": 4,
            "c_puct": c_puct,
            "sample_until_ply": 4,
            "temperature": 1.0,
            "max_plies": MINI_RULES.maximum_game_plies,
            "tactical_rollouts": True,
            "noise_fraction": noise_fraction,
            "noise_total_concentration": 10.0,
            "model_sha256": "model-hash",
        },
        "environment": {"git_commit": "test-commit"},
    }


class SelfPlayAuditTests(unittest.TestCase):
    def test_valid_shards_may_differ_only_by_master_seed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index in range(2):
                path = root / f"shard-{index}" / "chunk_00000.npz"
                path.parent.mkdir()
                save_chunk(path, (_game(index + 1),), metadata=_metadata(index))

            report = audit_corpus(
                root,
                expected_games=2,
                expected_rules=MINI_RULES.name,
                expected_simulations=4,
                expected_c_puct=0.75,
            )

        self.assertEqual(report["status"], "valid")
        self.assertEqual(report["games"], 2)
        self.assertEqual(report["chunks"], 2)

    def test_configuration_drift_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, c_puct in enumerate((0.75, 1.5)):
                path = root / f"shard-{index}" / "chunk_00000.npz"
                path.parent.mkdir()
                save_chunk(path, (_game(7),), metadata=_metadata(index, c_puct=c_puct))

            with self.assertRaisesRegex(ValueError, "configuration differs"):
                audit_corpus(
                    root,
                    expected_games=2,
                    expected_rules=MINI_RULES.name,
                    expected_simulations=4,
                    expected_c_puct=0.75,
                )

    def test_duplicate_seed_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index in range(2):
                path = root / f"shard-{index}" / "chunk_00000.npz"
                path.parent.mkdir()
                save_chunk(path, (_game(7),), metadata=_metadata(index))

            with self.assertRaisesRegex(ValueError, "seeds are not unique"):
                audit_corpus(
                    root,
                    expected_games=2,
                    expected_rules=MINI_RULES.name,
                    expected_simulations=4,
                    expected_c_puct=0.75,
                )

    def test_explicit_experiment_expectation_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "chunk_00000.npz"
            save_chunk(path, (_game(1),), metadata=_metadata(1))

            with self.assertRaisesRegex(ValueError, "noise_fraction"):
                audit_corpus(
                    root,
                    expected_games=1,
                    expected_rules=MINI_RULES.name,
                    expected_simulations=4,
                    expected_c_puct=0.75,
                    expected_model_sha256="model-hash",
                    expected_noise_fraction=0.1,
                    expected_noise_total_concentration=10.0,
                    expected_sample_until_ply=4,
                    expected_batch_size=8,
                )


if __name__ == "__main__":
    unittest.main()
