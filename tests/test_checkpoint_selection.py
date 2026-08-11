from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

from scripts.select_training_checkpoints import select_checkpoints


class CheckpointSelectionTests(unittest.TestCase):
    def test_selects_the_best_saved_checkpoint_and_verifies_its_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_root = root / "c32-b3-outcome"
            run_root.mkdir()
            first = run_root / "epoch_004.keras"
            second = run_root / "epoch_008.keras"
            first.write_bytes(b"first")
            second.write_bytes(b"second")
            run = {
                "status": "complete",
                "rules": "mini",
                "seed": 7,
                "batch_size": 32,
                "max_training_seconds": 10,
                "validation_fraction": 0.2,
                "train_games": 8,
                "validation_games": 2,
                "train_positions": 80,
                "validation_positions": 20,
                "inputs": [{"sha256": "data"}],
                "target": "outcome",
                "network": {"board_size": 5},
                "history": [
                    _record(4, 1.2, first),
                    _record(5, 0.1, None),
                    _record(8, 0.8, second),
                ],
            }
            (run_root / "run.json").write_text(json.dumps(run), encoding="utf-8")

            report = select_checkpoints(root, expected_runs=1)

        selected = report["models"]["c32-b3-outcome"]
        self.assertEqual(selected["epoch"], 8)
        self.assertEqual(selected["sha256"], sha256(b"second").hexdigest())


def _record(epoch: int, objective: float, path: Path | None) -> dict:
    checkpoint = None
    if path is not None:
        checkpoint = {
            "path": str(path),
            "sha256": sha256(path.read_bytes()).hexdigest(),
        }
    return {
        "epoch": epoch,
        "elapsed_seconds": float(epoch),
        "training_samples": 80,
        "validation": {"total": objective},
        "checkpoint": checkpoint,
    }


if __name__ == "__main__":
    unittest.main()
