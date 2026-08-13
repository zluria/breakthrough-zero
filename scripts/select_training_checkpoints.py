#!/usr/bin/env python3
"""Select one hash-verified validation checkpoint from each training run."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("training_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--expected-runs", type=int, required=True)
    return parser.parse_args()


def select_checkpoints(training_root: Path, expected_runs: int) -> dict[str, Any]:
    """Choose the lowest finite validation objective among saved checkpoints."""

    run_paths = sorted(training_root.glob("*/run.json"))
    if len(run_paths) != expected_runs:
        raise ValueError(f"expected {expected_runs} runs, found {len(run_paths)}")

    models = {}
    shared_contract = None
    for run_path in run_paths:
        run = json.loads(run_path.read_text(encoding="utf-8"))
        if run.get("status") != "complete":
            raise ValueError(f"training run is not complete: {run_path}")
        contract = {
            key: run.get(key)
            for key in (
                "rules",
                "seed",
                "batch_size",
                "max_training_seconds",
                "validation_fraction",
                "train_games",
                "validation_games",
                "train_positions",
                "validation_positions",
                "inputs",
                "source_mix",
            )
        }
        if shared_contract is None:
            shared_contract = contract
        elif contract != shared_contract:
            raise ValueError(f"training contract differs in: {run_path}")

        candidates = []
        optimizer_examples = 0
        for record in run.get("history", []):
            optimizer_examples += int(record.get("training_samples", 0))
            checkpoint = record.get("checkpoint")
            validation = record.get("validation", {})
            objective = float(validation.get("total", float("nan")))
            if checkpoint is not None and isfinite(objective):
                candidates.append(
                    (
                        objective,
                        int(record["epoch"]),
                        checkpoint,
                        record,
                        optimizer_examples,
                    )
                )
        if not candidates:
            raise ValueError(f"run has no finite saved checkpoint: {run_path}")
        objective, epoch, checkpoint, record, optimizer_examples = min(
            candidates, key=lambda item: item[:2]
        )
        checkpoint_path = Path(checkpoint["path"])
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"checkpoint is missing: {checkpoint_path}")
        actual_hash = _file_sha256(checkpoint_path)
        if actual_hash != checkpoint["sha256"]:
            raise ValueError(f"checkpoint hash mismatch: {checkpoint_path}")

        label = run_path.parent.name
        models[label] = {
            "path": str(checkpoint_path.resolve()),
            "sha256": actual_hash,
            "epoch": epoch,
            "elapsed_seconds": record["elapsed_seconds"],
            "optimizer_examples": optimizer_examples,
            "target": run["target"],
            "network": run["network"],
            "validation": record["validation"],
            "selection": "minimum validation total among saved checkpoints",
            "selection_objective": objective,
        }

    return {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "training_root": str(training_root.resolve()),
        "shared_contract": shared_contract,
        "models": dict(sorted(models.items())),
    }


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"selection output already exists: {args.output}")
    report = select_checkpoints(args.training_root, args.expected_runs)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
