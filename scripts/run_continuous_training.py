#!/usr/bin/env python3
"""Run bounded self-play and replay updates until a wall-clock deadline."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from time import perf_counter
from typing import Any, Sequence

from scripts.audit_selfplay import audit_corpus
from scripts.summarize_selfplay import summarize_corpus


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("historical_input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--initial-model", type=Path, required=True)
    parser.add_argument("--initial-model-sha256", required=True)
    parser.add_argument("--initial-training-state", type=Path)
    parser.add_argument("--duration-seconds", type=float, default=3600)
    parser.add_argument("--minimum-cycle-seconds", type=float, default=300)
    parser.add_argument("--games-per-cycle", type=int, default=1024)
    parser.add_argument("--fresh-window-cycles", type=int, default=4)
    parser.add_argument("--training-seconds-per-cycle", type=float, default=120)
    parser.add_argument("--max-replay-consumption", type=float, default=4)
    parser.add_argument("--historical-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    if not args.historical_input.exists():
        parser.error(f"historical input does not exist: {args.historical_input}")
    if not args.initial_model.is_file():
        parser.error(f"initial model does not exist: {args.initial_model}")
    if args.duration_seconds < args.minimum_cycle_seconds:
        parser.error("duration must permit at least one complete cycle")
    if args.games_per_cycle < 1 or args.fresh_window_cycles < 1:
        parser.error("cycle sizes must be positive")
    if args.training_seconds_per_cycle <= 0 or args.max_replay_consumption <= 0:
        parser.error("training limits must be positive")
    if not 0 < args.historical_fraction < 1:
        parser.error("historical fraction must be between zero and one")
    if not 0 <= args.seed < 2**64:
        parser.error("seed must fit in an unsigned 64-bit integer")
    return args


def main() -> None:
    args = parse_args()
    actual_hash = _file_sha256(args.initial_model)
    if actual_hash != args.initial_model_sha256:
        raise ValueError("initial model hash does not match the pinned hash")

    args.output.mkdir(parents=True)
    started = perf_counter()
    seed_source = random.Random(args.seed)
    actor = {
        "path": str(args.initial_model.resolve()),
        "sha256": actual_hash,
        "training_state": (
            str(args.initial_training_state.resolve())
            if args.initial_training_state is not None
            else None
        ),
    }
    progress: dict[str, Any] = {
        "status": "running",
        "phase": "preflight",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": args.duration_seconds,
        "settings": {
            "games_per_cycle": args.games_per_cycle,
            "fresh_window_cycles": args.fresh_window_cycles,
            "historical_fraction": args.historical_fraction,
            "max_replay_consumption": args.max_replay_consumption,
            "training_seconds_per_cycle": args.training_seconds_per_cycle,
            "root_noise_fraction": 0.0,
            "root_noise_total_concentration": 10.0,
            "sample_until_ply": 4,
            "simulations": 32,
            "c_puct": 1.5,
        },
        "initial_actor": actor,
        "latest_actor": actor,
        "totals": {
            "fresh_games": 0,
            "fresh_positions": 0,
            "training_examples_consumed": 0,
        },
        "cycles": [],
    }
    _write_json(args.output / "progress.json", progress)

    fresh_archives: list[Path] = []
    previous_cycle_seconds: float | None = None
    try:
        while _should_start_cycle(
            elapsed_seconds=perf_counter() - started,
            duration_seconds=args.duration_seconds,
            previous_cycle_seconds=previous_cycle_seconds,
            minimum_cycle_seconds=args.minimum_cycle_seconds,
        ):
            cycle_started = perf_counter()
            cycle_index = len(progress["cycles"]) + 1
            cycle = args.output / f"cycle_{cycle_index:03d}"
            selfplay = cycle / "selfplay"
            training = cycle / "training"
            cycle.mkdir()
            selfplay_seed = seed_source.getrandbits(64)
            training_seed = seed_source.getrandbits(64)

            progress["phase"] = f"cycle_{cycle_index:03d}_selfplay"
            _write_json(args.output / "progress.json", progress)
            print(f"cycle {cycle_index}: self-play from {actor['sha256']}", flush=True)
            _run_checked(
                [
                    sys.executable,
                    "scripts/generate_neural_selfplay.py",
                    str(selfplay),
                    "--model",
                    actor["path"],
                    "--rules",
                    "mini",
                    "--games",
                    str(args.games_per_cycle),
                    "--chunk-games",
                    "64",
                    "--batch-size",
                    "64",
                    "--simulations",
                    "32",
                    "--c-puct",
                    "1.5",
                    "--sample-until-ply",
                    "4",
                    "--temperature",
                    "1.0",
                    "--noise-fraction",
                    "0",
                    "--noise-total-concentration",
                    "10",
                    "--seed",
                    str(selfplay_seed),
                ],
                cycle / "selfplay.log",
            )
            audit = audit_corpus(
                selfplay,
                expected_games=args.games_per_cycle,
                expected_rules="breakthrough-5x5-one-row-v1",
                expected_simulations=32,
                expected_c_puct=1.5,
                expected_model_sha256=actor["sha256"],
                expected_noise_fraction=0,
                expected_noise_total_concentration=10,
                expected_sample_until_ply=4,
                expected_batch_size=64,
            )
            summary = summarize_corpus(selfplay)
            _write_json(selfplay / "audit.json", audit)
            _write_json(selfplay / "summary.json", summary)

            fresh_archives.append(selfplay)
            fresh_window = _fresh_window(
                fresh_archives, args.fresh_window_cycles
            )
            progress["phase"] = f"cycle_{cycle_index:03d}_training"
            _write_json(args.output / "progress.json", progress)
            print(
                f"cycle {cycle_index}: train on {len(fresh_window)} fresh archives",
                flush=True,
            )
            command = [
                sys.executable,
                "scripts/train_replay.py",
                str(args.historical_input),
                str(training),
            ]
            for archive in fresh_window:
                command.extend(("--fresh-input", str(archive)))
            command.extend(
                (
                    "--historical-fraction",
                    str(args.historical_fraction),
                    "--initial-model",
                    actor["path"],
                )
            )
            if actor["training_state"] is not None:
                command.extend(
                    ("--initial-training-state", actor["training_state"])
                )
            command.extend(
                (
                    "--target",
                    "mixed_z_q",
                    "--channels",
                    "32",
                    "--residual-blocks",
                    "3",
                    "--value-hidden",
                    "64",
                    "--batch-size",
                    "256",
                    "--max-steps",
                    "100000",
                    "--max-training-seconds",
                    str(args.training_seconds_per_cycle),
                    "--max-replay-consumption",
                    str(args.max_replay_consumption),
                    "--evaluate-every",
                    "20",
                    "--validation-batch-size",
                    "512",
                    "--learning-rate",
                    "0.0003",
                    "--l2",
                    "0.0001",
                    "--validation-fraction",
                    "0.2",
                    "--split-seed",
                    str(args.seed),
                    "--seed",
                    str(training_seed),
                )
            )
            _run_checked(command, cycle / "training.log")

            training_run = _read_json(training / "run.json")
            actor_manifest = _read_json(training / "actor.json")
            next_actor = actor_manifest["models"]["latest"]
            if next_actor["sha256"] == actor["sha256"]:
                raise RuntimeError("an optimizer update did not change the actor hash")
            if _file_sha256(Path(next_actor["path"])) != next_actor["sha256"]:
                raise RuntimeError("published actor hash does not match its model")

            previous_cycle_seconds = perf_counter() - cycle_started
            record = _cycle_record(
                index=cycle_index,
                elapsed_seconds=perf_counter() - started,
                cycle_seconds=previous_cycle_seconds,
                selfplay_seed=selfplay_seed,
                training_seed=training_seed,
                fresh_window=fresh_window,
                audit=audit,
                summary=summary,
                training_run=training_run,
                actor=next_actor,
            )
            progress["cycles"].append(record)
            actor = next_actor
            progress["latest_actor"] = actor
            progress["totals"] = {
                "fresh_games": sum(
                    item["selfplay"]["fresh_games"]
                    for item in progress["cycles"]
                ),
                "fresh_positions": sum(
                    item["selfplay"]["fresh_positions"]
                    for item in progress["cycles"]
                ),
                "training_examples_consumed": sum(
                    item["training"]["examples_consumed"]
                    for item in progress["cycles"]
                ),
            }
            progress["phase"] = "between_cycles"
            _write_json(args.output / "progress.json", progress)
            print(
                f"cycle {cycle_index}: actor {actor['sha256']}, "
                f"R={record['training']['replay_consumption']:.3f}",
                flush=True,
            )

        if not progress["cycles"]:
            raise RuntimeError("wall-clock budget completed no training cycle")
        progress["status"] = "complete"
        progress["phase"] = "complete"
        progress["elapsed_seconds"] = round(perf_counter() - started, 3)
        _write_json(args.output / "progress.json", progress)
    except BaseException as error:
        progress["status"] = "failed"
        progress["error"] = f"{type(error).__name__}: {error}"
        progress["elapsed_seconds"] = round(perf_counter() - started, 3)
        _write_json(args.output / "progress.json", progress)
        raise


def _cycle_record(
    *,
    index: int,
    elapsed_seconds: float,
    cycle_seconds: float,
    selfplay_seed: int,
    training_seed: int,
    fresh_window: Sequence[Path],
    audit: dict[str, Any],
    summary: dict[str, Any],
    training_run: dict[str, Any],
    actor: dict[str, Any],
) -> dict[str, Any]:
    replay = training_run["replay"]
    history = training_run["history"]
    return {
        "cycle": index,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "cycle_seconds": round(cycle_seconds, 3),
        "seeds": {"selfplay": selfplay_seed, "training": training_seed},
        "selfplay": {
            "archives": [str(path.resolve()) for path in fresh_window],
            "fresh_games": audit["games"],
            "fresh_positions": audit["positions"],
            "positions_per_second": summary[
                "recorded_generation_positions_per_second"
            ],
            "search_network_prior_kl": summary[
                "mean_search_network_prior_kl"
            ],
            "visited_low_prior_fraction": summary[
                "visited_low_network_prior_fraction"
            ],
            "immediate_win_selected_fraction": summary[
                "immediate_win_selected_fraction"
            ],
        },
        "training": {
            "examples_consumed": replay["examples_presented"],
            "replay_consumption": training_run["replay_consumption"],
            "historical_presentations": replay["historical_presentations"],
            "fresh_presentations": replay["fresh_presentations"],
            "completed_steps": training_run["completed_steps"],
            "optimizer": training_run["optimizer"],
            "last_train": history[-1]["train"],
            "last_validation": history[-1]["validation"],
            "best_validation": training_run["diagnostics"]["best_validation"],
        },
        "actor": actor,
    }


def _should_start_cycle(
    *,
    elapsed_seconds: float,
    duration_seconds: float,
    previous_cycle_seconds: float | None,
    minimum_cycle_seconds: float,
) -> bool:
    if previous_cycle_seconds is None:
        latest_safe_start = max(1.0, duration_seconds - minimum_cycle_seconds)
        return elapsed_seconds <= latest_safe_start
    estimate = minimum_cycle_seconds
    estimate = max(estimate, 1.15 * previous_cycle_seconds)
    return elapsed_seconds + estimate <= duration_seconds


def _fresh_window(paths: Sequence[Path], capacity: int) -> tuple[Path, ...]:
    if capacity < 1:
        raise ValueError("fresh replay capacity must be positive")
    return tuple(paths[-capacity:])


def _run_checked(command: Sequence[str], log_path: Path) -> None:
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            command,
            cwd=Path(__file__).resolve().parents[1],
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if result.returncode:
        raise RuntimeError(
            f"command failed with exit {result.returncode}; see {log_path}"
        )


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _write_json(path: Path, payload: Any) -> None:
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


if __name__ == "__main__":
    main()
