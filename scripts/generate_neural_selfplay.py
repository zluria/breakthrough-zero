#!/usr/bin/env python3
"""Generate immutable neural-PUCT games with independent-leaf batching."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import platform
import random
import subprocess
import sys
from time import perf_counter
from typing import Any

import numpy as np

from breakthrough_zero.data import SCHEMA_VERSION, load_chunk, save_chunk
from breakthrough_zero.evaluators import (
    HeadAblationEvaluator,
    SymmetryEnsembleEvaluator,
)
from breakthrough_zero.game import MINI_RULES, STANDARD_RULES, Ruleset
from breakthrough_zero.network import KerasEvaluator, load_network
from breakthrough_zero.search import BatchEvaluator, RootNoiseConfig, SearchConfig
from breakthrough_zero.selfplay import SelfPlayConfig, play_batched_games


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--games", type=int, default=64)
    parser.add_argument("--chunk-games", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--simulations", type=int, default=16)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--sample-until-ply", type=int, default=12)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--max-plies",
        type=int,
        default=None,
        help="override the rules-derived safety bound (normally leave unset)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rules", choices=("mini", "standard"), default="standard")
    parser.add_argument("--noise-fraction", type=float, default=0.0)
    parser.add_argument("--noise-total-concentration", type=float, default=10.0)
    parser.add_argument("--symmetry-ensemble", action="store_true")
    parser.add_argument("--uniform-policy", action="store_true")
    parser.add_argument("--zero-value", action="store_true")
    args = parser.parse_args()
    if args.games < 1 or args.chunk_games < 1 or args.batch_size < 1:
        parser.error("game, chunk, and batch counts must be positive")
    if not 0 <= args.seed < 2**64:
        parser.error("--seed must fit in an unsigned 64-bit integer")
    if not args.model.is_file():
        parser.error(f"model does not exist: {args.model}")
    return args


def main() -> None:
    args = parse_args()
    rules = MINI_RULES if args.rules == "mini" else STANDARD_RULES
    noise = (
        RootNoiseConfig(
            fraction=args.noise_fraction,
            total_concentration=args.noise_total_concentration,
        )
        if args.noise_fraction
        else None
    )
    config = SelfPlayConfig(
        search=SearchConfig(
            simulations=args.simulations,
            c_puct=args.c_puct,
        ),
        sample_until_ply=args.sample_until_ply,
        temperature=args.temperature,
        max_plies=args.max_plies,
        root_noise=noise,
    )
    model_digest = file_sha256(args.model)
    run_config = make_run_config(args, rules, config, model_digest)
    seeds = game_seeds(args.seed, args.games)
    args.output.mkdir(parents=True, exist_ok=True)

    evaluator: BatchEvaluator | None = None
    generated_games = 0
    generated_positions = 0
    started = perf_counter()
    for start in range(0, args.games, args.chunk_games):
        stop = min(start + args.chunk_games, args.games)
        chunk_index = start // args.chunk_games
        path = args.output / f"chunk_{chunk_index:05d}.npz"
        expected_seeds = seeds[start:stop]

        if path.exists() or path.with_suffix(".json").exists():
            games, manifest = load_chunk(path)
            check_existing_chunk(
                path, games, manifest, run_config, start, stop, expected_seeds
            )
            print(f"verified {path.name}: games {start}..{stop - 1}", flush=True)
            continue

        if evaluator is None:
            base = KerasEvaluator(load_network(args.model))
            evaluator = (
                SymmetryEnsembleEvaluator(base)
                if args.symmetry_ensemble
                else base
            )
            if args.uniform_policy or args.zero_value:
                evaluator = HeadAblationEvaluator(
                    evaluator,
                    uniform_policy=args.uniform_policy,
                    zero_value=args.zero_value,
                )
        chunk_started = perf_counter()
        games = play_batched_games(
            evaluator,
            config,
            expected_seeds,
            rules=rules,
            batch_size=args.batch_size,
        )
        metadata = {
            "generator": "batched-neural-puct",
            "run_config": run_config,
            "model_path": str(args.model.resolve()),
            "start_game": start,
            "stop_game": stop,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "environment": environment(),
        }
        save_chunk(path, games, metadata=metadata)
        loaded, manifest = load_chunk(path)
        check_existing_chunk(
            path, loaded, manifest, run_config, start, stop, expected_seeds
        )

        positions = sum(len(game.positions) for game in games)
        generated_games += len(games)
        generated_positions += positions
        seconds = perf_counter() - chunk_started
        print(
            f"wrote {path.name}: {len(games)} games, {positions} positions, "
            f"{seconds:.2f}s, {positions / seconds:.2f} positions/s",
            flush=True,
        )

    elapsed = perf_counter() - started
    print(
        json.dumps(
            {
                "status": "complete",
                "generated_games": generated_games,
                "generated_positions": generated_positions,
                "elapsed_seconds": round(elapsed, 3),
                "output": str(args.output.resolve()),
                "model_sha256": model_digest,
            },
            sort_keys=True,
        )
    )


def game_seeds(master_seed: int, count: int) -> list[int]:
    source = random.Random(master_seed)
    return [source.getrandbits(64) for _ in range(count)]


def make_run_config(
    args: argparse.Namespace,
    rules: Ruleset,
    config: SelfPlayConfig,
    model_digest: str,
) -> dict[str, Any]:
    # The requested total is intentionally absent: a complete deterministic
    # prefix may be extended later without rewriting any published chunk.
    return {
        "schema_version": SCHEMA_VERSION,
        "master_seed": args.seed,
        "rules": rules.name,
        "chunk_games": args.chunk_games,
        "batch_size": args.batch_size,
        "simulations": config.search.simulations,
        "c_puct": config.search.c_puct,
        "sample_until_ply": config.sample_until_ply,
        "temperature": config.temperature,
        "max_plies": config.ply_limit(rules),
        "noise_fraction": args.noise_fraction,
        "noise_total_concentration": args.noise_total_concentration,
        "model_sha256": model_digest,
        "symmetry_ensemble": args.symmetry_ensemble,
        "uniform_policy": args.uniform_policy,
        "zero_value": args.zero_value,
    }


def check_existing_chunk(
    path: Path,
    games,
    manifest: dict[str, Any],
    run_config: dict[str, Any],
    start: int,
    stop: int,
    expected_seeds: list[int],
) -> None:
    metadata = manifest.get("metadata", {})
    if metadata.get("run_config") != run_config:
        raise RuntimeError(f"{path} was created with a different configuration")
    if (metadata.get("start_game"), metadata.get("stop_game")) != (start, stop):
        raise RuntimeError(f"{path} covers different game indices")
    if [game.seed for game in games] != expected_seeds:
        raise RuntimeError(f"{path} contains unexpected game seeds")


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def environment() -> dict[str, str | None]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "git_commit": git_commit(),
    }


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


if __name__ == "__main__":
    main()
