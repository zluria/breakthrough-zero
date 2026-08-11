"""Generate checksummed MCTS pretraining games with the dummy evaluator.

The default is intentionally tiny and uses the 5x5 debug game.  Re-running the
same command verifies and skips complete chunks, so an interrupted job can be
resumed without overwriting evidence.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
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
from breakthrough_zero.game import MINI_RULES, STANDARD_RULES, GameState, Ruleset
from breakthrough_zero.search import RootNoiseConfig, SearchConfig
from breakthrough_zero.selfplay import SelfPlayConfig, play_dummy_game


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="directory for immutable chunks")
    parser.add_argument("--games", type=int, default=2)
    parser.add_argument("--chunk-games", type=int, default=2)
    parser.add_argument("--simulations", type=int, default=8)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--sample-until-ply", type=int, default=6)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--max-plies",
        type=int,
        default=None,
        help="override the rules-derived safety bound (normally leave unset)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rules", choices=("mini", "standard"), default="mini")
    parser.add_argument("--tactical-rollouts", action="store_true")
    parser.add_argument("--noise-fraction", type=float, default=0.0)
    parser.add_argument("--noise-total-concentration", type=float, default=10.0)
    args = parser.parse_args()
    if args.games < 1 or args.chunk_games < 1:
        parser.error("--games and --chunk-games must be positive")
    if not 0 <= args.seed < 2**64:
        parser.error("--seed must fit in an unsigned 64-bit integer")
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
        search=SearchConfig(simulations=args.simulations, c_puct=args.c_puct),
        sample_until_ply=args.sample_until_ply,
        temperature=args.temperature,
        max_plies=args.max_plies,
        root_noise=noise,
    )
    run_config = _run_config(args, rules, config)
    seeds = _game_seeds(args.seed, args.games)
    args.output.mkdir(parents=True, exist_ok=True)

    start_time = perf_counter()
    generated_games = 0
    generated_positions = 0
    for start in range(0, args.games, args.chunk_games):
        stop = min(start + args.chunk_games, args.games)
        chunk_index = start // args.chunk_games
        path = args.output / f"chunk_{chunk_index:05d}.npz"
        expected_seeds = seeds[start:stop]

        if path.exists() or path.with_suffix(".json").exists():
            games, manifest = load_chunk(path)
            _check_existing_chunk(
                path, games, manifest, run_config, start, stop, expected_seeds
            )
            print(f"verified {path.name}: games {start}..{stop - 1}", flush=True)
            continue

        chunk_start = perf_counter()
        games = tuple(
            play_dummy_game(
                config,
                seed=game_seed,
                prefer_tactical_rollouts=args.tactical_rollouts,
                initial_state=GameState.initial(rules),
            )
            for game_seed in expected_seeds
        )
        metadata = {
            "generator": "dummy-mcts",
            "run_config": run_config,
            "start_game": start,
            "stop_game": stop,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "environment": _environment(),
        }
        save_chunk(path, games, metadata=metadata)
        loaded, manifest = load_chunk(path)
        _check_existing_chunk(
            path, loaded, manifest, run_config, start, stop, expected_seeds
        )

        positions = sum(len(game.positions) for game in games)
        generated_games += len(games)
        generated_positions += positions
        seconds = perf_counter() - chunk_start
        rate = positions / seconds if seconds else float("inf")
        print(
            f"wrote {path.name}: {len(games)} games, {positions} positions, "
            f"{seconds:.2f}s, {rate:.2f} positions/s",
            flush=True,
        )

    elapsed = perf_counter() - start_time
    print(
        json.dumps(
            {
                "status": "complete",
                "generated_games": generated_games,
                "generated_positions": generated_positions,
                "elapsed_seconds": round(elapsed, 3),
                "output": str(args.output.resolve()),
            },
            sort_keys=True,
        )
    )


def _game_seeds(master_seed: int, count: int) -> list[int]:
    source = random.Random(master_seed)
    return [source.getrandbits(64) for _ in range(count)]


def _run_config(
    args: argparse.Namespace, rules: Ruleset, config: SelfPlayConfig
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "master_seed": args.seed,
        "rules": rules.name,
        "chunk_games": args.chunk_games,
        "simulations": config.search.simulations,
        "c_puct": config.search.c_puct,
        "sample_until_ply": config.sample_until_ply,
        "temperature": config.temperature,
        "max_plies": config.ply_limit(rules),
        "tactical_rollouts": args.tactical_rollouts,
        "noise_fraction": args.noise_fraction,
        "noise_total_concentration": args.noise_total_concentration,
    }


def _check_existing_chunk(
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


def _environment() -> dict[str, str | None]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "git_commit": _git_commit(),
    }


def _git_commit() -> str | None:
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
