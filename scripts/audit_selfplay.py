#!/usr/bin/env python3
"""Fail closed unless a recursive self-play corpus is internally consistent."""

from __future__ import annotations

import argparse
import json
from math import isclose
from pathlib import Path
from typing import Any

import numpy as np

from breakthrough_zero.data import load_chunk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--expected-games", type=int, required=True)
    parser.add_argument("--expected-rules", required=True)
    parser.add_argument("--expected-simulations", type=int, required=True)
    parser.add_argument("--expected-c-puct", type=float, required=True)
    parser.add_argument("--expected-model-sha256")
    parser.add_argument("--expected-noise-fraction", type=float)
    parser.add_argument("--expected-noise-total-concentration", type=float)
    parser.add_argument("--expected-sample-until-ply", type=int)
    parser.add_argument("--expected-batch-size", type=int)
    return parser.parse_args()


def audit_corpus(
    root: Path,
    *,
    expected_games: int,
    expected_rules: str,
    expected_simulations: int,
    expected_c_puct: float,
    expected_model_sha256: str | None = None,
    expected_noise_fraction: float | None = None,
    expected_noise_total_concentration: float | None = None,
    expected_sample_until_ply: int | None = None,
    expected_batch_size: int | None = None,
) -> dict[str, Any]:
    """Reload every chunk and verify the assumptions of one training corpus."""

    paths = sorted(root.rglob("chunk_*.npz"))
    if not paths:
        raise ValueError(f"no self-play chunks found under {root}")

    games = []
    reference_config: dict[str, Any] | None = None
    commits: set[str | None] = set()
    for path in paths:
        chunk_games, manifest = load_chunk(path)
        run_config = manifest.get("metadata", {}).get("run_config")
        if not isinstance(run_config, dict):
            raise ValueError(f"chunk has no run_config: {path}")
        comparable = {
            key: value for key, value in run_config.items() if key != "master_seed"
        }
        if reference_config is None:
            reference_config = comparable
        elif comparable != reference_config:
            raise ValueError(f"search configuration differs in chunk: {path}")
        commits.add(
            manifest.get("metadata", {}).get("environment", {}).get("git_commit")
        )
        games.extend(chunk_games)

    assert reference_config is not None
    if len(games) != expected_games:
        raise ValueError(f"expected {expected_games} games, found {len(games)}")
    seeds = [game.seed for game in games]
    if len(seeds) != len(set(seeds)):
        raise ValueError("game seeds are not unique across the corpus")
    if reference_config.get("rules") != expected_rules:
        raise ValueError("the corpus uses an unexpected ruleset")
    if reference_config.get("simulations") != expected_simulations:
        raise ValueError("the corpus uses an unexpected simulation budget")
    if not isclose(
        float(reference_config.get("c_puct", float("nan"))),
        expected_c_puct,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("the corpus uses an unexpected c_puct")
    exact_expectations = {
        "model_sha256": expected_model_sha256,
        "sample_until_ply": expected_sample_until_ply,
        "batch_size": expected_batch_size,
    }
    for key, expected in exact_expectations.items():
        if expected is not None and reference_config.get(key) != expected:
            raise ValueError(f"the corpus uses an unexpected {key}")
    float_expectations = {
        "noise_fraction": expected_noise_fraction,
        "noise_total_concentration": expected_noise_total_concentration,
    }
    for key, expected in float_expectations.items():
        if expected is not None and not isclose(
            float(reference_config.get(key, float("nan"))),
            expected,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"the corpus uses an unexpected {key}")
    if len(commits) != 1 or None in commits:
        raise ValueError("all chunks must record the same Git commit")

    positions = [position for game in games for position in game.positions]
    if any(position.root_visits != expected_simulations for position in positions):
        raise ValueError("a position has the wrong root visit count")
    if any(
        not position.full_search or position.sample_weight != 1.0
        for position in positions
    ):
        raise ValueError("baseline data must contain full, unit-weight searches")
    ply_bound = positions[0].state.rules.maximum_game_plies
    lengths = np.asarray([len(game.positions) for game in games], dtype=np.int64)
    if int(lengths.max()) > ply_bound:
        raise ValueError("a game exceeds the rules-derived ply bound")

    states = {
        (p.state.p1, p.state.p2, p.state.to_move) for p in positions
    }
    trajectories = {
        tuple((p.selected_move.source, p.selected_move.target) for p in game.positions)
        for game in games
    }
    return {
        "status": "valid",
        "root": str(root.resolve()),
        "chunks": len(paths),
        "games": len(games),
        "positions": len(positions),
        "unique_positions": len(states),
        "unique_trajectories": len(trajectories),
        "p1_win_fraction": round(float(np.mean([g.outcome == 1 for g in games])), 4),
        "game_plies": {
            "mean": round(float(lengths.mean()), 3),
            "p50": float(np.percentile(lengths, 50)),
            "p90": float(np.percentile(lengths, 90)),
            "p99": float(np.percentile(lengths, 99)),
            "maximum": int(lengths.max()),
            "proof_bound": ply_bound,
        },
        "git_commit": next(iter(commits)),
        "run_config": reference_config,
    }


def main() -> None:
    args = parse_args()
    report = audit_corpus(
        args.root,
        expected_games=args.expected_games,
        expected_rules=args.expected_rules,
        expected_simulations=args.expected_simulations,
        expected_c_puct=args.expected_c_puct,
        expected_model_sha256=args.expected_model_sha256,
        expected_noise_fraction=args.expected_noise_fraction,
        expected_noise_total_concentration=args.expected_noise_total_concentration,
        expected_sample_until_ply=args.expected_sample_until_ply,
        expected_batch_size=args.expected_batch_size,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
