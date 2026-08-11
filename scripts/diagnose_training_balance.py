#!/usr/bin/env python3
"""Reproduce a training sample and report target/augmentation balance."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from breakthrough_zero.data import Target, load_chunk, split_game_indices, value_target
from breakthrough_zero.symmetry import Symmetry
from breakthrough_zero.training import PositionSample, samples_from_games


TARGETS: tuple[Target, ...] = (
    "outcome",
    "soft_z",
    "a0c",
    "played_q",
    "greedy_backup",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--target", choices=TARGETS, default="soft_z")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--max-train-positions", type=int, default=2500)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=202608120801)
    args = parser.parse_args()
    if args.max_train_positions < 1 or args.epochs < 1 or args.batch_size < 1:
        parser.error("position, epoch, and batch counts must be positive")
    return args


def load_games(root: Path):
    paths = [root] if root.is_file() else sorted(root.rglob("chunk_*.npz"))
    if not paths:
        raise ValueError(f"no self-play chunks found under {root}")
    games = []
    inputs = []
    for path in paths:
        chunk_games, manifest = load_chunk(path)
        games.extend(chunk_games)
        inputs.append({"path": str(path.resolve()), "sha256": manifest["sha256"]})
    return games, inputs


def exact_training_samples(games, args: argparse.Namespace):
    train_indices, validation_indices = split_game_indices(
        len(games), args.validation_fraction, seed=args.seed
    )
    available = samples_from_games([games[index] for index in train_indices])
    if len(available) < args.max_train_positions:
        raise ValueError("the reproduced training split has too few positions")
    selected = np.random.default_rng(args.seed + 1).choice(
        len(available),
        size=args.max_train_positions,
        replace=False,
    )
    samples = [available[int(index)] for index in selected]
    return samples, train_indices, validation_indices, len(available)


def policy_plane_mass(sample: PositionSample) -> np.ndarray:
    actions = sample.position.actions
    visits = np.asarray([action.visits for action in actions], dtype=np.float64)
    if visits.sum() > 0:
        weights = visits / visits.sum()
    else:
        weights = np.asarray(
            [action.network_prior for action in actions], dtype=np.float64
        )
        weights /= weights.sum()
    mass = np.zeros(3, dtype=np.float64)
    for action, weight in zip(actions, weights, strict=True):
        mass[sample.position.state.policy_index(action.move) % 3] += weight
    return mass


def augmentation_schedule(sample_count: int, args: argparse.Namespace):
    rng = np.random.default_rng(args.seed)
    seen = np.zeros((sample_count, 4), dtype=np.bool_)
    counts = np.zeros(4, dtype=np.int64)
    order = np.arange(sample_count)
    for _ in range(args.epochs):
        rng.shuffle(order)
        for start in range(0, sample_count, args.batch_size):
            for sample_index in order[start : start + args.batch_size]:
                symmetry = int(rng.integers(4))
                seen[int(sample_index), symmetry] = True
                counts[symmetry] += 1
    coverage = Counter(int(row.sum()) for row in seen)
    return counts, coverage


def main() -> None:
    args = parse_args()
    games, inputs = load_games(args.input)
    samples, train_indices, validation_indices, available = exact_training_samples(
        games, args
    )
    targets = np.asarray(
        [
            value_target(sample.position, sample.outcome, args.target)
            for sample in samples
        ],
        dtype=np.float64,
    )
    players = np.asarray(
        [sample.position.state.to_move for sample in samples], dtype=np.int8
    )
    plane_mass = np.stack([policy_plane_mass(sample) for sample in samples])
    symmetry_counts, coverage = augmentation_schedule(len(samples), args)
    by_player = {}
    for player in (1, -1):
        selected = players == player
        by_player[str(player)] = {
            "positions": int(selected.sum()),
            "target_mean": float(targets[selected].mean()),
            "target_std": float(targets[selected].std()),
            "policy_left_forward_right_mean": plane_mass[selected].mean(axis=0).tolist(),
        }

    train_outcomes = [games[index].outcome for index in train_indices]
    report = {
        "inputs": inputs,
        "games": len(games),
        "train_games": len(train_indices),
        "validation_games": len(validation_indices),
        "train_game_p1_win_fraction": float(np.mean(np.asarray(train_outcomes) == 1)),
        "available_train_positions": available,
        "selected_train_positions": len(samples),
        "target": args.target,
        "target_mean": float(targets.mean()),
        "target_std": float(targets.std()),
        "target_mean_after_all_four_symmetries": float(
            np.mean(np.concatenate((targets, targets, -targets, -targets)))
        ),
        "by_absolute_player_to_move": by_player,
        "policy_left_minus_right_mean": float(
            np.mean(plane_mass[:, 0] - plane_mass[:, 2])
        ),
        "augmentation_draw_counts": {
            symmetry.name: int(symmetry_counts[index])
            for index, symmetry in enumerate(Symmetry)
        },
        "distinct_symmetries_seen_per_sample": {
            str(key): value for key, value in sorted(coverage.items())
        },
        "all_four_symmetries_seen_fraction": float(
            coverage.get(4, 0) / len(samples)
        ),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "seed": args.seed,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
