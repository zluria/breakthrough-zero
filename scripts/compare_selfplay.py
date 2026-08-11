#!/usr/bin/env python3
"""Compare two self-play runs that used the same per-game seeds."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import mean

from breakthrough_zero.data import GameRecord, load_chunk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    return parser.parse_args()


def load_games(directory: Path) -> dict[int, GameRecord]:
    games: dict[int, GameRecord] = {}
    paths = sorted(directory.glob("chunk_*.npz"))
    if not paths:
        raise ValueError(f"no chunks found in {directory}")
    for path in paths:
        chunk, _ = load_chunk(path)
        for game in chunk:
            if game.seed in games:
                raise ValueError(f"duplicate game seed {game.seed} in {directory}")
            games[game.seed] = game
    return games


def moves(game: GameRecord, depth: int | None = None) -> tuple[tuple[int, int], ...]:
    positions = game.positions if depth is None else game.positions[:depth]
    return tuple(
        (position.selected_move.source, position.selected_move.target)
        for position in positions
    )


def main() -> None:
    args = parse_args()
    left = load_games(args.left)
    right = load_games(args.right)
    if left.keys() != right.keys():
        raise ValueError("runs do not contain exactly the same game seeds")

    seeds = sorted(left)
    transitions = Counter((left[seed].outcome, right[seed].outcome) for seed in seeds)
    result = {
        "games": len(seeds),
        "left_p1_win_fraction": mean(left[seed].outcome == 1 for seed in seeds),
        "right_p1_win_fraction": mean(right[seed].outcome == 1 for seed in seeds),
        "left_mean_plies": mean(len(left[seed].positions) for seed in seeds),
        "right_mean_plies": mean(len(right[seed].positions) for seed in seeds),
        "outcome_transitions": {
            f"{before:+d} -> {after:+d}": count
            for (before, after), count in sorted(transitions.items())
        },
        "identical_trajectories": sum(
            moves(left[seed]) == moves(right[seed]) for seed in seeds
        ),
        "identical_prefixes": {
            str(depth): sum(
                moves(left[seed], depth) == moves(right[seed], depth)
                for seed in seeds
            )
            for depth in (4, 8, 12)
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
