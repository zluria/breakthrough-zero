"""Verify raw self-play chunks and print compact training-data diagnostics."""

from __future__ import annotations

import argparse
import json
from math import log
from pathlib import Path

import numpy as np

from breakthrough_zero.data import load_chunk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    return parser.parse_args()


def main() -> None:
    root = parse_args().root
    groups = [path for path in sorted(root.iterdir()) if path.is_dir()]
    if not groups:
        groups = [root]
    summaries = [_summarize(group) for group in groups]
    print(json.dumps(summaries, indent=2, sort_keys=True))


def _summarize(directory: Path) -> dict[str, object]:
    paths = sorted(directory.glob("chunk_*.npz"))
    if not paths:
        raise ValueError(f"no chunks found in {directory}")

    games = []
    manifests = []
    for path in paths:
        chunk_games, manifest = load_chunk(path)
        games.extend(chunk_games)
        manifests.append(manifest)

    positions = [position for game in games for position in game.positions]
    entropies = []
    top_shares = []
    soft_z_errors = []
    for game in games:
        for position in game.positions:
            visits = np.array(
                [action.visits for action in position.actions], dtype=np.float64
            )
            if visits.sum() == 0:
                weights = np.array(
                    [action.network_prior for action in position.actions],
                    dtype=np.float64,
                )
                weights /= weights.sum()
            else:
                weights = visits / visits.sum()
            positive = weights[weights > 0]
            entropy = -float(np.sum(positive * np.log(positive)))
            maximum_entropy = log(len(weights)) if len(weights) > 1 else 1.0
            entropies.append(entropy / maximum_entropy)
            top_shares.append(float(weights.max()))
            soft_z_errors.append(abs(position.root_q - game.outcome))

    return {
        "directory": str(directory.resolve()),
        "chunks": len(paths),
        "games": len(games),
        "positions": len(positions),
        "mean_game_plies": round(float(np.mean([len(game.positions) for game in games])), 3),
        "p1_win_fraction": round(float(np.mean([game.outcome == 1 for game in games])), 4),
        "mean_normalized_policy_entropy": round(float(np.mean(entropies)), 4),
        "mean_top_visit_share": round(float(np.mean(top_shares)), 4),
        "mean_absolute_soft_z_error": round(float(np.mean(soft_z_errors)), 4),
        "bytes": sum(path.stat().st_size for path in paths),
        "schema_versions": sorted({manifest["schema_version"] for manifest in manifests}),
    }


if __name__ == "__main__":
    main()
