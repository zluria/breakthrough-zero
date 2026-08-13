"""Verify raw self-play chunks and print compact training-data diagnostics."""

from __future__ import annotations

import argparse
import json
from math import log
from pathlib import Path

import numpy as np

from breakthrough_zero.data import PositionRecord, load_chunk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    return parser.parse_args()


def main() -> None:
    root = parse_args().root
    groups = [path for path in sorted(root.iterdir()) if path.is_dir()]
    if not groups:
        groups = [root]
    summaries = [summarize_corpus(group) for group in groups]
    print(json.dumps(summaries, indent=2, sort_keys=True))


def summarize_corpus(directory: Path) -> dict[str, object]:
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
    network_entropies = []
    top_shares = []
    soft_z_errors = []
    search_network_prior_kls = []
    low_prior_actions = 0
    visited_low_prior_actions = 0
    low_prior_visit_masses = []
    immediate_win_positions = 0
    immediate_win_top_visits = 0
    immediate_win_selections = 0
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

            priors = np.array(
                [action.network_prior for action in position.actions],
                dtype=np.float64,
            )
            priors /= priors.sum()
            positive_priors = priors[priors > 0]
            prior_entropy = -float(
                np.sum(positive_priors * np.log(positive_priors))
            )
            network_entropies.append(prior_entropy / maximum_entropy)

            diagnostic = _position_diagnostic(position, weights)
            search_network_prior_kls.append(diagnostic["search_network_prior_kl"])
            low_prior_actions += diagnostic["low_prior_actions"]
            visited_low_prior_actions += diagnostic["visited_low_prior_actions"]
            low_prior_visit_masses.append(diagnostic["low_prior_visit_mass"])
            immediate_win_positions += diagnostic["has_immediate_win"]
            immediate_win_top_visits += diagnostic["immediate_win_has_top_visit"]
            immediate_win_selections += diagnostic["immediate_win_selected"]

    state_keys = {
        (p.state.p1, p.state.p2, p.state.to_move) for p in positions
    }
    trajectories = {
        tuple((p.selected_move.source, p.selected_move.target) for p in game.positions)
        for game in games
    }
    prefix_counts = {
        str(depth): len(
            {
                tuple(
                    (p.selected_move.source, p.selected_move.target)
                    for p in game.positions[:depth]
                )
                for game in games
            }
        )
        for depth in (4, 8, 12)
    }
    generation_seconds = [
        manifest.get("metadata", {}).get("generation_seconds")
        for manifest in manifests
    ]
    recorded_generation_seconds = (
        float(sum(generation_seconds))
        if all(seconds is not None for seconds in generation_seconds)
        else None
    )

    return {
        "directory": str(directory.resolve()),
        "chunks": len(paths),
        "games": len(games),
        "positions": len(positions),
        "mean_game_plies": round(float(np.mean([len(game.positions) for game in games])), 3),
        "p1_win_fraction": round(float(np.mean([game.outcome == 1 for game in games])), 4),
        "mean_normalized_policy_entropy": round(float(np.mean(entropies)), 4),
        "mean_normalized_network_entropy": round(
            float(np.mean(network_entropies)), 4
        ),
        "mean_top_visit_share": round(float(np.mean(top_shares)), 4),
        "mean_absolute_soft_z_error": round(float(np.mean(soft_z_errors)), 4),
        "mean_search_network_prior_kl": round(
            float(np.mean(search_network_prior_kls)), 6
        ),
        "low_network_prior_actions": low_prior_actions,
        "visited_low_network_prior_fraction": round(
            visited_low_prior_actions / low_prior_actions, 4
        ) if low_prior_actions else None,
        "mean_visit_mass_on_low_network_prior_moves": round(
            float(np.mean(low_prior_visit_masses)), 4
        ),
        "positions_with_immediate_win": immediate_win_positions,
        "immediate_win_top_visit_fraction": round(
            immediate_win_top_visits / immediate_win_positions, 4
        ) if immediate_win_positions else None,
        "immediate_win_selected_fraction": round(
            immediate_win_selections / immediate_win_positions, 4
        ) if immediate_win_positions else None,
        "unique_positions": len(state_keys),
        "unique_position_fraction": round(len(state_keys) / len(positions), 4),
        "unique_trajectories": len(trajectories),
        "unique_prefixes": prefix_counts,
        "root_visit_counts": sorted({p.root_visits for p in positions}),
        "recorded_generation_seconds": recorded_generation_seconds,
        "recorded_generation_positions_per_second": (
            round(len(positions) / recorded_generation_seconds, 3)
            if recorded_generation_seconds
            else None
        ),
        "bytes": sum(path.stat().st_size for path in paths),
        "schema_versions": sorted({manifest["schema_version"] for manifest in manifests}),
    }


def _position_diagnostic(
    position: PositionRecord, visit_weights: np.ndarray
) -> dict[str, float | int]:
    """Measure exploration without confusing randomness with useful coverage."""

    network = np.asarray(
        [action.network_prior for action in position.actions], dtype=np.float64
    )
    search = np.asarray(
        [action.prior for action in position.actions], dtype=np.float64
    )
    network_total = float(network.sum())
    search_total = float(search.sum())
    if (
        not np.all(np.isfinite(network))
        or not np.all(np.isfinite(search))
        or np.any(network < 0)
        or np.any(search < 0)
        or network_total <= 0
        or search_total <= 0
    ):
        raise ValueError("stored search and network priors must be distributions")
    network /= network_total
    search /= search_total
    positive_search = search > 0
    prior_kl = float(
        np.sum(
            search[positive_search]
            * np.log(
                search[positive_search]
                / np.maximum(network[positive_search], np.finfo(float).tiny)
            )
        )
    )

    low_prior = network < 0.5 / len(network)
    visits = np.asarray(
        [action.visits for action in position.actions], dtype=np.int64
    )
    winning = []
    mover = position.state.to_move
    child = position.state.clone()
    for action in position.actions:
        undo = child.make_move(action.move, validate=False)
        winning.append(child.outcome == mover)
        child.unmake_move(action.move, undo)
    winning_mask = np.asarray(winning, dtype=np.bool_)
    has_immediate_win = bool(winning_mask.any())
    top_visit = visits == visits.max()

    return {
        "search_network_prior_kl": prior_kl,
        "low_prior_actions": int(low_prior.sum()),
        "visited_low_prior_actions": int(np.sum(low_prior & (visits > 0))),
        "low_prior_visit_mass": float(np.sum(visit_weights[low_prior])),
        "has_immediate_win": int(has_immediate_win),
        "immediate_win_has_top_visit": int(
            has_immediate_win and np.any(winning_mask & top_visit)
        ),
        "immediate_win_selected": int(
            has_immediate_win
            and any(
                is_winning and action.move == position.selected_move
                for action, is_winning in zip(
                    position.actions, winning, strict=True
                )
            )
        ),
    }


if __name__ == "__main__":
    main()
