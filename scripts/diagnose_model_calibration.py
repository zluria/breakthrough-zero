#!/usr/bin/env python3
"""Report held-out policy and absolute-value calibration for one checkpoint."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import numpy as np

from breakthrough_zero.data import PositionRecord, load_chunk, split_game_indices
from breakthrough_zero.network import KerasEvaluator, load_network
from breakthrough_zero.training import samples_from_games


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("data", type=Path)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=202608112301)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-positions", type=int)
    args = parser.parse_args()
    if not args.model.is_file():
        parser.error(f"model does not exist: {args.model}")
    if not 0 < args.validation_fraction < 1:
        parser.error("--validation-fraction must be between zero and one")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.max_positions is not None and args.max_positions < 1:
        parser.error("--max-positions must be positive")
    return args


def load_games(root: Path):
    paths = [root] if root.is_file() else sorted(root.rglob("chunk_*.npz"))
    if not paths:
        raise ValueError(f"no self-play chunks found under {root}")
    games = []
    inputs = []
    for path in paths:
        chunk, manifest = load_chunk(path)
        games.extend(chunk)
        inputs.append({"path": str(path.resolve()), "sha256": manifest["sha256"]})
    return games, inputs


def target_policy(position: PositionRecord) -> np.ndarray:
    visits = np.asarray(
        [action.visits for action in position.actions], dtype=np.float64
    )
    if visits.sum() > 0:
        weights = visits / visits.sum()
    else:
        weights = np.asarray(
            [action.network_prior for action in position.actions], dtype=np.float64
        )
        weights /= weights.sum()
    target = np.zeros(position.state.rules.action_size, dtype=np.float64)
    for action, weight in zip(position.actions, weights, strict=True):
        target[position.state.policy_index(action.move)] = weight
    return target


def immediate_win_mask(position: PositionRecord) -> np.ndarray:
    mask = np.zeros(position.state.rules.action_size, dtype=np.bool_)
    child = position.state.clone()
    mover = position.state.to_move
    for action in position.actions:
        undo = child.make_move(action.move, validate=False)
        if child.outcome == mover:
            mask[position.state.policy_index(action.move)] = True
        child.unmake_move(action.move, undo)
    return mask


def distribution(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
        "p99": float(np.quantile(values, 0.99)),
        "maximum": float(np.max(values)),
    }


def calibration_table(
    predictions: np.ndarray, outcomes: np.ndarray, root_values: np.ndarray
) -> list[dict[str, float | int | str]]:
    boundaries = np.linspace(-1.0, 1.0, 6)
    indices = np.minimum(
        np.searchsorted(boundaries, predictions, side="right") - 1, 4
    )
    rows = []
    for index in range(5):
        selected = indices == index
        if not np.any(selected):
            continue
        rows.append(
            {
                "range": f"[{boundaries[index]:.1f}, {boundaries[index + 1]:.1f}]",
                "positions": int(np.sum(selected)),
                "mean_prediction": float(np.mean(predictions[selected])),
                "p1_win_fraction": float(np.mean(outcomes[selected] == 1)),
                "mean_root_q": float(np.mean(root_values[selected])),
            }
        )
    return rows


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    games, inputs = load_games(args.data)
    _, validation_indices = split_game_indices(
        len(games), args.validation_fraction, seed=args.split_seed
    )
    samples = samples_from_games([games[index] for index in validation_indices])
    if args.max_positions is not None and len(samples) > args.max_positions:
        chosen = np.random.default_rng(args.split_seed + 2).choice(
            len(samples), size=args.max_positions, replace=False
        )
        samples = [samples[int(index)] for index in sorted(chosen)]

    evaluator = KerasEvaluator(load_network(args.model))
    evaluations = []
    for start in range(0, len(samples), args.batch_size):
        states = [
            sample.position.state
            for sample in samples[start : start + args.batch_size]
        ]
        evaluations.extend(evaluator.evaluate_batch(states))

    predictions = np.asarray([value for _, value in evaluations], dtype=np.float64)
    outcomes = np.asarray([sample.outcome for sample in samples], dtype=np.int8)
    root_values = np.asarray(
        [sample.position.root_q for sample in samples], dtype=np.float64
    )
    players = np.asarray(
        [sample.position.state.to_move for sample in samples], dtype=np.int8
    )
    policy_kls = []
    policy_top_matches = []
    immediate_win_masses = []
    immediate_win_top_matches = []
    for sample, (prediction, _) in zip(samples, evaluations, strict=True):
        target = target_policy(sample.position)
        positive = target > 0
        policy_kls.append(
            float(
                np.sum(
                    target[positive]
                    * np.log(
                        target[positive]
                        / np.maximum(prediction[positive], np.finfo(float).tiny)
                    )
                )
            )
        )
        predicted_action = int(np.argmax(prediction))
        policy_top_matches.append(
            int(target[predicted_action] == np.max(target))
        )
        winning = immediate_win_mask(sample.position)
        if np.any(winning):
            immediate_win_masses.append(float(np.sum(prediction[winning])))
            immediate_win_top_matches.append(int(winning[predicted_action]))

    by_player: dict[str, dict[str, float | int]] = {}
    for player in (1, -1):
        selected = players == player
        by_player[str(player)] = {
            "positions": int(np.sum(selected)),
            "mean_prediction": float(np.mean(predictions[selected])),
            "mean_root_q": float(np.mean(root_values[selected])),
            "p1_win_fraction": float(np.mean(outcomes[selected] == 1)),
            "outcome_mae": float(
                np.mean(np.abs(predictions[selected] - outcomes[selected]))
            ),
            "root_q_mae": float(
                np.mean(np.abs(predictions[selected] - root_values[selected]))
            ),
        }

    report: dict[str, Any] = {
        "model": str(args.model.resolve()),
        "model_sha256": file_sha256(args.model),
        "inputs": inputs,
        "games": len(games),
        "validation_games": len(validation_indices),
        "validation_positions": len(samples),
        "validation_fraction": args.validation_fraction,
        "split_seed": args.split_seed,
        "rules": samples[0].position.state.rules.name,
        "policy_kl": distribution(np.asarray(policy_kls)),
        "policy_top_move_agreement": float(np.mean(policy_top_matches)),
        "value_prediction": distribution(predictions),
        "outcome_mae": float(np.mean(np.abs(predictions - outcomes))),
        "outcome_brier": float(
            np.mean(np.square((predictions + 1) / 2 - (outcomes + 1) / 2))
        ),
        "outcome_sign_accuracy": float(
            np.mean((predictions >= 0) == (outcomes == 1))
        ),
        "root_q_mae": float(np.mean(np.abs(predictions - root_values))),
        "root_q_bias": float(np.mean(predictions - root_values)),
        "by_absolute_player_to_move": by_player,
        "calibration": calibration_table(predictions, outcomes, root_values),
        "positions_with_immediate_win": len(immediate_win_masses),
        "immediate_win_policy_mass": (
            distribution(np.asarray(immediate_win_masses))
            if immediate_win_masses
            else None
        ),
        "immediate_win_policy_top_fraction": (
            float(np.mean(immediate_win_top_matches))
            if immediate_win_top_matches
            else None
        ),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
