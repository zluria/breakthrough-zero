#!/usr/bin/env python3
"""Measure learned value and policy consistency under exact game symmetries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np

from breakthrough_zero.game import PLAYER_1, GameState, STANDARD_RULES
from breakthrough_zero.network import KerasEvaluator, load_network
from breakthrough_zero.symmetry import Symmetry, transform_move, transform_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--states", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-random-plies", type=int, default=48)
    parser.add_argument("--seed", type=int, default=20260812)
    args = parser.parse_args()
    if args.states < 1 or args.batch_size < 1 or args.max_random_plies < 1:
        parser.error("counts and random-ply limit must be positive")
    if not args.model.is_file():
        parser.error(f"model does not exist: {args.model}")
    return args


def sample_states(count: int, *, max_plies: int, seed: int) -> list[GameState]:
    rng = random.Random(seed)
    states = []
    while len(states) < count:
        state = GameState.initial(STANDARD_RULES)
        for _ in range(rng.randrange(max_plies + 1)):
            if state.outcome is not None:
                break
            state.make_move(state.random_legal_move(rng), validate=False)
        if state.outcome is None:
            states.append(state)
    return states


def evaluate_all(
    evaluator: KerasEvaluator,
    states: list[GameState],
    *,
    batch_size: int,
):
    evaluations = []
    for start in range(0, len(states), batch_size):
        evaluations.extend(
            evaluator.evaluate_batch(states[start : start + batch_size])
        )
    return evaluations


def policy_comparison(
    state: GameState,
    policy: np.ndarray,
    transformed_state: GameState,
    transformed_policy: np.ndarray,
    symmetry: Symmetry,
) -> tuple[float, float, bool]:
    original = []
    mapped = []
    for move in state.legal_moves():
        transformed_move = transform_move(move, symmetry, state.rules)
        original.append(float(policy[state.policy_index(move)]))
        mapped.append(
            float(
                transformed_policy[
                    transformed_state.policy_index(transformed_move)
                ]
            )
        )
    p = np.asarray(original, dtype=np.float64)
    q = np.asarray(mapped, dtype=np.float64)
    midpoint = 0.5 * (p + q)
    js = 0.5 * float(np.sum(p * np.log(p / midpoint)))
    js += 0.5 * float(np.sum(q * np.log(q / midpoint)))
    return float(np.sum(np.abs(p - q))), js, int(np.argmax(p)) == int(np.argmax(q))


def distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
    }


def main() -> None:
    args = parse_args()
    evaluator = KerasEvaluator(load_network(args.model))
    states = sample_states(
        args.states,
        max_plies=args.max_random_plies,
        seed=args.seed,
    )
    swapped = [transform_state(state, Symmetry.SWAP_PLAYERS) for state in states]
    mirrored = [
        transform_state(state, Symmetry.MIRROR_LEFT_RIGHT) for state in states
    ]
    all_evaluations = evaluate_all(
        evaluator,
        states + swapped + mirrored,
        batch_size=args.batch_size,
    )
    original_eval = all_evaluations[: args.states]
    swapped_eval = all_evaluations[args.states : 2 * args.states]
    mirrored_eval = all_evaluations[2 * args.states :]

    swap_value_errors = []
    mirror_value_errors = []
    swap_policy_l1 = []
    swap_policy_js = []
    swap_top_matches = []
    mirror_policy_l1 = []
    mirror_policy_js = []
    mirror_top_matches = []
    for index, state in enumerate(states):
        policy, value = original_eval[index]
        swap_policy, swap_value = swapped_eval[index]
        mirror_policy, mirror_value = mirrored_eval[index]
        swap_value_errors.append(abs(value + swap_value))
        mirror_value_errors.append(abs(value - mirror_value))

        l1, js, top = policy_comparison(
            state,
            policy,
            swapped[index],
            swap_policy,
            Symmetry.SWAP_PLAYERS,
        )
        swap_policy_l1.append(l1)
        swap_policy_js.append(js)
        swap_top_matches.append(top)
        l1, js, top = policy_comparison(
            state,
            policy,
            mirrored[index],
            mirror_policy,
            Symmetry.MIRROR_LEFT_RIGHT,
        )
        mirror_policy_l1.append(l1)
        mirror_policy_js.append(js)
        mirror_top_matches.append(top)

    initial = GameState.initial(STANDARD_RULES)
    initial_swap = transform_state(initial, Symmetry.SWAP_PLAYERS)
    initial_values = evaluator.evaluate_batch((initial, initial_swap))
    by_player = {
        str(player): [
            value
            for state, (_, value) in zip(states, original_eval, strict=True)
            if state.to_move == player
        ]
        for player in (PLAYER_1, -PLAYER_1)
    }
    report = {
        "model": str(args.model.resolve()),
        "states": args.states,
        "batch_size": args.batch_size,
        "initial_value": initial_values[0][1],
        "swapped_initial_value": initial_values[1][1],
        "initial_swap_residual": abs(initial_values[0][1] + initial_values[1][1]),
        "value_swap_absolute_residual": distribution(swap_value_errors),
        "value_mirror_absolute_residual": distribution(mirror_value_errors),
        "policy_swap_l1": distribution(swap_policy_l1),
        "policy_swap_js": distribution(swap_policy_js),
        "policy_swap_top_move_agreement": float(np.mean(swap_top_matches)),
        "policy_mirror_l1": distribution(mirror_policy_l1),
        "policy_mirror_js": distribution(mirror_policy_js),
        "policy_mirror_top_move_agreement": float(np.mean(mirror_top_matches)),
        "mean_value_by_absolute_player_to_move": {
            player: float(np.mean(values)) for player, values in by_player.items()
        },
        "positions_by_absolute_player_to_move": {
            player: len(values) for player, values in by_player.items()
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
