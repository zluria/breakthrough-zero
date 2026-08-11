"""Short, deterministic benchmark for the rules and rollout hot paths."""

from __future__ import annotations

import argparse
import json
import random
from time import perf_counter

from breakthrough_zero.game import GameState
from breakthrough_zero.reference import reference_legal_moves


def sample_positions(count: int, seed: int) -> list[GameState]:
    rng = random.Random(seed)
    state = GameState()
    positions = []
    while len(positions) < count:
        if state.outcome is not None:
            state = GameState()
        positions.append(state.clone())
        state.make_move(state.random_legal_move(rng), validate=False)
    return positions


def timed_calls(function, positions: list[GameState], repeats: int) -> float:
    start = perf_counter()
    for _ in range(repeats):
        for state in positions:
            function(state)
    return perf_counter() - start


def rollout(games: int, *, method: str, seed: int) -> tuple[float, int]:
    rng = random.Random(seed)
    moves = 0
    start = perf_counter()
    for _ in range(games):
        state = GameState()
        while state.outcome is None:
            if method == "list":
                move = rng.choice(state.legal_moves())
            else:
                move = state.random_legal_move(
                    rng, prefer_tactical=(method == "tactical")
                )
            state.make_move(move, validate=False)
            moves += 1
    return perf_counter() - start, moves


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--positions", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--games", type=int, default=300)
    args = parser.parse_args()

    positions = sample_positions(args.positions, seed=7)
    rng = random.Random(11)
    fast_time = timed_calls(lambda state: state.legal_moves(), positions, args.repeats)
    slow_time = timed_calls(reference_legal_moves, positions, args.repeats)
    list_time = timed_calls(
        lambda state: rng.choice(state.legal_moves()), positions, args.repeats
    )
    bit_time = timed_calls(
        lambda state: state.random_legal_move(rng), positions, args.repeats
    )
    tactical_time = timed_calls(
        lambda state: state.random_legal_move(rng, prefer_tactical=True),
        positions,
        args.repeats,
    )
    list_rollout_time, list_moves = rollout(args.games, method="list", seed=13)
    uniform_rollout_time, uniform_moves = rollout(
        args.games, method="uniform", seed=13
    )
    tactical_rollout_time, tactical_moves = rollout(
        args.games, method="tactical", seed=13
    )

    calls = args.positions * args.repeats
    result = {
        "legal_calls": calls,
        "fast_legal_calls_per_second": calls / fast_time,
        "reference_legal_calls_per_second": calls / slow_time,
        "fast_legal_speedup": slow_time / fast_time,
        "list_choice_per_second": calls / list_time,
        "bit_choice_per_second": calls / bit_time,
        "tactical_choice_per_second": calls / tactical_time,
        "list_rollout_games_per_second": args.games / list_rollout_time,
        "list_rollout_moves_per_second": list_moves / list_rollout_time,
        "uniform_rollout_moves_per_second": uniform_moves / uniform_rollout_time,
        "uniform_rollout_games_per_second": args.games / uniform_rollout_time,
        "tactical_rollout_moves_per_second": tactical_moves / tactical_rollout_time,
        "tactical_rollout_games_per_second": args.games / tactical_rollout_time,
        "list_mean_game_length": list_moves / args.games,
        "uniform_mean_game_length": uniform_moves / args.games,
        "tactical_mean_game_length": tactical_moves / args.games,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
