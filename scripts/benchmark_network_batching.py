#!/usr/bin/env python3
"""Measure Keras leaf-evaluation throughput at realistic batch sizes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from time import perf_counter

import numpy as np

from breakthrough_zero.game import GameState, STANDARD_RULES
from breakthrough_zero.network import KerasEvaluator, load_network


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--batch-sizes", default="1,4,8,16,32,64")
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=20260812)
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("--seconds must be positive")
    try:
        args.batch_sizes = tuple(
            int(value) for value in args.batch_sizes.split(",")
        )
    except ValueError as error:
        parser.error(f"invalid --batch-sizes: {error}")
    if not args.batch_sizes or any(size < 1 for size in args.batch_sizes):
        parser.error("batch sizes must be positive")
    return args


def sample_states(count: int, *, seed: int) -> tuple[GameState, ...]:
    """Create varied legal states without using the model being measured."""

    rng = random.Random(seed)
    states = []
    while len(states) < count:
        state = GameState.initial(STANDARD_RULES)
        target_plies = rng.randrange(0, 36)
        for _ in range(target_plies):
            if state.outcome is not None:
                break
            state.make_move(state.random_legal_move(rng), validate=False)
        if state.outcome is None:
            states.append(state)
    return tuple(states)


def measure(
    evaluator: KerasEvaluator,
    states: tuple[GameState, ...],
    *,
    seconds: float,
) -> dict[str, float | int]:
    evaluator.evaluate_batch(states)  # Trace and allocate outside the clock.
    calls = 0
    started = perf_counter()
    while perf_counter() - started < seconds:
        evaluator.evaluate_batch(states)
        calls += 1
    elapsed = perf_counter() - started
    positions = calls * len(states)
    return {
        "batch_size": len(states),
        "calls": calls,
        "elapsed_seconds": elapsed,
        "batch_latency_ms": 1000 * elapsed / calls,
        "positions_per_second": positions / elapsed,
    }


def main() -> None:
    args = parse_args()
    evaluator = KerasEvaluator(load_network(args.model))
    pool = sample_states(max(args.batch_sizes), seed=args.seed)
    results = [
        measure(evaluator, pool[:size], seconds=args.seconds)
        for size in args.batch_sizes
    ]
    print(
        json.dumps(
            {
                "model": str(args.model.resolve()),
                "batch_sizes": list(args.batch_sizes),
                "seconds_per_setting": args.seconds,
                "results": results,
                "numpy": np.__version__,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
