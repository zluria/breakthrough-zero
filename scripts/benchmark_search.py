"""Measure scalar PUCT overhead separately from rollout or neural inference."""

from __future__ import annotations

import argparse
import json
from math import sqrt
from statistics import median
from time import perf_counter

import numpy as np

from breakthrough_zero.game import ACTION_SIZE, PLAYER_1, GameState
from breakthrough_zero.search import Node, PUCTSearch, SearchConfig, backup


class ZeroEvaluator:
    """Uniform policy with a zero absolute value, for search timing only."""

    def __init__(self) -> None:
        self.policy = np.ones(ACTION_SIZE, dtype=np.float32)

    def evaluate(self, state: GameState) -> tuple[np.ndarray, float]:
        return self.policy, 0.0


def replay_search(search: PUCTSearch, state: GameState) -> Node:
    """Reference strategy: clone the root and replay the selected path."""

    root = Node(state=state.clone())
    for _ in range(search.config.simulations):
        position = state.clone()
        node = root
        path = [root]

        while node.expanded and node.children:
            direction = 1.0 if position.to_move == PLAYER_1 else -1.0
            scale = sqrt(node.visits)

            def score(item):
                _, child = item
                exploration = (
                    search.config.c_puct
                    * child.prior
                    * scale
                    / (1 + child.visits)
                )
                return direction * child.q + exploration

            move, node = max(node.children.items(), key=score)
            position.make_move(move, validate=False)
            path.append(node)

        if position.outcome is not None:
            value = float(position.outcome)
        else:
            policy, value = search.evaluator.evaluate(position)
            search._expand(node, position, policy)
        node.evaluation = value
        backup(path, value)

    return root


def measure(function, search: PUCTSearch, searches: int) -> float:
    start = perf_counter()
    for _ in range(searches):
        function(search, GameState())
    return perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-simulations", type=int, default=20000)
    parser.add_argument("--budgets", type=int, nargs="+", default=[32, 100, 400])
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()

    results = {}
    for budget in args.budgets:
        searches = max(1, args.total_simulations // budget)
        search = PUCTSearch(
            ZeroEvaluator(), SearchConfig(simulations=budget, c_puct=1.5)
        )
        cached_times = []
        replay_times = []
        for round_index in range(args.rounds):
            methods = (
                ((lambda engine, state: engine.run(state)), replay_search)
                if round_index % 2 == 0
                else (replay_search, (lambda engine, state: engine.run(state)))
            )
            elapsed = [measure(method, search, searches) for method in methods]
            if round_index % 2 == 0:
                cached_times.append(elapsed[0])
                replay_times.append(elapsed[1])
            else:
                replay_times.append(elapsed[0])
                cached_times.append(elapsed[1])

        total = searches * budget
        cached_rate = total / median(cached_times)
        replay_rate = total / median(replay_times)
        results[str(budget)] = {
            "cached_simulations_per_second": cached_rate,
            "replay_simulations_per_second": replay_rate,
            "replay_over_cached": replay_rate / cached_rate,
            "searches_per_round": searches,
        }

    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
