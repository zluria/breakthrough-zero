"""Compare cloning with make/unmake on the alpha-beta DFS workload."""

from __future__ import annotations

import argparse
import random
from statistics import median
from time import perf_counter

from breakthrough_zero.alphabeta import heuristic_value, ordered_moves
from breakthrough_zero.game import MINI_RULES, PLAYER_1, STANDARD_RULES, GameState


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positions", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--mini-depth", type=int, default=5)
    parser.add_argument("--standard-depth", type=int, default=4)
    return parser.parse_args()


def reachable_states(rules, count: int, seed: int) -> list[GameState]:
    rng = random.Random(seed)
    states: list[GameState] = []
    while len(states) < count:
        state = GameState.initial(rules)
        target_ply = rng.randint(4, 10 if rules == MINI_RULES else 16)
        for _ in range(target_ply):
            if state.outcome is not None:
                break
            state.make_move(rng.choice(state.legal_moves()), validate=False)
        if state.outcome is None:
            states.append(state)
    return states


def search_make_unmake(
    state: GameState, depth: int, alpha: float = -1.0, beta: float = 1.0
) -> tuple[float, int]:
    if state.outcome is not None:
        return float(state.outcome), 1
    if depth == 0:
        return heuristic_value(state), 1

    maximizing = state.to_move == PLAYER_1
    value = -1.0 if maximizing else 1.0
    nodes = 1
    for move in ordered_moves(state):
        undo = state.make_move(move, validate=False)
        try:
            child, child_nodes = search_make_unmake(
                state, depth - 1, alpha, beta
            )
        finally:
            state.unmake_move(move, undo)
        nodes += child_nodes
        if maximizing:
            value = max(value, child)
            alpha = max(alpha, value)
        else:
            value = min(value, child)
            beta = min(beta, value)
        if alpha >= beta:
            break
    return value, nodes


def search_clone(
    state: GameState, depth: int, alpha: float = -1.0, beta: float = 1.0
) -> tuple[float, int]:
    if state.outcome is not None:
        return float(state.outcome), 1
    if depth == 0:
        return heuristic_value(state), 1

    maximizing = state.to_move == PLAYER_1
    value = -1.0 if maximizing else 1.0
    nodes = 1
    for move in ordered_moves(state):
        child_state = state.clone()
        child_state.make_move(move, validate=False)
        child, child_nodes = search_clone(child_state, depth - 1, alpha, beta)
        nodes += child_nodes
        if maximizing:
            value = max(value, child)
            alpha = max(alpha, value)
        else:
            value = min(value, child)
            beta = min(beta, value)
        if alpha >= beta:
            break
    return value, nodes


def measure(states: list[GameState], depth: int, repeats: int) -> None:
    expected = [search_clone(state, depth) for state in states]
    actual = [search_make_unmake(state, depth) for state in states]
    if actual != expected:
        raise RuntimeError("the two state strategies searched different trees")

    nodes_per_round = sum(nodes for _, nodes in expected)
    methods = (
        ("clone-child", search_clone),
        ("make-unmake", search_make_unmake),
    )
    rates: dict[str, list[float]] = {name: [] for name, _ in methods}
    for round_index in range(repeats):
        order = methods if round_index % 2 == 0 else tuple(reversed(methods))
        for name, search in order:
            started = perf_counter()
            for state in states:
                search(state, depth)
            elapsed = perf_counter() - started
            rates[name].append(nodes_per_round / elapsed)

    for name, _ in methods:
        samples = ", ".join(f"{rate:,.0f}" for rate in rates[name])
        print(
            f"{name}: median {median(rates[name]):,.0f} nodes/s "
            f"({nodes_per_round:,} nodes/round; samples: {samples})"
        )


def main() -> None:
    args = parse_args()
    cases = (
        ("mini", MINI_RULES, args.mini_depth, 71),
        ("standard", STANDARD_RULES, args.standard_depth, 73),
    )
    for name, rules, depth, seed in cases:
        print(f"{name}: depth={depth}, positions={args.positions}")
        measure(
            reachable_states(rules, args.positions, seed), depth, args.repeats
        )


if __name__ == "__main__":
    main()
