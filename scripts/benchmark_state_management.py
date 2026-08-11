"""Compare state-management choices on real Breakthrough move paths."""

from __future__ import annotations

import argparse
import json
import random
from sys import getsizeof
from time import perf_counter

from breakthrough_zero.game import GameState, Move


def long_game(minimum_depth: int) -> tuple[GameState, list[Move]]:
    """Find one deterministic random game long enough for every benchmark."""

    for seed in range(1000):
        rng = random.Random(seed)
        state = GameState()
        moves = []
        while state.outcome is None:
            move = state.random_legal_move(rng)
            moves.append(move)
            state.make_move(move, validate=False)
        if len(moves) >= minimum_depth:
            return GameState(), moves
    raise RuntimeError("could not find a sufficiently long random game")


def clone_and_replay(root: GameState, moves: list[Move], repeats: int) -> int:
    checksum = 0
    for _ in range(repeats):
        state = root.clone()
        for move in moves:
            state.make_move(move, validate=False)
        checksum ^= state.p1
    return checksum


def reset_and_replay(root: GameState, moves: list[Move], repeats: int) -> int:
    state = root.clone()
    checksum = 0
    for _ in range(repeats):
        state.p1 = root.p1
        state.p2 = root.p2
        state.to_move = root.to_move
        state.winner = root.winner
        state.ply = root.ply
        for move in moves:
            state.make_move(move, validate=False)
        checksum ^= state.p1
    return checksum


def make_and_unmake(root: GameState, moves: list[Move], repeats: int) -> int:
    state = root.clone()
    checksum = 0
    for _ in range(repeats):
        undos = []
        for move in moves:
            undos.append(state.make_move(move, validate=False))
        checksum ^= state.p1
        for move, undo in zip(reversed(moves), reversed(undos), strict=True):
            state.unmake_move(move, undo)
    assert state == root
    return checksum


def lazy_child_creation(root: GameState, moves: list[Move], repeats: int) -> int:
    """Measure the one clone-plus-move paid when a tree node is first visited."""

    parents = [root]
    for move in moves[:-1]:
        child = parents[-1].clone()
        child.make_move(move, validate=False)
        parents.append(child)

    checksum = 0
    for _ in range(repeats):
        for parent, move in zip(parents, moves, strict=True):
            child = parent.clone()
            child.make_move(move, validate=False)
            checksum ^= child.p1
    return checksum


def timed(function, *args) -> float:
    start = perf_counter()
    function(*args)
    return perf_counter() - start


def approximate_state_bytes(state: GameState) -> int:
    """Conservative shallow size including the five referenced integers."""

    return getsizeof(state) + sum(
        getsizeof(value)
        for value in (state.p1, state.p2, state.to_move, state.winner, state.ply)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=5000)
    parser.add_argument("--depths", type=int, nargs="+", default=[4, 8, 16, 32])
    args = parser.parse_args()

    root, complete_game = long_game(max(args.depths))
    results = {
        "approximate_bytes_per_cached_state": approximate_state_bytes(root),
        "depths": {},
        "interpretation": (
            "Lazy caching pays clone+make once per first-visited node; later "
            "traversals have no state-transition work."
        ),
    }
    for depth in args.depths:
        moves = complete_game[:depth]
        replay_repeats = args.repeats
        child_repeats = max(1, args.repeats // depth)
        results["depths"][str(depth)] = {
            "clone_replay_us": 1e6
            * timed(clone_and_replay, root, moves, replay_repeats)
            / replay_repeats,
            "reset_replay_us": 1e6
            * timed(reset_and_replay, root, moves, replay_repeats)
            / replay_repeats,
            "make_unmake_us": 1e6
            * timed(make_and_unmake, root, moves, replay_repeats)
            / replay_repeats,
            "lazy_child_creation_us_per_node": 1e6
            * timed(lazy_child_creation, root, moves, child_repeats)
            / (child_repeats * depth),
        }

    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
