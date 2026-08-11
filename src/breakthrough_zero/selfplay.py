"""Reproducible self-play that preserves raw, absolute search statistics."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterator

import numpy as np

from .data import GameRecord, PositionRecord
from .evaluators import RandomRolloutEvaluator
from .game import STANDARD_RULES, GameState, Move, Ruleset
from .search import (
    Evaluator,
    Node,
    PUCTSearch,
    RootNoiseConfig,
    SearchConfig,
    best_move,
)


@dataclass(frozen=True, slots=True)
class SelfPlayConfig:
    """Search and move-selection settings for one self-play game."""

    search: SearchConfig = SearchConfig()
    sample_until_ply: int = 12
    temperature: float = 1.0
    max_plies: int = 128
    root_noise: RootNoiseConfig | None = None

    def __post_init__(self) -> None:
        if self.sample_until_ply < 0:
            raise ValueError("sample_until_ply cannot be negative")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.max_plies < 1:
            raise ValueError("max_plies must be positive")


def play_game(
    evaluator: Evaluator,
    config: SelfPlayConfig,
    *,
    seed: int,
    initial_state: GameState | None = None,
) -> GameRecord:
    """Play one game with a caller-supplied evaluator.

    ``seed`` controls search noise and visit sampling. If the evaluator is
    stochastic, its own seed must also be reproducible.
    """

    streams = random.Random(seed)
    return _play_game(
        evaluator,
        config,
        record_seed=seed,
        search_seed=streams.getrandbits(64),
        move_seed=streams.getrandbits(64),
        initial_state=initial_state,
    )


def play_dummy_game(
    config: SelfPlayConfig,
    *,
    seed: int,
    prefer_tactical_rollouts: bool = False,
    initial_state: GameState | None = None,
) -> GameRecord:
    """Play one sanity-check game with uniform policy and random rollouts."""

    streams = random.Random(seed)
    evaluator = RandomRolloutEvaluator(
        streams.getrandbits(64), prefer_tactical=prefer_tactical_rollouts
    )
    return _play_game(
        evaluator,
        config,
        record_seed=seed,
        search_seed=streams.getrandbits(64),
        move_seed=streams.getrandbits(64),
        initial_state=initial_state,
    )


def generate_dummy_games(
    count: int,
    config: SelfPlayConfig,
    *,
    seed: int,
    prefer_tactical_rollouts: bool = False,
    rules: Ruleset = STANDARD_RULES,
) -> Iterator[GameRecord]:
    """Yield independent games whose individual seeds are stored in the data."""

    if count < 1:
        raise ValueError("game count must be positive")
    seed_source = random.Random(seed)
    for _ in range(count):
        yield play_dummy_game(
            config,
            seed=seed_source.getrandbits(64),
            prefer_tactical_rollouts=prefer_tactical_rollouts,
            initial_state=GameState.initial(rules),
        )


def sample_move(
    root: Node, rng: np.random.Generator, *, temperature: float
) -> Move:
    """Sample from root visits, or choose deterministically at temperature 0."""

    if not root.children:
        raise ValueError("cannot choose a move from an unexpanded root")
    if temperature < 0:
        raise ValueError("temperature cannot be negative")
    if temperature == 0:
        return best_move(root)

    moves = list(root.children)
    visits = np.array(
        [root.children[move].visits for move in moves], dtype=np.float64
    )
    if visits.sum() == 0:
        weights = np.array(
            [root.children[move].prior for move in moves], dtype=np.float64
        )
    elif temperature == 1:
        weights = visits
    else:
        weights = np.power(visits, 1.0 / temperature)

    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0:
        raise RuntimeError("move-sampling weights are not usable")
    index = int(rng.choice(len(moves), p=weights / total))
    return moves[index]


def _play_game(
    evaluator: Evaluator,
    config: SelfPlayConfig,
    *,
    record_seed: int,
    search_seed: int,
    move_seed: int,
    initial_state: GameState | None,
) -> GameRecord:
    if not 0 <= record_seed < 2**64:
        raise ValueError("game seed must fit in an unsigned 64-bit integer")

    state = initial_state.clone() if initial_state is not None else GameState()
    if state.outcome is not None:
        raise ValueError("self-play cannot start from a terminal state")

    search = PUCTSearch(evaluator, config.search, seed=search_seed)
    move_rng = np.random.default_rng(move_seed)
    positions: list[PositionRecord] = []

    while state.outcome is None:
        if len(positions) >= config.max_plies:
            raise RuntimeError("self-play exceeded the configured ply limit")

        root = search.run(state, root_noise=config.root_noise)
        expected_child_visits = root.visits - 1
        if sum(child.visits for child in root.children.values()) != expected_child_visits:
            raise RuntimeError("root visit accounting is inconsistent")

        temperature = (
            config.temperature if state.ply < config.sample_until_ply else 0.0
        )
        selected_move = sample_move(root, move_rng, temperature=temperature)
        positions.append(PositionRecord.from_search(state, root, selected_move))
        state.make_move(selected_move, validate=False)

    return GameRecord(
        positions=tuple(positions), outcome=int(state.outcome), seed=record_seed
    )
