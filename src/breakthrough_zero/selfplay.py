"""Reproducible self-play that preserves raw, absolute search statistics."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
import random
from typing import Iterator, Protocol

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


class BatchEvaluator(Evaluator, Protocol):
    """An evaluator that can score leaves from independent trees together."""

    def evaluate_batch(
        self, states: Sequence[GameState]
    ) -> tuple[tuple[np.ndarray, float], ...]:
        """Return one policy and absolute Player-1 value per state."""


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


@dataclass(slots=True)
class _BatchedGame:
    """Mutable orchestration state for one independent game."""

    index: int
    record_seed: int
    state: GameState
    search: PUCTSearch
    move_rng: np.random.Generator
    positions: list[PositionRecord] = field(default_factory=list)


def play_batched_games(
    evaluator: BatchEvaluator,
    config: SelfPlayConfig,
    seeds: Sequence[int],
    *,
    rules: Ruleset = STANDARD_RULES,
    batch_size: int = 16,
) -> tuple[GameRecord, ...]:
    """Play independent games while batching one leaf from each active tree.

    Every tree still performs ordinary scalar PUCT. Only the network boundary
    is shared, so no virtual loss or thread-dependent tree updates are needed.
    Finished slots are refilled to keep the inference batch useful.
    """

    if not seeds:
        raise ValueError("at least one game seed is required")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if any(not 0 <= seed < 2**64 for seed in seeds):
        raise ValueError("game seeds must fit in unsigned 64-bit integers")

    completed: list[GameRecord | None] = [None] * len(seeds)
    active: list[_BatchedGame] = []
    next_index = 0

    def fill_slots() -> None:
        nonlocal next_index
        while len(active) < batch_size and next_index < len(seeds):
            record_seed = seeds[next_index]
            streams = random.Random(record_seed)
            active.append(
                _BatchedGame(
                    index=next_index,
                    record_seed=record_seed,
                    state=GameState.initial(rules),
                    search=PUCTSearch(
                        evaluator,
                        config.search,
                        seed=streams.getrandbits(64),
                    ),
                    move_rng=np.random.default_rng(streams.getrandbits(64)),
                )
            )
            next_index += 1

    fill_slots()
    while active:
        if any(len(slot.positions) >= config.max_plies for slot in active):
            raise RuntimeError("self-play exceeded the configured ply limit")
        roots = [Node(state=slot.state.clone()) for slot in active]
        for _ in range(config.search.simulations):
            pending = [
                slot.search.begin_simulation(
                    root, root_noise=config.root_noise
                )
                for slot, root in zip(active, roots, strict=True)
            ]
            needs_network = [item for item in pending if item.needs_evaluation]
            evaluations = evaluator.evaluate_batch(
                [item.position for item in needs_network]
            )
            if len(evaluations) != len(needs_network):
                raise RuntimeError("batch evaluator returned the wrong result count")
            evaluated = iter(evaluations)
            for slot, item in zip(active, pending, strict=True):
                evaluation = next(evaluated) if item.needs_evaluation else None
                slot.search.complete_simulation(item, evaluation)

        survivors = []
        for slot, root in zip(active, roots, strict=True):
            expected_child_visits = root.visits - 1
            child_visits = sum(child.visits for child in root.children.values())
            if child_visits != expected_child_visits:
                raise RuntimeError("root visit accounting is inconsistent")

            temperature = (
                config.temperature
                if slot.state.ply < config.sample_until_ply
                else 0.0
            )
            selected_move = sample_move(
                root, slot.move_rng, temperature=temperature
            )
            slot.positions.append(
                PositionRecord.from_search(slot.state, root, selected_move)
            )
            slot.state.make_move(selected_move, validate=False)
            if slot.state.outcome is None:
                survivors.append(slot)
            else:
                completed[slot.index] = GameRecord(
                    positions=tuple(slot.positions),
                    outcome=int(slot.state.outcome),
                    seed=slot.record_seed,
                )
        active[:] = survivors
        fill_slots()

    if any(game is None for game in completed):
        raise AssertionError("batched self-play lost a game result")
    return tuple(game for game in completed if game is not None)


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
