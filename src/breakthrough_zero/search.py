"""A small PUCT search with values fixed to Player 1's point of view."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from math import sqrt
from time import perf_counter
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from .game import ACTION_SIZE, PLAYER_1, GameState, Move


class Evaluator(Protocol):
    """The boundary shared by the dummy evaluator and the Keras model."""

    def evaluate(self, state: GameState) -> tuple[NDArray[np.float32], float]:
        """Return raw policy weights and an absolute Player 1 value."""


@dataclass(slots=True)
class Node:
    """One search node; all stored values are absolute Player 1 values."""

    # A state is cached only after the node is visited. Search treats cached
    # states as read-only and clones a parent once when materializing a child.
    state: GameState | None = None
    prior: float = 1.0
    network_prior: float = 1.0
    parent: Node | None = None
    children: dict[Move, Node] = field(default_factory=dict)
    visits: int = 0
    value_sum: float = 0.0
    value_square_sum: float = 0.0
    evaluation: float | None = None
    expanded: bool = False

    @property
    def q(self) -> float:
        """Mean value, using the parent's Q as first-play urgency."""

        if self.visits:
            return self.value_sum / self.visits
        if self.parent is not None:
            return self.parent.q
        return 0.0


@dataclass(frozen=True, slots=True)
class SearchConfig:
    simulations: int = 100
    c_puct: float = 1.5

    def __post_init__(self) -> None:
        if self.simulations < 1:
            raise ValueError("simulations must be positive")
        if self.c_puct < 0:
            raise ValueError("c_puct cannot be negative")


@dataclass(frozen=True, slots=True)
class RootNoiseConfig:
    """Opt-in root exploration noise, scaled across the legal moves."""

    fraction: float
    total_concentration: float = 10.0

    def __post_init__(self) -> None:
        if not 0 <= self.fraction <= 1:
            raise ValueError("noise fraction must be in [0, 1]")
        if self.total_concentration <= 0:
            raise ValueError("total concentration must be positive")


@dataclass(frozen=True, slots=True)
class PendingSimulation:
    """One selected leaf waiting for either a terminal or evaluator value.

    A caller must complete this simulation before selecting another one from
    the same root. Independent roots may wait together for batched inference.
    """

    root: Node
    path: tuple[Node, ...]
    leaf: Node
    position: GameState
    root_noise: RootNoiseConfig | None

    @property
    def needs_evaluation(self) -> bool:
        return self.position.outcome is None


class PUCTSearch:
    """Run fixed-budget PUCT without modifying the caller's game state."""

    def __init__(
        self,
        evaluator: Evaluator,
        config: SearchConfig = SearchConfig(),
        *,
        seed: int = 0,
    ) -> None:
        self.evaluator = evaluator
        self.config = config
        self.rng = np.random.default_rng(seed)

    def run(
        self, state: GameState, *, root_noise: RootNoiseConfig | None = None
    ) -> Node:
        root = Node(state=state.clone())

        for _ in range(self.config.simulations):
            self._simulate(root, root_noise)

        return root

    def run_for_time(
        self,
        state: GameState,
        time_limit_seconds: float,
        *,
        min_simulations: int = 2,
        clock: Callable[[], float] = perf_counter,
    ) -> Node:
        """Run complete simulations under a wall-clock budget without noise."""

        if time_limit_seconds <= 0:
            raise ValueError("time limit must be positive")
        if min_simulations < 1:
            raise ValueError("min_simulations must be positive")
        if state.outcome is not None:
            raise ValueError("cannot search a terminal state")

        root = Node(state=state.clone())
        deadline = clock() + time_limit_seconds
        last_simulation_seconds = 0.0
        while True:
            now = clock()
            remaining = deadline - now
            predicted_cost = 1.25 * last_simulation_seconds
            if root.visits >= min_simulations and (
                remaining <= 0 or predicted_cost >= remaining
            ):
                break

            simulation_started = clock()
            self._simulate(root, root_noise=None)
            last_simulation_seconds = max(0.0, clock() - simulation_started)

        return root

    def _simulate(
        self, root: Node, root_noise: RootNoiseConfig | None
    ) -> None:
        pending = self.begin_simulation(root, root_noise=root_noise)
        evaluation = (
            self.evaluator.evaluate(pending.position)
            if pending.needs_evaluation
            else None
        )
        self.complete_simulation(pending, evaluation)

    def begin_simulation(
        self, root: Node, *, root_noise: RootNoiseConfig | None = None
    ) -> PendingSimulation:
        """Select and materialize one leaf without evaluating or backing it up."""

        if root.state is None:
            raise ValueError("a search root must have a cached state")
        node = root
        path = [root]

        while node.expanded and node.children:
            parent = node
            move, node = select_child(parent, self.config.c_puct)
            if node.state is None:
                assert parent.state is not None
                node.state = parent.state.clone()
                node.state.make_move(move, validate=False)
            path.append(node)

        assert node.state is not None
        return PendingSimulation(
            root=root,
            path=tuple(path),
            leaf=node,
            position=node.state,
            root_noise=root_noise,
        )

    def complete_simulation(
        self,
        pending: PendingSimulation,
        evaluation: tuple[NDArray[np.float32], float] | None,
    ) -> None:
        """Expand and back up one leaf selected by :meth:`begin_simulation`."""

        position = pending.position
        if position.outcome is not None:
            if evaluation is not None:
                raise ValueError("a terminal leaf must not be evaluated")
            value = float(position.outcome)
        else:
            if evaluation is None:
                raise ValueError("a non-terminal leaf requires an evaluation")
            policy, value = evaluation
            self._expand(pending.leaf, position, policy)
            if pending.leaf is pending.root and pending.root_noise is not None:
                self._add_root_noise(pending.root, pending.root_noise)

        pending.leaf.evaluation = value
        backup(list(pending.path), value)

    def _expand(
        self,
        node: Node,
        state: GameState,
        raw_policy: NDArray[np.float32],
    ) -> None:
        if node.expanded:
            raise ValueError("a node cannot be expanded twice")

        policy = np.asarray(raw_policy, dtype=np.float64)
        if policy.shape != (ACTION_SIZE,):
            raise ValueError(f"policy must have shape ({ACTION_SIZE},)")
        if not np.all(np.isfinite(policy)) or np.any(policy < 0):
            raise ValueError("policy weights must be finite and non-negative")

        moves = state.legal_moves()
        if not moves:
            raise RuntimeError("non-terminal search node has no legal moves")
        weights = np.array(
            [policy[state.policy_index(move)] for move in moves], dtype=np.float64
        )
        total = float(weights.sum())
        if total <= 0:
            weights.fill(1.0 / len(moves))
        else:
            weights /= total

        node.children = {
            move: Node(
                prior=float(prior),
                network_prior=float(prior),
                parent=node,
            )
            for move, prior in zip(moves, weights, strict=True)
        }
        node.expanded = True

    def _add_root_noise(self, root: Node, config: RootNoiseConfig) -> None:
        if config.fraction == 0 or not root.children:
            return

        alpha = config.total_concentration / len(root.children)
        noise = self.rng.dirichlet([alpha] * len(root.children))
        for child, sample in zip(root.children.values(), noise, strict=True):
            child.prior = (
                (1 - config.fraction) * child.prior
                + config.fraction * float(sample)
            )


def select_child(parent: Node, c_puct: float) -> tuple[Move, Node]:
    """Select one child, maximizing for P1 and minimizing for P2."""

    if not parent.children:
        raise ValueError("cannot select from a node without children")

    if parent.state is None:
        raise ValueError("a selected parent must have a cached state")
    direction = 1.0 if parent.state.to_move == PLAYER_1 else -1.0
    scale = sqrt(parent.visits)

    def score(item: tuple[Move, Node]) -> float:
        _, child = item
        exploration = c_puct * child.prior * scale / (1 + child.visits)
        return direction * child.q + exploration

    return max(parent.children.items(), key=score)


def backup(path: list[Node], absolute_value: float) -> None:
    """Back up one Player 1 value without alternating its sign."""

    if not -1.0 <= absolute_value <= 1.0:
        raise ValueError("value must be in [-1, 1]")
    for node in path:
        node.visits += 1
        node.value_sum += absolute_value
        node.value_square_sum += absolute_value * absolute_value


def visit_policy(root: Node) -> dict[Move, float]:
    """Return normalized root visits, with priors as a zero-visit fallback."""

    total = sum(child.visits for child in root.children.values())
    if total:
        return {move: child.visits / total for move, child in root.children.items()}
    return {move: child.prior for move, child in root.children.items()}


def best_move(root: Node) -> Move:
    """Choose the most visited move with a deterministic move-order tie break."""

    if not root.children:
        raise ValueError("root has no legal moves")
    return max(
        root.children,
        key=lambda move: (root.children[move].visits, -move.source, -move.target),
    )


def greedy_leaf_value(root: Node) -> float:
    """Follow most-visited children and return the last leaf evaluation."""

    node = root
    while node.children:
        move = best_move(node)
        child = node.children[move]
        if child.visits == 0:
            break
        node = child
    return node.evaluation if node.evaluation is not None else node.q
