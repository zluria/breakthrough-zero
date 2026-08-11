"""Simple evaluators used before a neural network is trusted."""

from __future__ import annotations

import random
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from .game import GameState
from .search import BatchEvaluator
from .symmetry import Symmetry, transform_move, transform_state


class RandomRolloutEvaluator:
    """Uniform policy weights and an optional tactical random rollout."""

    def __init__(self, seed: int = 0, *, prefer_tactical: bool = False) -> None:
        self.rng = random.Random(seed)
        self.prefer_tactical = prefer_tactical

    def evaluate(self, state: GameState) -> tuple[NDArray[np.float32], float]:
        rollout = state.clone()
        while rollout.outcome is None:
            move = rollout.random_legal_move(
                self.rng, prefer_tactical=self.prefer_tactical
            )
            rollout.make_move(move, validate=False)

        policy = np.ones(state.rules.action_size, dtype=np.float32)
        return policy, float(rollout.outcome)


class SymmetryEnsembleEvaluator:
    """Average predictions over all four exact game symmetries.

    The wrapped evaluator sees one flat batch. Policies are mapped back to the
    caller's move coordinates; values are negated only for player-swapping
    transforms. The result is symmetric by construction without changing PUCT.
    """

    def __init__(self, base: BatchEvaluator) -> None:
        self.base = base

    def evaluate(self, state: GameState) -> tuple[NDArray[np.float32], float]:
        return self.evaluate_batch((state,))[0]

    def evaluate_batch(
        self, states: Sequence[GameState]
    ) -> tuple[tuple[NDArray[np.float32], float], ...]:
        if not states:
            return ()
        symmetries = tuple(Symmetry)
        groups = [
            tuple(transform_state(state, symmetry) for symmetry in symmetries)
            for state in states
        ]
        transformed_states = [state for group in groups for state in group]
        raw = self.base.evaluate_batch(transformed_states)
        if len(raw) != len(transformed_states):
            raise RuntimeError("base evaluator returned the wrong result count")

        results = []
        cursor = 0
        for state, group in zip(states, groups, strict=True):
            policy = np.zeros(state.rules.action_size, dtype=np.float64)
            value = 0.0
            moves = state.legal_moves()
            for symmetry, transformed in zip(symmetries, group, strict=True):
                transformed_policy, transformed_value = raw[cursor]
                cursor += 1
                sign = -1.0 if symmetry.swap_players else 1.0
                value += sign * transformed_value / len(symmetries)
                for move in moves:
                    transformed_move = transform_move(
                        move, symmetry, state.rules
                    )
                    policy[state.policy_index(move)] += (
                        transformed_policy[
                            transformed.policy_index(transformed_move)
                        ]
                        / len(symmetries)
                    )
            if not np.isclose(policy.sum(), 1.0, atol=1e-6):
                raise RuntimeError("symmetry-averaged policy is not normalized")
            results.append(
                (policy.astype(np.float32), float(np.clip(value, -1.0, 1.0)))
            )
        return tuple(results)


class HeadAblationEvaluator:
    """Replace either learned head while preserving the batch interface."""

    def __init__(
        self,
        base: BatchEvaluator,
        *,
        uniform_policy: bool = False,
        zero_value: bool = False,
    ) -> None:
        if not uniform_policy and not zero_value:
            raise ValueError("an ablation must replace at least one head")
        self.base = base
        self.uniform_policy = uniform_policy
        self.zero_value = zero_value

    def evaluate(self, state: GameState) -> tuple[NDArray[np.float32], float]:
        return self.evaluate_batch((state,))[0]

    def evaluate_batch(
        self, states: Sequence[GameState]
    ) -> tuple[tuple[NDArray[np.float32], float], ...]:
        raw = self.base.evaluate_batch(states)
        if len(raw) != len(states):
            raise RuntimeError("base evaluator returned the wrong result count")
        results = []
        for state, (learned_policy, learned_value) in zip(
            states, raw, strict=True
        ):
            if self.uniform_policy:
                legal = state.legal_action_indices()
                policy = np.zeros(state.rules.action_size, dtype=np.float32)
                policy[legal] = 1.0 / len(legal)
            else:
                policy = learned_policy
            value = 0.0 if self.zero_value else learned_value
            results.append((policy, value))
        return tuple(results)
