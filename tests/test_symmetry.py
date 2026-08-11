from __future__ import annotations

import random
import unittest

import numpy as np

from breakthrough_zero.evaluators import SymmetryEnsembleEvaluator
from breakthrough_zero.game import ACTION_SIZE, MINI_RULES, PLAYER_1, GameState
from breakthrough_zero.symmetry import (
    Symmetry,
    transform_move,
    transform_outcome,
    transform_state,
)


class AsymmetricBatchEvaluator:
    """A deterministic evaluator that deliberately violates every symmetry."""

    def evaluate(self, state: GameState):
        return self.evaluate_batch((state,))[0]

    def evaluate_batch(self, states):
        results = []
        for state in states:
            policy = np.zeros(ACTION_SIZE, dtype=np.float32)
            for move in state.legal_moves():
                policy[state.policy_index(move)] = 1 + move.source + move.target
            policy /= policy.sum()
            value = 0.4 if state.to_move == PLAYER_1 else -0.1
            results.append((policy, value))
        return tuple(results)


def reachable_states(
    seed: int, count: int, initial: GameState | None = None
) -> list[GameState]:
    rng = random.Random(seed)
    state = initial.clone() if initial is not None else GameState()
    states = [state.clone()]
    while len(states) < count and state.outcome is None:
        state.make_move(rng.choice(state.legal_moves()))
        states.append(state.clone())
    return states


class SymmetryTests(unittest.TestCase):
    def test_ensemble_is_exactly_consistent_under_all_symmetries(self) -> None:
        state = reachable_states(seed=31, count=12)[-1]
        evaluator = SymmetryEnsembleEvaluator(AsymmetricBatchEvaluator())
        policy, value = evaluator.evaluate(state)

        for symmetry in Symmetry:
            transformed = transform_state(state, symmetry)
            transformed_policy, transformed_value = evaluator.evaluate(transformed)
            expected_value = -value if symmetry.swap_players else value
            self.assertAlmostEqual(transformed_value, expected_value, places=7)
            for move in state.legal_moves():
                mapped_move = transform_move(move, symmetry, state.rules)
                self.assertAlmostEqual(
                    float(policy[state.policy_index(move)]),
                    float(
                        transformed_policy[
                            transformed.policy_index(mapped_move)
                        ]
                    ),
                    places=7,
                )
    def test_all_four_symmetries_preserve_rules_and_policy_mapping(self) -> None:
        starts = (GameState(), GameState.initial(MINI_RULES))
        for initial in starts:
            for state in reachable_states(seed=11, count=40, initial=initial):
                legal = state.legal_moves()
                for symmetry in Symmetry:
                    transformed = transform_state(state, symmetry)
                    expected_moves = {
                        transform_move(move, symmetry, state.rules) for move in legal
                    }
                    self.assertEqual(set(transformed.legal_moves()), expected_moves)

                    expected_indices = {
                        transformed.policy_index(
                            transform_move(move, symmetry, state.rules)
                        )
                        for move in legal
                    }
                    self.assertEqual(
                        set(transformed.legal_action_indices()), expected_indices
                    )

    def test_every_symmetry_is_its_own_inverse(self) -> None:
        starts = (GameState(), GameState.initial(MINI_RULES))
        for initial in starts:
            for state in reachable_states(seed=19, count=20, initial=initial):
                for symmetry in Symmetry:
                    self.assertEqual(
                        transform_state(transform_state(state, symmetry), symmetry),
                        state,
                    )
                    for move in state.legal_moves():
                        transformed = transform_move(move, symmetry, state.rules)
                        self.assertEqual(
                            transform_move(transformed, symmetry, state.rules),
                            move,
                        )

    def test_swapping_players_negates_absolute_outcome(self) -> None:
        self.assertEqual(
            transform_outcome(PLAYER_1, Symmetry.SWAP_PLAYERS), -PLAYER_1
        )
        self.assertEqual(
            transform_outcome(PLAYER_1, Symmetry.MIRROR_LEFT_RIGHT), PLAYER_1
        )
        self.assertIsNone(transform_outcome(None, Symmetry.SWAP_AND_MIRROR))

    def test_four_absolute_symmetries_remain_four_absolute_examples(self) -> None:
        state = reachable_states(seed=23, count=10)[-1]
        encodings = [
            transform_state(state, symmetry).encode() for symmetry in Symmetry
        ]
        unique = {encoding.tobytes() for encoding in encodings}
        self.assertEqual(len(unique), 4)

        # Ignoring the absolute-player plane leaves only identity and mirror.
        spatial = {encoding[:, :, :2].tobytes() for encoding in encodings}
        self.assertEqual(len(spatial), 2)


if __name__ == "__main__":
    unittest.main()
