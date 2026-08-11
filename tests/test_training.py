from __future__ import annotations

import unittest

import numpy as np

from breakthrough_zero.game import ACTION_SIZE, MINI_RULES, GameState
from breakthrough_zero.data import transform_position
from breakthrough_zero.search import SearchConfig
from breakthrough_zero.selfplay import SelfPlayConfig, play_dummy_game
from breakthrough_zero.symmetry import Symmetry, transform_outcome
from breakthrough_zero.training import (
    PositionSample,
    make_training_batch,
    samples_from_games,
)


class TrainingDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        config = SelfPlayConfig(
            search=SearchConfig(simulations=8), sample_until_ply=4
        )
        cls.game = play_dummy_game(
            config,
            seed=20260812,
            initial_state=GameState.initial(MINI_RULES),
        )

    def test_identity_batch_has_exact_policy_and_legal_masks(self) -> None:
        samples = samples_from_games([self.game])
        batch = make_training_batch(samples, target="outcome", augment=False)

        self.assertEqual(batch.policies.shape, (len(samples), ACTION_SIZE))
        np.testing.assert_allclose(batch.policies.sum(axis=1), 1.0, atol=1e-6)
        self.assertTrue(np.all(batch.policies[~batch.legal_masks] == 0))
        for row, sample in enumerate(samples):
            self.assertEqual(
                int(batch.legal_masks[row].sum()), len(sample.position.actions)
            )
        np.testing.assert_array_equal(batch.values, self.game.outcome)

    def test_each_symmetry_transforms_absolute_value_and_policy(self) -> None:
        sample = PositionSample(self.game.positions[0], self.game.outcome)
        symmetries = tuple(Symmetry)
        for wanted_index, symmetry in enumerate(symmetries):
            rng = _FixedSymmetryGenerator(wanted_index)
            batch = make_training_batch(
                [sample], target="outcome", rng=rng, augment=True
            )
            expected_outcome = transform_outcome(self.game.outcome, symmetry)
            self.assertEqual(float(batch.values[0]), expected_outcome)
            self.assertEqual(
                int(batch.legal_masks[0].sum()), len(sample.position.actions)
            )
            self.assertAlmostEqual(float(batch.policies[0].sum()), 1.0, places=6)
            transformed = transform_position(sample.position, symmetry)
            total_visits = sum(action.visits for action in transformed.actions)
            for action in transformed.actions:
                policy_index = transformed.state.policy_index(action.move)
                expected = action.visits / total_visits
                self.assertAlmostEqual(
                    float(batch.policies[0, policy_index]), expected, places=6
                )

    def test_random_augmentation_is_reproducible(self) -> None:
        samples = samples_from_games([self.game])
        first = make_training_batch(
            samples, target="soft_z", rng=np.random.default_rng(9), augment=True
        )
        second = make_training_batch(
            samples, target="soft_z", rng=np.random.default_rng(9), augment=True
        )
        np.testing.assert_array_equal(first.boards, second.boards)
        np.testing.assert_array_equal(first.policies, second.policies)
        np.testing.assert_array_equal(first.values, second.values)


class _FixedSymmetryGenerator:
    def __init__(self, index: int) -> None:
        self.index = index

    def integers(self, stop: int) -> int:
        if not 0 <= self.index < stop:
            raise ValueError("fixed symmetry is outside the requested range")
        return self.index


if __name__ == "__main__":
    unittest.main()
