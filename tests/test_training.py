from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.train_pretraining import (
    _apply_source_loss_fraction,
    _checkpoint_due,
    _limit_samples,
    _load_games,
    _prepare_weighted_source_mix,
)
from breakthrough_zero.game import MINI_RULES, GameState
from breakthrough_zero.data import (
    GameRecord,
    save_chunk,
    transform_position,
    value_target,
)
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

        self.assertEqual(batch.boards.shape, (len(samples), 5, 5, 3))
        self.assertEqual(
            batch.policies.shape, (len(samples), MINI_RULES.action_size)
        )
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

    def test_mixed_value_target_is_the_documented_half_and_half_target(self) -> None:
        position = self.game.positions[0]
        expected = 0.5 * (self.game.outcome + position.root_q)
        self.assertAlmostEqual(
            value_target(position, self.game.outcome, "mixed_z_q"), expected
        )

    def test_explicit_augmentation_cycle_uses_every_symmetry_once(self) -> None:
        sample = PositionSample(self.game.positions[0], self.game.outcome)
        batches = [
            make_training_batch(
                [sample],
                target="outcome",
                symmetry_indices=[index],
                augment=True,
            )
            for index in range(len(tuple(Symmetry)))
        ]
        expected = [
            transform_position(sample.position, symmetry).state.encode()
            for symmetry in Symmetry
        ]
        for batch, board in zip(batches, expected, strict=True):
            np.testing.assert_array_equal(batch.boards[0], board)

    def test_position_limit_is_exact_reproducible_and_fail_loud(self) -> None:
        samples = list(range(20))
        first = _limit_samples(samples, 7, seed=31)
        second = _limit_samples(samples, 7, seed=31)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 7)
        self.assertEqual(len(set(first)), 7)
        with self.assertRaises(ValueError):
            _limit_samples(samples, 21, seed=31)

    def test_checkpoint_schedule_always_preserves_the_stopping_epoch(self) -> None:
        self.assertFalse(_checkpoint_due(3, 4, False))
        self.assertTrue(_checkpoint_due(4, 4, False))
        self.assertTrue(_checkpoint_due(3, 4, True))

    def test_weighted_sources_have_the_requested_exact_loss_share(self) -> None:
        primary = samples_from_games([self.game])
        secondary = primary[:3]
        mixed, weights = _apply_source_loss_fraction(primary, secondary, 0.75)

        primary_total = sum(
            sample.loss_weight for sample in mixed[: len(primary)]
        )
        secondary_total = sum(
            sample.loss_weight for sample in mixed[len(primary) :]
        )
        self.assertAlmostEqual(
            primary_total / (primary_total + secondary_total), 0.75
        )
        self.assertGreater(weights["secondary_position_weight"], 0)
        batch = make_training_batch(mixed, target="mixed_z_q", augment=False)
        np.testing.assert_allclose(
            batch.sample_weights,
            [sample.loss_weight for sample in mixed],
        )

    def test_weighted_source_mix_splits_each_source_by_complete_game(self) -> None:
        primary = [
            GameRecord(self.game.positions, self.game.outcome, seed)
            for seed in range(10, 20)
        ]
        secondary = [
            GameRecord(self.game.positions, self.game.outcome, seed)
            for seed in range(30, 40)
        ]
        train, validation, report = _prepare_weighted_source_mix(
            primary,
            secondary,
            primary_fraction=0.75,
            validation_fraction=0.2,
            seed=41,
        )

        self.assertEqual(report["train_games"], 16)
        self.assertEqual(report["validation_games"], 4)
        self.assertEqual(
            len(train) + len(validation), 20 * len(self.game.positions)
        )
        for split in (train, validation):
            total = sum(sample.loss_weight for sample in split)
            primary_positions = report[
                "train" if split is train else "validation"
            ]["primary_positions"]
            self.assertAlmostEqual(
                sum(
                    sample.loss_weight for sample in split[:primary_positions]
                )
                / total,
                0.75,
            )

    def test_multiple_input_roots_reject_duplicate_chunks_and_seeds(self) -> None:
        second_game = GameRecord(
            positions=self.game.positions,
            outcome=self.game.outcome,
            seed=self.game.seed + 1,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left = root / "left"
            right = root / "right"
            duplicate = root / "duplicate"
            save_chunk(left / "chunk_00000.npz", (self.game,), metadata={})
            save_chunk(right / "chunk_00000.npz", (second_game,), metadata={})
            save_chunk(duplicate / "chunk_00000.npz", (self.game,), metadata={})

            games, inputs = _load_games([left, right])
            self.assertEqual(
                [game.seed for game in games],
                [self.game.seed, second_game.seed],
            )
            self.assertEqual(len(inputs), 2)
            with self.assertRaisesRegex(ValueError, "supplied more than once"):
                _load_games([left, left])
            with self.assertRaisesRegex(ValueError, "across input chunks"):
                _load_games([left, duplicate])


class _FixedSymmetryGenerator:
    def __init__(self, index: int) -> None:
        self.index = index

    def integers(self, stop: int) -> int:
        if not 0 <= self.index < stop:
            raise ValueError("fixed symmetry is outside the requested range")
        return self.index


if __name__ == "__main__":
    unittest.main()
