from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from breakthrough_zero.game import ACTION_SIZE, MINI_RULES, STANDARD_RULES, GameState
from breakthrough_zero.learner import KerasLearner
from breakthrough_zero.network import (
    KerasEvaluator,
    NetworkConfig,
    build_network,
    load_network,
    network_config,
)
from breakthrough_zero.search import SearchConfig
from breakthrough_zero.selfplay import SelfPlayConfig, play_dummy_game
from breakthrough_zero.training import make_training_batch, samples_from_games

try:
    import tensorflow as tf
except ImportError:
    tf = None


@unittest.skipUnless(tf is not None, "TensorFlow is tested in the HPC environment")
class NetworkTests(unittest.TestCase):
    def test_shapes_evaluator_and_save_load(self) -> None:
        model = build_network(
            NetworkConfig(
                board_size=5, channels=8, residual_blocks=1, value_hidden=8
            )
        )
        state = GameState.initial(MINI_RULES)
        outputs = model(state.encode()[None, ...], training=False)
        self.assertEqual(
            tuple(outputs["policy_logits"].shape),
            (1, MINI_RULES.action_size),
        )
        self.assertEqual(tuple(outputs["value"].shape), (1, 1))

        policy, value = KerasEvaluator(model).evaluate(state)
        self.assertAlmostEqual(float(policy.sum()), 1.0, places=6)
        self.assertTrue(np.all(policy >= 0))
        self.assertTrue(np.all(policy[state.legal_action_indices()] > 0))
        self.assertTrue(-1 <= value <= 1)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.keras"
            model.save(path)
            loaded = load_network(path)
            loaded_outputs = loaded(state.encode()[None, ...], training=False)
            np.testing.assert_allclose(
                outputs["policy_logits"], loaded_outputs["policy_logits"], atol=1e-6
            )
            np.testing.assert_allclose(outputs["value"], loaded_outputs["value"], atol=1e-6)

    def test_batch_evaluation_matches_scalar_boundary(self) -> None:
        model = build_network(
            NetworkConfig(
                board_size=5, channels=8, residual_blocks=1, value_hidden=8
            )
        )
        evaluator = KerasEvaluator(model)
        states = [GameState.initial(MINI_RULES), GameState.initial(MINI_RULES)]
        states[1].make_move(states[1].legal_moves()[3])

        batched = evaluator.evaluate_batch(states)
        self.assertEqual(len(batched), len(states))
        for state, (batch_policy, batch_value) in zip(
            states, batched, strict=True
        ):
            scalar_policy, scalar_value = evaluator.evaluate(state)
            # cuDNN may choose a different convolution kernel for batch 1 and
            # batch 2.  The boundary should be numerically equivalent, while
            # legality and normalization below remain exact invariants.
            np.testing.assert_allclose(
                batch_policy, scalar_policy, atol=3e-5, rtol=3e-4
            )
            self.assertLessEqual(abs(batch_value - scalar_value), 5e-4)
            legal = state.legal_action_indices()
            self.assertLessEqual(
                abs(float(batch_policy[legal].sum()) - 1.0), 1e-6
            )
            illegal = np.ones(MINI_RULES.action_size, dtype=np.bool_)
            illegal[legal] = False
            self.assertTrue(np.all(batch_policy[illegal] == 0))

    def test_tiny_transformer_shapes_save_load_and_config(self) -> None:
        config = NetworkConfig(
            board_size=5,
            architecture="transformer",
            channels=16,
            residual_blocks=2,
            value_hidden=16,
            attention_heads=4,
        )
        model = build_network(config)
        state = GameState.initial(MINI_RULES)
        outputs = model(state.encode()[None, ...], training=False)
        self.assertEqual(tuple(outputs["policy_logits"].shape), (1, 75))
        self.assertEqual(tuple(outputs["value"].shape), (1, 1))
        self.assertEqual(network_config(model), config)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "transformer.keras"
            model.save(path)
            loaded = load_network(path)
            self.assertEqual(network_config(loaded), config)
            loaded_outputs = loaded(state.encode()[None, ...], training=False)
            np.testing.assert_allclose(
                outputs["policy_logits"], loaded_outputs["policy_logits"], atol=1e-6
            )
            np.testing.assert_allclose(outputs["value"], loaded_outputs["value"], atol=1e-6)

    def test_one_training_batch_has_finite_diagnostics(self) -> None:
        game = play_dummy_game(
            SelfPlayConfig(search=SearchConfig(simulations=4)),
            seed=7,
            initial_state=GameState.initial(MINI_RULES),
        )
        batch = make_training_batch(
            samples_from_games([game]), target="outcome", augment=False
        )
        model = build_network(
            NetworkConfig(
                board_size=5, channels=8, residual_blocks=1, value_hidden=8
            )
        )
        metrics = KerasLearner(model).train_batch(batch)
        self.assertEqual(
            set(metrics),
            {
                "total",
                "policy",
                "policy_target_entropy",
                "policy_kl",
                "legal_policy",
                "legal_policy_kl",
                "value",
                "regularization",
                "policy_accuracy",
                "legal_policy_accuracy",
                "illegal_mass",
                "value_mae",
            },
        )
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))

    def test_models_reject_the_other_ruleset_before_inference(self) -> None:
        mini = KerasEvaluator(
            build_network(
                NetworkConfig(
                    board_size=5,
                    channels=4,
                    residual_blocks=0,
                    value_hidden=4,
                )
            )
        )
        standard = KerasEvaluator(
            build_network(
                NetworkConfig(
                    board_size=8,
                    channels=4,
                    residual_blocks=0,
                    value_hidden=4,
                )
            )
        )
        with self.assertRaisesRegex(ValueError, "expects 5x5"):
            mini.evaluate(GameState.initial(STANDARD_RULES))
        with self.assertRaisesRegex(ValueError, "expects 8x8"):
            standard.evaluate(GameState.initial(MINI_RULES))

        outputs = standard.model(
            GameState.initial(STANDARD_RULES).encode()[None, ...], training=False
        )
        self.assertEqual(tuple(outputs["policy_logits"].shape), (1, ACTION_SIZE))

    def test_training_state_restores_adam_moments(self) -> None:
        game = play_dummy_game(
            SelfPlayConfig(search=SearchConfig(simulations=4)),
            seed=11,
            initial_state=GameState.initial(MINI_RULES),
        )
        batch = make_training_batch(
            samples_from_games([game]), target="outcome", augment=False
        )
        config = NetworkConfig(
            board_size=5, channels=4, residual_blocks=0, value_hidden=4
        )
        first = KerasLearner(build_network(config))
        first.train_batch(batch)

        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.keras"
            state_path = Path(directory) / "state" / "checkpoint"
            first.model.save(model_path)
            saved_state = first.save_training_state(state_path)

            restored = KerasLearner(load_network(model_path))
            restored.restore_training_state(saved_state)
            self.assertEqual(
                int(restored.optimizer.iterations), int(first.optimizer.iterations)
            )

            first.train_batch(batch)
            restored.train_batch(batch)
            for expected, actual in zip(
                first.model.trainable_variables,
                restored.model.trainable_variables,
                strict=True,
            ):
                np.testing.assert_allclose(expected, actual, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
