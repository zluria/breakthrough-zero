"""A small Keras learner with explicit, inspectable AlphaZero losses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .training import TrainingBatch


@dataclass(frozen=True, slots=True)
class LossWeights:
    policy: float = 1.0
    value: float = 1.0

    def __post_init__(self) -> None:
        if self.policy < 0 or self.value < 0:
            raise ValueError("loss weights cannot be negative")
        if self.policy + self.value == 0:
            raise ValueError("at least one loss weight must be positive")


class KerasLearner:
    """Own the optimizer and the two compiled batch operations."""

    def __init__(
        self,
        model: Any,
        *,
        learning_rate: float = 1e-3,
        loss_weights: LossWeights = LossWeights(),
    ) -> None:
        if learning_rate <= 0:
            raise ValueError("learning rate must be positive")
        self.tf = _tensorflow()
        self.model = model
        self.loss_weights = loss_weights
        self.optimizer = self.tf.keras.optimizers.Adam(learning_rate)
        self._compiled_train = self.tf.function(
            self._train_tensors, reduce_retracing=True
        )
        self._compiled_evaluate = self.tf.function(
            self._evaluate_tensors, reduce_retracing=True
        )

    def train_batch(self, batch: TrainingBatch) -> dict[str, float]:
        return _as_floats(self._compiled_train(*_batch_arrays(batch)))

    def evaluate_batch(self, batch: TrainingBatch) -> dict[str, float]:
        return _as_floats(self._compiled_evaluate(*_batch_arrays(batch)))

    def _train_tensors(self, boards, policies, legal_masks, values, sample_weights):
        with self.tf.GradientTape() as tape:
            outputs = self.model(boards, training=True)
            losses = self._losses(
                outputs, policies, legal_masks, values, sample_weights
            )
        gradients = tape.gradient(losses["total"], self.model.trainable_variables)
        pairs = [
            (gradient, variable)
            for gradient, variable in zip(
                gradients, self.model.trainable_variables, strict=True
            )
            if gradient is not None
        ]
        self.optimizer.apply_gradients(pairs)
        return losses

    def _evaluate_tensors(
        self, boards, policies, legal_masks, values, sample_weights
    ):
        outputs = self.model(boards, training=False)
        return self._losses(outputs, policies, legal_masks, values, sample_weights)

    def _losses(self, outputs, policies, legal_masks, values, sample_weights):
        tf = self.tf
        logits = outputs["policy_logits"]
        predicted_values = tf.squeeze(outputs["value"], axis=1)
        masks = tf.cast(legal_masks, tf.bool)
        masked_logits = tf.where(masks, logits, tf.cast(-1e9, logits.dtype))

        policy_per_sample = tf.nn.softmax_cross_entropy_with_logits(
            labels=policies, logits=masked_logits
        )
        value_per_sample = tf.square(predicted_values - values)
        denominator = tf.maximum(tf.reduce_sum(sample_weights), 1e-12)
        policy_loss = tf.reduce_sum(policy_per_sample * sample_weights) / denominator
        value_loss = tf.reduce_sum(value_per_sample * sample_weights) / denominator
        regularization = (
            tf.add_n(self.model.losses)
            if self.model.losses
            else tf.constant(0.0, dtype=policy_loss.dtype)
        )
        total = (
            self.loss_weights.policy * policy_loss
            + self.loss_weights.value * value_loss
            + regularization
        )

        predicted_actions = tf.argmax(masked_logits, axis=1)
        target_actions = tf.argmax(policies, axis=1)
        accuracy = tf.reduce_sum(
            tf.cast(predicted_actions == target_actions, tf.float32)
            * sample_weights
        ) / denominator
        unmasked_policy = tf.nn.softmax(logits, axis=1)
        illegal_mass_per_sample = tf.reduce_sum(
            tf.where(masks, tf.zeros_like(unmasked_policy), unmasked_policy), axis=1
        )
        illegal_mass = tf.reduce_sum(
            illegal_mass_per_sample * sample_weights
        ) / denominator
        value_mae = tf.reduce_sum(
            tf.abs(predicted_values - values) * sample_weights
        ) / denominator

        return {
            "total": total,
            "policy": policy_loss,
            "value": value_loss,
            "regularization": regularization,
            "policy_accuracy": accuracy,
            "illegal_mass": illegal_mass,
            "value_mae": value_mae,
        }


def _batch_arrays(batch: TrainingBatch):
    return (
        batch.boards,
        batch.policies,
        batch.legal_masks,
        batch.values,
        batch.sample_weights,
    )


def _as_floats(metrics) -> dict[str, float]:
    return {name: float(value.numpy()) for name, value in metrics.items()}


def _tensorflow():
    try:
        import tensorflow as tf
    except ImportError as error:  # pragma: no cover - exercised on the HPC
        raise ImportError("TensorFlow is required for Keras training") from error
    return tf
