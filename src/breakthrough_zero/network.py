"""Small Keras policy/value network and its PUCT evaluator adapter."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .game import ACTION_SIZE, BOARD_SIZE, GameState


@dataclass(frozen=True, slots=True)
class NetworkConfig:
    """Readable architecture settings suitable for small controlled sweeps."""

    channels: int = 64
    residual_blocks: int = 4
    value_hidden: int = 64
    l2_regularization: float = 1e-4

    def __post_init__(self) -> None:
        if self.channels < 1 or self.residual_blocks < 0 or self.value_hidden < 1:
            raise ValueError("network dimensions must be positive")
        if self.l2_regularization < 0:
            raise ValueError("L2 regularization cannot be negative")


def build_network(config: NetworkConfig = NetworkConfig()):
    """Build a compact residual CNN with policy logits and an absolute value."""

    tf = _tensorflow()
    keras = tf.keras
    regularizer = keras.regularizers.L2(config.l2_regularization)
    inputs = keras.Input((BOARD_SIZE, BOARD_SIZE, 3), name="board")

    trunk = keras.layers.Conv2D(
        config.channels,
        3,
        padding="same",
        activation="relu",
        kernel_initializer="he_normal",
        kernel_regularizer=regularizer,
        name="stem",
    )(inputs)
    for block in range(config.residual_blocks):
        residual = trunk
        trunk = keras.layers.Conv2D(
            config.channels,
            3,
            padding="same",
            activation="relu",
            kernel_initializer="he_normal",
            kernel_regularizer=regularizer,
            name=f"residual_{block}_conv_1",
        )(trunk)
        trunk = keras.layers.Conv2D(
            config.channels,
            3,
            padding="same",
            kernel_initializer="he_normal",
            kernel_regularizer=regularizer,
            name=f"residual_{block}_conv_2",
        )(trunk)
        trunk = keras.layers.Add(name=f"residual_{block}_add")([residual, trunk])
        trunk = keras.layers.Activation("relu", name=f"residual_{block}_relu")(trunk)

    policy = keras.layers.Conv2D(
        3,
        1,
        padding="same",
        kernel_regularizer=regularizer,
        name="policy_planes",
    )(trunk)
    policy_logits = keras.layers.Reshape(
        (ACTION_SIZE,), name="policy_logits"
    )(policy)

    value = keras.layers.Conv2D(
        1,
        1,
        activation="relu",
        kernel_regularizer=regularizer,
        name="value_plane",
    )(trunk)
    value = keras.layers.Flatten(name="value_flatten")(value)
    value = keras.layers.Dense(
        config.value_hidden,
        activation="relu",
        kernel_regularizer=regularizer,
        name="value_hidden",
    )(value)
    absolute_value = keras.layers.Dense(
        1, activation="tanh", name="absolute_value"
    )(value)

    model = keras.Model(
        inputs=inputs,
        outputs={"policy_logits": policy_logits, "value": absolute_value},
        name="breakthrough_zero",
    )
    model.network_config = asdict(config)
    return model


def load_network(path: str | Path):
    """Load a model saved in Keras' native format."""

    return _tensorflow().keras.models.load_model(path)


class KerasEvaluator:
    """Adapt one Keras model to the evaluator protocol used by PUCT."""

    def __init__(self, model: Any) -> None:
        self.model = model

    def evaluate(self, state: GameState) -> tuple[NDArray[np.float32], float]:
        """Evaluate one state through the same checked batch boundary."""

        return self.evaluate_batch((state,))[0]

    def evaluate_batch(
        self, states: Sequence[GameState]
    ) -> tuple[tuple[NDArray[np.float32], float], ...]:
        """Evaluate independent leaves in one model call.

        Legal masking stays here rather than in the search. This keeps scalar
        and future parallel self-play on exactly the same policy boundary.
        """

        if not states:
            return ()
        boards = np.stack([state.encode() for state in states])
        outputs = self.model(boards, training=False)
        logits_batch = np.asarray(outputs["policy_logits"], dtype=np.float64)
        values = np.asarray(outputs["value"], dtype=np.float64)
        expected_logits = (len(states), ACTION_SIZE)
        expected_values = (len(states), 1)
        if logits_batch.shape != expected_logits or not np.all(
            np.isfinite(logits_batch)
        ):
            raise RuntimeError("network returned invalid policy logits")
        if values.shape != expected_values or not np.all(np.isfinite(values)):
            raise RuntimeError("network returned invalid absolute values")
        if np.any(values < -1.00001) or np.any(values > 1.00001):
            raise RuntimeError("network returned an invalid absolute value")

        evaluations = []
        for state, logits, raw_value in zip(
            states, logits_batch, values[:, 0], strict=True
        ):
            legal_indices = state.legal_action_indices()
            legal_logits = logits[legal_indices]
            legal_weights = np.exp(legal_logits - legal_logits.max())
            policy = np.zeros(ACTION_SIZE, dtype=np.float32)
            policy[legal_indices] = legal_weights / legal_weights.sum()
            evaluations.append(
                (policy, float(np.clip(raw_value, -1.0, 1.0)))
            )
        return tuple(evaluations)


def _tensorflow():
    try:
        import tensorflow as tf
    except ImportError as error:  # pragma: no cover - exercised on the HPC
        raise ImportError(
            "TensorFlow is required for the Keras network; install the train extra"
        ) from error
    return tf
