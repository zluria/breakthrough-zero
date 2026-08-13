"""Small Keras policy/value network and its PUCT evaluator adapter."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from .game import BOARD_SIZE, POLICY_PLANES, GameState


@dataclass(frozen=True, slots=True)
class NetworkConfig:
    """Readable architecture settings suitable for small controlled sweeps."""

    board_size: int = BOARD_SIZE
    architecture: Literal["cnn", "transformer"] = "cnn"
    channels: int = 64
    residual_blocks: int = 4
    value_hidden: int = 64
    attention_heads: int = 4
    l2_regularization: float = 1e-4

    def __post_init__(self) -> None:
        if not 2 <= self.board_size <= BOARD_SIZE:
            raise ValueError("network board size must be between 2 and 8")
        if self.architecture not in ("cnn", "transformer"):
            raise ValueError(f"unknown network architecture: {self.architecture}")
        if self.channels < 1 or self.residual_blocks < 0 or self.value_hidden < 1:
            raise ValueError("network dimensions must be positive")
        if self.attention_heads < 1 or self.channels % self.attention_heads:
            raise ValueError("attention heads must divide the channel width")
        if self.l2_regularization < 0:
            raise ValueError("L2 regularization cannot be negative")


def build_network(config: NetworkConfig = NetworkConfig()):
    """Build a compact policy/value model with an absolute Player-1 value."""

    tf = _tensorflow()
    keras = tf.keras
    regularizer = keras.regularizers.L2(config.l2_regularization)
    inputs = keras.Input(
        (config.board_size, config.board_size, 3), name="board"
    )

    if config.architecture == "cnn":
        trunk = _cnn_trunk(keras, inputs, config, regularizer)
        policy = keras.layers.Conv2D(
            3,
            1,
            padding="same",
            kernel_regularizer=regularizer,
            name="policy_planes",
        )(trunk)
        policy_logits = keras.layers.Reshape(
            (config.board_size * config.board_size * POLICY_PLANES,),
            name="policy_logits",
        )(policy)
        value = keras.layers.Conv2D(
            1,
            1,
            activation="relu",
            kernel_regularizer=regularizer,
            name="value_plane",
        )(trunk)
        value = keras.layers.Flatten(name="value_flatten")(value)
    else:
        trunk = _transformer_trunk(keras, inputs, config, regularizer)
        policy = keras.layers.Dense(
            POLICY_PLANES,
            kernel_regularizer=regularizer,
            name="policy_planes",
        )(trunk)
        policy_logits = keras.layers.Reshape(
            (config.board_size * config.board_size * POLICY_PLANES,),
            name="policy_logits",
        )(policy)
        value = keras.layers.GlobalAveragePooling1D(name="value_pool")(trunk)

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
        name=f"breakthrough_zero_{config.architecture}",
    )
    model.network_config = asdict(config)
    return model


def _cnn_trunk(keras, inputs, config: NetworkConfig, regularizer):
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
    return trunk


def _transformer_trunk(keras, inputs, config: NetworkConfig, regularizer):
    token_count = config.board_size * config.board_size
    tokens = keras.layers.Reshape((token_count, 3), name="board_tokens")(inputs)
    tokens = keras.layers.Dense(
        config.channels,
        kernel_regularizer=regularizer,
        name="token_embedding",
    )(tokens)
    # A learned position table is represented by an ordinary Embedding layer,
    # so the model remains save/load compatible without custom Keras objects.
    positions = keras.ops.arange(token_count)
    position_embedding = keras.layers.Embedding(
        token_count,
        config.channels,
        embeddings_regularizer=regularizer,
        name="position_embedding",
    )(positions)
    tokens = keras.layers.Add(name="add_position")([tokens, position_embedding])

    for block in range(config.residual_blocks):
        normalized = keras.layers.LayerNormalization(
            name=f"transformer_{block}_attention_norm"
        )(tokens)
        attention = keras.layers.MultiHeadAttention(
            num_heads=config.attention_heads,
            key_dim=config.channels // config.attention_heads,
            kernel_regularizer=regularizer,
            name=f"transformer_{block}_attention",
        )(normalized, normalized)
        tokens = keras.layers.Add(name=f"transformer_{block}_attention_add")(
            [tokens, attention]
        )
        normalized = keras.layers.LayerNormalization(
            name=f"transformer_{block}_mlp_norm"
        )(tokens)
        hidden = keras.layers.Dense(
            4 * config.channels,
            activation="gelu",
            kernel_regularizer=regularizer,
            name=f"transformer_{block}_mlp_expand",
        )(normalized)
        hidden = keras.layers.Dense(
            config.channels,
            kernel_regularizer=regularizer,
            name=f"transformer_{block}_mlp_project",
        )(hidden)
        tokens = keras.layers.Add(name=f"transformer_{block}_mlp_add")(
            [tokens, hidden]
        )
    return keras.layers.LayerNormalization(name="transformer_output_norm")(tokens)


def load_network(path: str | Path):
    """Load a model saved in Keras' native format."""

    return _tensorflow().keras.models.load_model(path)


def network_config(model: Any) -> NetworkConfig:
    """Recover the small set of architecture settings from a loaded model."""

    board_size = int(model.input_shape[1])
    layer_names = {layer.name for layer in model.layers}
    if "token_embedding" in layer_names:
        architecture = "transformer"
        channels = int(model.get_layer("token_embedding").units)
        residual_blocks = sum(
            name.startswith("transformer_") and name.endswith("_attention_add")
            for name in layer_names
        )
        attention_heads = int(model.get_layer("transformer_0_attention").num_heads)
        regularized_layer = model.get_layer("token_embedding")
    else:
        architecture = "cnn"
        channels = int(model.get_layer("stem").filters)
        residual_blocks = sum(
            name.startswith("residual_") and name.endswith("_add")
            for name in layer_names
        )
        attention_heads = 4
        regularized_layer = model.get_layer("stem")
    regularizer = regularized_layer.kernel_regularizer
    l2 = float(regularizer.l2) if regularizer is not None else 0.0
    return NetworkConfig(
        board_size=board_size,
        architecture=architecture,
        channels=channels,
        residual_blocks=residual_blocks,
        value_hidden=int(model.get_layer("value_hidden").units),
        attention_heads=attention_heads,
        l2_regularization=l2,
    )


class KerasEvaluator:
    """Adapt one Keras model to the evaluator protocol used by PUCT."""

    def __init__(self, model: Any) -> None:
        self.model = model
        shape = tuple(model.input_shape)
        if (
            len(shape) != 4
            or shape[1] is None
            or shape[1] != shape[2]
            or shape[3] != 3
            or not 2 <= int(shape[1]) <= BOARD_SIZE
        ):
            raise ValueError(f"model has an unsupported board input shape: {shape}")
        self.board_size = int(shape[1])
        self.action_size = self.board_size * self.board_size * POLICY_PLANES

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
        wrong_sizes = {
            state.rules.active_size
            for state in states
            if state.rules.active_size != self.board_size
        }
        if wrong_sizes:
            raise ValueError(
                f"model expects {self.board_size}x{self.board_size} states, "
                f"received board sizes {sorted(wrong_sizes)}"
            )
        boards = np.stack([state.encode() for state in states])
        outputs = self.model(boards, training=False)
        logits_batch = np.asarray(outputs["policy_logits"], dtype=np.float64)
        values = np.asarray(outputs["value"], dtype=np.float64)
        expected_logits = (len(states), self.action_size)
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
            policy = np.zeros(self.action_size, dtype=np.float32)
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
