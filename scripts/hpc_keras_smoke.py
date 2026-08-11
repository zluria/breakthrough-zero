"""Fail-fast TensorFlow/Keras GPU smoke test for one allocated Slurm GPU."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from time import perf_counter

import numpy as np
import tensorflow as tf


def build_model() -> tf.keras.Model:
    """Build a deliberately tiny two-head CNN; this is not the final model."""

    inputs = tf.keras.Input((8, 8, 3), name="position")
    trunk = tf.keras.layers.Conv2D(
        16, 3, padding="same", use_bias=False, name="trunk_conv"
    )(inputs)
    trunk = tf.keras.layers.BatchNormalization(name="trunk_norm")(trunk)
    trunk = tf.keras.layers.Activation("relu", name="trunk_relu")(trunk)

    policy = tf.keras.layers.Conv2D(4, 1, activation="relu", name="policy_conv")(
        trunk
    )
    policy = tf.keras.layers.Flatten()(policy)
    policy = tf.keras.layers.Dense(192, activation="softmax", name="policy")(
        policy
    )

    value = tf.keras.layers.Conv2D(1, 1, activation="relu", name="value_conv")(
        trunk
    )
    value = tf.keras.layers.Flatten()(value)
    value = tf.keras.layers.Dense(32, activation="relu", name="value_hidden")(
        value
    )
    value = tf.keras.layers.Dense(1, activation="tanh", name="value")(value)
    return tf.keras.Model(inputs=inputs, outputs={"policy": policy, "value": value})


def main() -> None:
    started = perf_counter()
    tf.keras.utils.set_random_seed(20260811)
    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) != 1:
        raise RuntimeError(f"expected exactly one allocated GPU, found {gpus}")
    tf.config.experimental.set_memory_growth(gpus[0], True)

    with tf.device("/GPU:0"):
        product = tf.linalg.matmul(tf.ones((64, 64)), tf.ones((64, 64)))
    if "GPU" not in product.device.upper():
        raise RuntimeError(f"explicit matrix multiplication used {product.device}")

    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss={
            "policy": tf.keras.losses.CategoricalCrossentropy(),
            "value": tf.keras.losses.MeanSquaredError(),
        },
    )
    rng = np.random.default_rng(20260811)
    positions = rng.integers(0, 2, size=(32, 8, 8, 3)).astype(np.float32)
    policy_targets = np.eye(192, dtype=np.float32)[rng.integers(0, 192, 32)]
    value_targets = rng.choice((-1.0, 1.0), size=(32, 1)).astype(np.float32)

    before = [weight.numpy().copy() for weight in model.trainable_weights]
    losses = model.train_on_batch(
        positions,
        {"policy": policy_targets, "value": value_targets},
        return_dict=True,
    )
    if not all(np.isfinite(float(loss)) for loss in losses.values()):
        raise RuntimeError(f"training produced non-finite losses: {losses}")
    max_weight_change = max(
        float(np.max(np.abs(old - new.numpy())))
        for old, new in zip(before, model.trainable_weights)
    )
    if max_weight_change == 0:
        raise RuntimeError("the optimizer did not change any trainable weight")

    predictions = model.predict(positions[:4], verbose=0)
    temporary_root = os.environ.get("SLURM_TMPDIR")
    with tempfile.TemporaryDirectory(prefix="btz-smoke-", dir=temporary_root) as tmp:
        model_path = Path(tmp) / "smoke.keras"
        model.save(model_path)
        restored = tf.keras.models.load_model(model_path, compile=False)
        restored_predictions = restored.predict(positions[:4], verbose=0)
        for head in ("policy", "value"):
            np.testing.assert_allclose(
                predictions[head], restored_predictions[head], rtol=1e-5, atol=1e-6
            )

    print(
        json.dumps(
            {
                "status": "pass",
                "tensorflow": tf.__version__,
                "cuda_build": tf.test.is_built_with_cuda(),
                "gpu": gpus[0].name,
                "matmul_device": product.device,
                "losses": {name: float(value) for name, value in losses.items()},
                "max_weight_change": max_weight_change,
                "elapsed_seconds": round(perf_counter() - started, 3),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
