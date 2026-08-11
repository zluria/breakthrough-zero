"""Train a small Keras policy/value model from immutable MCTS chunks."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import random
import subprocess
from typing import Any

import numpy as np

from breakthrough_zero.data import Target, load_chunk, split_game_indices
from breakthrough_zero.learner import KerasLearner, LossWeights
from breakthrough_zero.network import NetworkConfig, build_network
from breakthrough_zero.training import make_training_batch, samples_from_games


TARGETS: tuple[Target, ...] = (
    "outcome",
    "soft_z",
    "a0c",
    "played_q",
    "greedy_backup",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="chunk file or directory")
    parser.add_argument("output", type=Path, help="new run directory")
    parser.add_argument("--target", choices=TARGETS, default="outcome")
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--residual-blocks", type=int, default=4)
    parser.add_argument("--value-hidden", type=int, default=64)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--policy-loss-weight", type=float, default=1.0)
    parser.add_argument("--value-loss-weight", type=float, default=1.0)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--max-train-positions", type=int)
    parser.add_argument("--max-validation-positions", type=int)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive")
    if args.max_train_positions is not None and args.max_train_positions < 1:
        parser.error("--max-train-positions must be positive")
    if (
        args.max_validation_positions is not None
        and args.max_validation_positions < 1
    ):
        parser.error("--max-validation-positions must be positive")
    if not 0 <= args.seed < 2**64:
        parser.error("--seed must fit in an unsigned 64-bit integer")
    return args


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"training output already exists: {args.output}")

    games, inputs = _load_games(args.input)
    rule_names = {position.state.rules.name for game in games for position in game.positions}
    if len(rule_names) != 1:
        raise ValueError("one training run cannot mix rulesets")
    train_indices, validation_indices = split_game_indices(
        len(games), args.validation_fraction, seed=args.seed
    )
    train_samples = samples_from_games([games[index] for index in train_indices])
    validation_samples = samples_from_games(
        [games[index] for index in validation_indices]
    )
    available_train_positions = len(train_samples)
    available_validation_positions = len(validation_samples)
    train_samples = _limit_samples(
        train_samples, args.max_train_positions, seed=args.seed + 1
    )
    validation_samples = _limit_samples(
        validation_samples, args.max_validation_positions, seed=args.seed + 2
    )

    _seed_everything(args.seed)
    network_config = NetworkConfig(
        channels=args.channels,
        residual_blocks=args.residual_blocks,
        value_hidden=args.value_hidden,
        l2_regularization=args.l2,
    )
    model = build_network(network_config)
    learner = KerasLearner(
        model,
        learning_rate=args.learning_rate,
        loss_weights=LossWeights(
            policy=args.policy_loss_weight, value=args.value_loss_weight
        ),
    )
    rng = np.random.default_rng(args.seed)
    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = _run_epoch(
            train_samples,
            learner.train_batch,
            target=args.target,
            batch_size=args.batch_size,
            rng=rng,
            augment=True,
            shuffle=True,
        )
        validation_metrics = _run_epoch(
            validation_samples,
            learner.evaluate_batch,
            target=args.target,
            batch_size=args.batch_size,
            rng=None,
            augment=False,
            shuffle=False,
        )
        record = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": validation_metrics,
        }
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)

    args.output.mkdir(parents=True)
    model_path = args.output / "model.keras"
    model.save(model_path)
    run = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "target": args.target,
        "network": asdict(network_config),
        "optimizer": {
            "name": "Adam",
            "learning_rate": args.learning_rate,
            "policy_loss_weight": args.policy_loss_weight,
            "value_loss_weight": args.value_loss_weight,
        },
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "validation_fraction": args.validation_fraction,
        "seed": args.seed,
        "rules": next(iter(rule_names)),
        "train_games": len(train_indices),
        "validation_games": len(validation_indices),
        "train_positions": len(train_samples),
        "validation_positions": len(validation_samples),
        "available_train_positions": available_train_positions,
        "available_validation_positions": available_validation_positions,
        "inputs": inputs,
        "history": history,
        "environment": _environment(),
    }
    (args.output / "run.json").write_text(
        json.dumps(run, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"status": "complete", "output": str(args.output.resolve())}))


def _load_games(root: Path):
    paths = [root] if root.is_file() else sorted(root.rglob("chunk_*.npz"))
    if not paths:
        raise ValueError(f"no self-play chunks found under {root}")
    games = []
    inputs = []
    for path in paths:
        chunk_games, manifest = load_chunk(path)
        games.extend(chunk_games)
        inputs.append({"path": str(path.resolve()), "sha256": manifest["sha256"]})
    if len(games) < 2:
        raise ValueError("training requires at least two complete games")
    return games, inputs


def _run_epoch(
    samples,
    operation,
    *,
    target: Target,
    batch_size: int,
    rng: np.random.Generator | None,
    augment: bool,
    shuffle: bool,
) -> dict[str, float]:
    order = np.arange(len(samples))
    if shuffle:
        assert rng is not None
        rng.shuffle(order)

    totals: dict[str, float] = {}
    total_weight = 0.0
    for start in range(0, len(order), batch_size):
        selected = [samples[index] for index in order[start : start + batch_size]]
        batch = make_training_batch(
            selected, target=target, rng=rng, augment=augment
        )
        metrics = operation(batch)
        weight = float(batch.sample_weights.sum())
        total_weight += weight
        for name, value in metrics.items():
            totals[name] = totals.get(name, 0.0) + value * weight
    return {name: value / total_weight for name, value in totals.items()}


def _limit_samples(samples, limit: int | None, *, seed: int):
    if limit is None:
        return samples
    if len(samples) < limit:
        raise ValueError(
            f"requested {limit} positions, but the split contains only {len(samples)}"
        )
    indices = np.random.default_rng(seed).choice(len(samples), size=limit, replace=False)
    return [samples[int(index)] for index in indices]


def _seed_everything(seed: int) -> None:
    # NumPy's legacy global generator and TensorFlow accept 32-bit seeds.  The
    # run still records the full 64-bit master seed, while the explicit
    # default_rng used for augmentation consumes it without truncation.
    library_seed = seed % 2**32
    random.seed(seed)
    np.random.seed(library_seed)
    import tensorflow as tf

    tf.keras.utils.set_random_seed(library_seed)


def _environment() -> dict[str, Any]:
    import tensorflow as tf

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "tensorflow": tf.__version__,
        "git_commit": _git_commit(),
        "gpu_devices": [device.name for device in tf.config.list_physical_devices("GPU")],
    }


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], text=True, capture_output=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


if __name__ == "__main__":
    main()
