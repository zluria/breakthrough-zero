"""Train a small Keras policy/value model from immutable MCTS chunks."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from math import isclose
from pathlib import Path
import platform
import random
import subprocess
from time import perf_counter
from typing import Any

import numpy as np

from breakthrough_zero.data import Target, load_chunk, split_game_indices
from breakthrough_zero.learner import KerasLearner, LossWeights
from breakthrough_zero.network import NetworkConfig, build_network, load_network
from breakthrough_zero.symmetry import Symmetry
from breakthrough_zero.training import make_training_batch, samples_from_games


TARGETS: tuple[Target, ...] = (
    "outcome",
    "soft_z",
    "mixed_z_q",
    "a0c",
    "played_q",
    "greedy_backup",
)


@dataclass(frozen=True, slots=True)
class EpochResult:
    metrics: dict[str, float]
    sample_count: int
    sample_weight: float
    completed: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="chunk file or directory")
    parser.add_argument("output", type=Path, help="new run directory")
    parser.add_argument(
        "--extra-input",
        action="append",
        default=[],
        type=Path,
        help="add another immutable chunk file or directory",
    )
    parser.add_argument(
        "--initial-model",
        type=Path,
        help="initialize from a saved model instead of random weights",
    )
    parser.add_argument("--target", choices=TARGETS, default="outcome")
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--residual-blocks", type=int, default=4)
    parser.add_argument("--value-hidden", type=int, default=64)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--max-training-seconds",
        type=float,
        help="wall-clock budget including validation and checkpointing",
    )
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
    if args.max_training_seconds is not None and args.max_training_seconds <= 0:
        parser.error("--max-training-seconds must be positive")
    if args.max_train_positions is not None and args.max_train_positions < 1:
        parser.error("--max-train-positions must be positive")
    if (
        args.max_validation_positions is not None
        and args.max_validation_positions < 1
    ):
        parser.error("--max-validation-positions must be positive")
    if not 0 <= args.seed < 2**64:
        parser.error("--seed must fit in an unsigned 64-bit integer")
    if args.initial_model is not None and not args.initial_model.is_file():
        parser.error(f"--initial-model does not exist: {args.initial_model}")
    return args


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"training output already exists: {args.output}")

    games, inputs = _load_games([args.input, *args.extra_input])
    rule_names = {position.state.rules.name for game in games for position in game.positions}
    if len(rule_names) != 1:
        raise ValueError("one training run cannot mix rulesets")
    board_size = games[0].positions[0].state.rules.active_size
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
        board_size=board_size,
        channels=args.channels,
        residual_blocks=args.residual_blocks,
        value_hidden=args.value_hidden,
        l2_regularization=args.l2,
    )
    if args.initial_model is None:
        model = build_network(network_config)
        initial_model = None
    else:
        model = load_network(args.initial_model)
        _check_network_config(model, network_config)
        initial_model = {
            "path": str(args.initial_model.resolve()),
            "sha256": _file_sha256(args.initial_model),
        }
    learner = KerasLearner(
        model,
        learning_rate=args.learning_rate,
        loss_weights=LossWeights(
            policy=args.policy_loss_weight, value=args.value_loss_weight
        ),
    )
    args.output.mkdir(parents=True)
    checkpoints = args.output / "checkpoints"
    checkpoints.mkdir()
    rng = np.random.default_rng(args.seed)
    history = []
    training_started = perf_counter()
    deadline = (
        training_started + args.max_training_seconds
        if args.max_training_seconds is not None
        else None
    )
    optimizer_examples = 0
    for epoch in range(1, args.epochs + 1):
        train_result = _run_epoch(
            train_samples,
            learner.train_batch,
            target=args.target,
            batch_size=args.batch_size,
            rng=rng,
            augment=True,
            shuffle=True,
            symmetry_cycle=epoch - 1,
            deadline=deadline,
        )
        optimizer_examples += train_result.sample_count
        validation_result = _run_epoch(
            validation_samples,
            learner.evaluate_batch,
            target=args.target,
            batch_size=args.batch_size,
            rng=None,
            augment=False,
            shuffle=False,
            symmetry_cycle=None,
        )
        checkpoint = checkpoints / f"epoch_{epoch:03d}.keras"
        model.save(checkpoint)
        record = {
            "epoch": epoch,
            "train": train_result.metrics,
            "validation": validation_result.metrics,
            "training_samples": train_result.sample_count,
            "training_sample_weight": train_result.sample_weight,
            "completed_epoch": train_result.completed,
            "elapsed_seconds": round(perf_counter() - training_started, 3),
            "checkpoint": {
                "path": str(checkpoint.resolve()),
                "sha256": _file_sha256(checkpoint),
            },
        }
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        if not train_result.completed or (
            deadline is not None and perf_counter() >= deadline
        ):
            break

    model_path = args.output / "model.keras"
    model.save(model_path)
    run = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "target": args.target,
        "network": asdict(network_config),
        "initial_model": initial_model,
        "optimizer": {
            "name": "Adam",
            "learning_rate": args.learning_rate,
            "policy_loss_weight": args.policy_loss_weight,
            "value_loss_weight": args.value_loss_weight,
        },
        "epochs": args.epochs,
        "completed_epochs": sum(item["completed_epoch"] for item in history),
        "max_training_seconds": args.max_training_seconds,
        "training_elapsed_seconds": round(perf_counter() - training_started, 3),
        "optimizer_examples": optimizer_examples,
        "batch_size": args.batch_size,
        "augmentation": "balanced four-epoch symmetry cycle",
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


def _load_games(roots: list[Path]):
    paths = []
    for root in roots:
        root_paths = [root] if root.is_file() else sorted(root.rglob("chunk_*.npz"))
        if not root_paths:
            raise ValueError(f"no self-play chunks found under {root}")
        paths.extend(root_paths)
    resolved_paths = [path.resolve() for path in paths]
    if len(resolved_paths) != len(set(resolved_paths)):
        raise ValueError("the same self-play chunk was supplied more than once")

    games = []
    inputs = []
    game_seeds: set[int] = set()
    for path in paths:
        chunk_games, manifest = load_chunk(path)
        chunk_seeds = [game.seed for game in chunk_games]
        if len(chunk_seeds) != len(set(chunk_seeds)):
            raise ValueError(f"duplicate game seed within input chunk {path}")
        duplicate_seeds = game_seeds.intersection(chunk_seeds)
        if duplicate_seeds:
            duplicate = min(duplicate_seeds)
            raise ValueError(f"duplicate game seed {duplicate} across input chunks")
        games.extend(chunk_games)
        game_seeds.update(chunk_seeds)
        inputs.append({"path": str(path.resolve()), "sha256": manifest["sha256"]})
    if len(games) < 2:
        raise ValueError("training requires at least two complete games")
    return games, inputs


def _check_network_config(model, expected: NetworkConfig) -> None:
    """Fail early if checkpoint architecture flags describe another model."""

    actual = NetworkConfig(
        board_size=int(model.input_shape[1]),
        channels=model.get_layer("stem").filters,
        residual_blocks=sum(
            layer.name.startswith("residual_") and layer.name.endswith("_add")
            for layer in model.layers
        ),
        value_hidden=model.get_layer("value_hidden").units,
        l2_regularization=float(model.get_layer("stem").kernel_regularizer.l2),
    )
    dimensions_match = (
        actual.board_size == expected.board_size
        and actual.channels == expected.channels
        and actual.residual_blocks == expected.residual_blocks
        and actual.value_hidden == expected.value_hidden
    )
    if not dimensions_match or not isclose(
        actual.l2_regularization,
        expected.l2_regularization,
        rel_tol=1e-6,
        abs_tol=1e-9,
    ):
        raise ValueError(
            f"initial model architecture {actual} does not match requested {expected}"
        )


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run_epoch(
    samples,
    operation,
    *,
    target: Target,
    batch_size: int,
    rng: np.random.Generator | None,
    augment: bool,
    shuffle: bool,
    symmetry_cycle: int | None = None,
    deadline: float | None = None,
) -> EpochResult:
    order = np.arange(len(samples))
    if shuffle:
        assert rng is not None
        rng.shuffle(order)

    totals: dict[str, float] = {}
    total_weight = 0.0
    sample_count = 0
    completed = True
    for start in range(0, len(order), batch_size):
        if total_weight > 0 and deadline is not None and perf_counter() >= deadline:
            completed = False
            break
        selected_indices = order[start : start + batch_size]
        selected = [samples[index] for index in selected_indices]
        symmetry_indices = (
            [
                (int(index) + symmetry_cycle) % len(tuple(Symmetry))
                for index in selected_indices
            ]
            if symmetry_cycle is not None
            else None
        )
        batch = make_training_batch(
            selected,
            target=target,
            symmetry_indices=symmetry_indices,
            rng=None if symmetry_indices is not None else rng,
            augment=augment,
        )
        metrics = operation(batch)
        weight = float(batch.sample_weights.sum())
        sample_count += len(selected)
        total_weight += weight
        for name, value in metrics.items():
            totals[name] = totals.get(name, 0.0) + value * weight
    return EpochResult(
        metrics={name: value / total_weight for name, value in totals.items()},
        sample_count=sample_count,
        sample_weight=total_weight,
        completed=completed,
    )


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
