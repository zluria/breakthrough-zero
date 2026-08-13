#!/usr/bin/env python3
"""Update the current actor from bounded historical and fresh replay."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from hashlib import sha256
import json
from math import isclose
import os
from pathlib import Path
import random
import tempfile
from time import perf_counter
from typing import Any, Sequence

import numpy as np

from breakthrough_zero.data import Target, load_chunk
from breakthrough_zero.learner import KerasLearner, LossWeights
from breakthrough_zero.network import (
    NetworkConfig,
    build_network,
    load_network,
    network_config as read_network_config,
)
from breakthrough_zero.replay import ReplaySampler, phase_name, step_limit_for_replay
from breakthrough_zero.training import (
    PositionSample,
    make_training_batch,
    samples_from_games,
)


TARGETS: tuple[Target, ...] = (
    "outcome",
    "soft_z",
    "mixed_z_q",
    "a0c",
    "played_q",
    "greedy_backup",
)


@dataclass(frozen=True, slots=True)
class SourceSplit:
    name: str
    train: tuple[PositionSample, ...]
    validation: tuple[PositionSample, ...]
    train_games: int
    validation_games: int
    inputs: tuple[dict[str, str], ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("historical_input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--fresh-input",
        action="append",
        default=[],
        type=Path,
        help="newest self-play chunks; may be supplied more than once",
    )
    parser.add_argument(
        "--historical-fraction",
        type=float,
        default=0.25,
        help="exact fraction of each two-source batch drawn from history",
    )
    parser.add_argument("--initial-model", type=Path)
    parser.add_argument(
        "--initial-training-state",
        type=Path,
        help="TensorFlow checkpoint prefix paired with --initial-model",
    )
    parser.add_argument("--target", choices=TARGETS, default="mixed_z_q")
    parser.add_argument("--architecture", choices=("cnn", "transformer"), default="cnn")
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--residual-blocks", type=int, default=3)
    parser.add_argument("--value-hidden", type=int, default=64)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--policy-loss-weight", type=float, default=1.0)
    parser.add_argument("--value-loss-weight", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=100_000)
    parser.add_argument("--max-training-seconds", type=float)
    parser.add_argument(
        "--max-replay-consumption",
        type=float,
        default=4.0,
        help="maximum optimizer examples divided by newest training positions",
    )
    parser.add_argument("--evaluate-every", type=int, default=50)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--validation-batch-size", type=int, default=512)
    parser.add_argument(
        "--split-seed",
        type=int,
        default=0,
        help="fixed game-ID split seed; independent of optimizer randomness",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--surprise-strength",
        type=float,
        default=0.0,
        help="multiply training weights by normalized 1 + strength * KL(visits || prior)",
    )
    parser.add_argument("--surprise-cap", type=float, default=3.0)
    args = parser.parse_args()

    positive = (
        args.batch_size,
        args.max_steps,
        args.evaluate_every,
        args.validation_batch_size,
    )
    if any(value < 1 for value in positive):
        parser.error("batch sizes, max steps, and evaluation interval must be positive")
    if not 0 < args.validation_fraction < 1:
        parser.error("--validation-fraction must be between zero and one")
    if not 0 <= args.historical_fraction <= 1:
        parser.error("--historical-fraction must be in [0, 1]")
    if args.fresh_input and not 0 < args.historical_fraction < 1:
        parser.error("two-source replay needs a historical fraction between zero and one")
    if args.max_training_seconds is not None and args.max_training_seconds <= 0:
        parser.error("--max-training-seconds must be positive")
    if args.max_replay_consumption <= 0:
        parser.error("--max-replay-consumption must be positive")
    if args.surprise_strength < 0 or args.surprise_cap < 1:
        parser.error("surprise strength must be non-negative and cap at least one")
    if not 0 <= args.seed < 2**64 or not 0 <= args.split_seed < 2**64:
        parser.error("seeds must fit in an unsigned 64-bit integer")
    if args.initial_training_state is not None and args.initial_model is None:
        parser.error("--initial-training-state requires --initial-model")
    if args.initial_model is not None and not args.initial_model.is_file():
        parser.error(f"initial model does not exist: {args.initial_model}")
    return args


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"training output already exists: {args.output}")

    historical_games, historical_inputs = _load_games([args.historical_input])
    historical = _split_source(
        "historical",
        historical_games,
        historical_inputs,
        validation_fraction=args.validation_fraction,
        seed=args.split_seed,
    )
    fresh = None
    if args.fresh_input:
        fresh_games, fresh_inputs = _load_games(args.fresh_input)
        _reject_duplicate_seeds(historical_games, fresh_games)
        fresh = _split_source(
            "fresh",
            fresh_games,
            fresh_inputs,
            validation_fraction=args.validation_fraction,
            seed=args.split_seed + 1,
        )
    historical = _weight_training_surprise(
        historical, strength=args.surprise_strength, cap=args.surprise_cap
    )
    if fresh is not None:
        fresh = _weight_training_surprise(
            fresh, strength=args.surprise_strength, cap=args.surprise_cap
        )
    _check_rules(historical, fresh)

    _seed_everything(args.seed)
    board_size = historical.train[0].position.state.rules.active_size
    network_config = NetworkConfig(
        board_size=board_size,
        architecture=args.architecture,
        channels=args.channels,
        residual_blocks=args.residual_blocks,
        value_hidden=args.value_hidden,
        attention_heads=args.attention_heads,
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
            policy=args.policy_loss_weight,
            value=args.value_loss_weight,
        ),
    )
    if args.initial_training_state is not None:
        learner.restore_training_state(args.initial_training_state)
    initial_optimizer_step = int(learner.optimizer.iterations.numpy())

    sampler = ReplaySampler(
        historical.train,
        fresh.train if fresh is not None else None,
        batch_size=args.batch_size,
        historical_fraction=args.historical_fraction,
        seed=args.seed + 2,
    )
    if fresh is None:
        newest_positions = len(historical.train)
    else:
        newest_games, newest_inputs = _load_games([args.fresh_input[-1]])
        newest = _split_source(
            "newest",
            newest_games,
            newest_inputs,
            validation_fraction=args.validation_fraction,
            seed=args.split_seed + 1,
        )
        newest_positions = len(newest.train)
    replay_step_limit = step_limit_for_replay(
        batch_size=args.batch_size,
        newest_positions=newest_positions,
        maximum_consumption=args.max_replay_consumption,
    )
    step_limit = min(args.max_steps, replay_step_limit)

    args.output.mkdir(parents=True)
    (args.output / "checkpoints").mkdir()
    (args.output / "states").mkdir()
    started = perf_counter()
    deadline = (
        started + args.max_training_seconds
        if args.max_training_seconds is not None
        else None
    )

    history: list[dict[str, Any]] = []
    baseline_validation = _validation_report(
        learner,
        historical,
        fresh,
        target=args.target,
        batch_size=args.validation_batch_size,
        historical_fraction=sampler.historical_count / args.batch_size,
    )
    baseline_state = learner.save_training_state(args.output / "states/step_000000/state")
    if initial_model is None:
        baseline_model = _save_model(model, args.output / "checkpoints/step_000000.keras")
    else:
        baseline_model = initial_model
    baseline = {
        "step": 0,
        "optimizer_step": initial_optimizer_step,
        "elapsed_seconds": round(perf_counter() - started, 3),
        "train": None,
        "validation": baseline_validation,
        "model": baseline_model,
        "training_state": str(Path(baseline_state).resolve()),
        "replay_consumption": 0.0,
    }
    history.append(baseline)
    print(json.dumps(baseline, sort_keys=True), flush=True)
    interval_totals: dict[str, float] = {}
    interval_weight = 0.0
    stop_reason = "replay_cap" if step_limit == replay_step_limit else "step_cap"
    for step in range(1, step_limit + 1):
        if deadline is not None and perf_counter() >= deadline:
            stop_reason = "wall_clock"
            break
        draw = sampler.draw()
        batch = make_training_batch(
            draw.samples,
            target=args.target,
            symmetry_indices=draw.symmetry_indices,
            augment=True,
        )
        metrics = learner.train_batch(batch)
        weight = float(np.sum(batch.sample_weights))
        interval_weight += weight
        for name, value in metrics.items():
            interval_totals[name] = interval_totals.get(name, 0.0) + value * weight

        time_expired = deadline is not None and perf_counter() >= deadline
        due = step % args.evaluate_every == 0 or step == step_limit or time_expired
        if not due:
            continue
        validation = _validation_report(
            learner,
            historical,
            fresh,
            target=args.target,
            batch_size=args.validation_batch_size,
            historical_fraction=sampler.historical_count / args.batch_size,
        )
        model_info = _save_model(
            model, args.output / f"checkpoints/step_{step:06d}.keras"
        )
        state_prefix = learner.save_training_state(
            args.output / f"states/step_{step:06d}/state"
        )
        record = {
            "step": step,
            "optimizer_step": int(learner.optimizer.iterations.numpy()),
            "elapsed_seconds": round(perf_counter() - started, 3),
            "train": {
                name: value / interval_weight
                for name, value in interval_totals.items()
            },
            "validation": validation,
            "model": model_info,
            "training_state": str(Path(state_prefix).resolve()),
            "replay_consumption": sampler.replay_consumption,
            "source_presentations": {
                "historical": sampler.historical.presentations,
                "fresh": sampler.fresh.presentations if sampler.fresh else 0,
            },
        }
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        interval_totals, interval_weight = {}, 0.0
        if time_expired:
            stop_reason = "wall_clock"
            break

    completed_steps = int(history[-1]["step"])
    if completed_steps == 0:
        raise RuntimeError("training stopped before completing one optimizer step")

    actor_manifest = _actor_manifest(history)
    actor = actor_manifest["models"]["latest"]
    best_validation = actor_manifest["models"]["best_validation"]
    run = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(perf_counter() - started, 3),
        "stop_reason": stop_reason,
        "completed_steps": completed_steps,
        "step_limit": step_limit,
        "replay_step_limit": replay_step_limit,
        "replay_consumption": sampler.replay_consumption,
        "target": args.target,
        "surprise_weighting": {
            "strength": args.surprise_strength,
            "cap": args.surprise_cap,
        },
        "network": asdict(network_config),
        "initial_model": initial_model,
        "initial_training_state": (
            str(args.initial_training_state.resolve())
            if args.initial_training_state is not None
            else None
        ),
        "optimizer": {
            "name": "Adam",
            "learning_rate": args.learning_rate,
            "state_policy": "restored when supplied; saved at every evaluation",
            "initial_step": initial_optimizer_step,
            "final_step": int(learner.optimizer.iterations.numpy()),
        },
        "replay": {
            "batch_size": args.batch_size,
            "historical_per_batch": sampler.historical_count,
            "fresh_per_batch": sampler.fresh_count,
            "maximum_consumption": args.max_replay_consumption,
            "newest_training_positions": newest_positions,
            "fresh_window_training_positions": (
                len(fresh.train) if fresh is not None else 0
            ),
            "examples_presented": sampler.examples_presented,
            "historical_presentations": sampler.historical.presentations,
            "fresh_presentations": (
                sampler.fresh.presentations if sampler.fresh is not None else 0
            ),
        },
        "sources": [_source_report(historical), *([_source_report(fresh)] if fresh else [])],
        "history": history,
        "actor": actor,
        "diagnostics": {"best_validation": best_validation},
    }
    _write_json(args.output / "run.json", run)
    _write_json(args.output / "actor.json", actor_manifest)


def _validation_report(
    learner: KerasLearner,
    historical: SourceSplit,
    fresh: SourceSplit | None,
    *,
    target: Target,
    batch_size: int,
    historical_fraction: float,
) -> dict[str, Any]:
    sources = {
        "historical": _evaluate(
            learner, historical.validation, target=target, batch_size=batch_size
        )
    }
    if fresh is not None:
        sources["fresh"] = _evaluate(
            learner, fresh.validation, target=target, batch_size=batch_size
        )
        objective = (
            historical_fraction * sources["historical"]["total"]
            + (1 - historical_fraction) * sources["fresh"]["total"]
        )
        all_samples = historical.validation + fresh.validation
    else:
        objective = sources["historical"]["total"]
        all_samples = historical.validation

    phases = {}
    for name in ("opening", "middle", "late"):
        panel = tuple(sample for sample in all_samples if phase_name(sample) == name)
        if panel:
            phases[name] = _evaluate(
                learner, panel, target=target, batch_size=batch_size
            )
    return {
        "monitoring_objective": objective,
        "sources": sources,
        "phases": phases,
    }


def _actor_manifest(history: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Publish the newest checkpoint; retain validation-best for inspection only."""

    if not history or int(history[-1]["step"]) == 0:
        raise RuntimeError("training has no completed checkpoint")
    latest = history[-1]
    best = min(
        history,
        key=lambda record: record["validation"]["monitoring_objective"],
    )

    def checkpoint(record: dict[str, Any]) -> dict[str, Any]:
        return {
            "run_step": int(record["step"]),
            "optimizer_step": int(record["optimizer_step"]),
            "path": record["model"]["path"],
            "sha256": record["model"]["sha256"],
            "training_state": record["training_state"],
        }

    diagnostic = checkpoint(best)
    diagnostic["monitoring_objective"] = best["validation"][
        "monitoring_objective"
    ]
    return {
        "status": "complete",
        "actor_key": "latest",
        "models": {
            "latest": checkpoint(latest),
            "best_validation": diagnostic,
        },
        "evaluation_role": "diagnostic_only",
    }


def _evaluate(
    learner: KerasLearner,
    samples: Sequence[PositionSample],
    *,
    target: Target,
    batch_size: int,
) -> dict[str, float]:
    totals: dict[str, float] = {}
    total_weight = 0.0
    for start in range(0, len(samples), batch_size):
        batch = make_training_batch(
            samples[start : start + batch_size], target=target, augment=False
        )
        metrics = learner.evaluate_batch(batch)
        weight = float(np.sum(batch.sample_weights))
        total_weight += weight
        for name, value in metrics.items():
            totals[name] = totals.get(name, 0.0) + value * weight
    return {
        "positions": len(samples),
        **{name: value / total_weight for name, value in totals.items()},
    }


def _load_games(paths: Sequence[Path]):
    games = []
    inputs = []
    seen_paths: set[Path] = set()
    seen_seeds: set[int] = set()
    for root in paths:
        chunk_paths = [root] if root.is_file() else sorted(root.rglob("chunk_*.npz"))
        if not chunk_paths:
            raise ValueError(f"no chunks found under {root}")
        for path in chunk_paths:
            resolved = path.resolve()
            if resolved in seen_paths:
                raise ValueError(f"chunk supplied more than once: {resolved}")
            chunk, manifest = load_chunk(path)
            duplicate = seen_seeds.intersection(game.seed for game in chunk)
            if duplicate:
                raise ValueError(f"duplicate game seed across chunks: {min(duplicate)}")
            seen_paths.add(resolved)
            seen_seeds.update(game.seed for game in chunk)
            games.extend(chunk)
            inputs.append({"path": str(resolved), "sha256": manifest["sha256"]})
    return games, tuple(inputs)


def _split_source(
    name: str,
    games,
    inputs,
    *,
    validation_fraction: float,
    seed: int,
) -> SourceSplit:
    train_indices, validation_indices = _stable_split_indices(
        games, validation_fraction, seed=seed
    )
    return SourceSplit(
        name=name,
        train=tuple(samples_from_games([games[index] for index in train_indices])),
        validation=tuple(
            samples_from_games([games[index] for index in validation_indices])
        ),
        train_games=len(train_indices),
        validation_games=len(validation_indices),
        inputs=inputs,
    )


def _stable_split_indices(games, validation_fraction: float, *, seed: int):
    """Split by immutable game seed so growing a replay window cannot reshuffle it."""

    train = []
    validation = []
    threshold = int(validation_fraction * 2**64)
    for index, game in enumerate(games):
        digest = sha256(f"{seed}:{game.seed}".encode()).digest()
        bucket = int.from_bytes(digest[:8], "big")
        (validation if bucket < threshold else train).append(index)
    if not train or not validation:
        raise ValueError("game-ID split produced an empty train or validation set")
    return train, validation


def _source_report(source: SourceSplit) -> dict[str, Any]:
    return {
        "name": source.name,
        "train_games": source.train_games,
        "validation_games": source.validation_games,
        "train_positions": len(source.train),
        "validation_positions": len(source.validation),
        "inputs": source.inputs,
    }


def _policy_surprise(sample: PositionSample) -> float:
    """KL(search visit policy || stored network prior) for one position."""

    actions = sample.position.actions
    visits = np.asarray([action.visits for action in actions], dtype=np.float64)
    priors = np.asarray(
        [action.network_prior for action in actions], dtype=np.float64
    )
    visit_total = float(visits.sum())
    prior_total = float(priors.sum())
    if visit_total <= 0 or prior_total <= 0:
        return 0.0
    policy = visits / visit_total
    priors /= prior_total
    positive = policy > 0
    return float(
        np.sum(
            policy[positive]
            * np.log(policy[positive] / np.maximum(priors[positive], 1e-12))
        )
    )


def _weight_training_surprise(
    source: SourceSplit, *, strength: float, cap: float
) -> SourceSplit:
    """Apply bounded mean-one surprise weights to training positions only."""

    if strength == 0:
        return source
    raw = np.asarray(
        [min(cap, 1.0 + strength * _policy_surprise(sample)) for sample in source.train],
        dtype=np.float64,
    )
    normalized = raw / raw.mean()
    weighted = tuple(
        replace(sample, loss_weight=sample.loss_weight * float(weight))
        for sample, weight in zip(source.train, normalized, strict=True)
    )
    return replace(source, train=weighted)


def _reject_duplicate_seeds(left, right) -> None:
    duplicate = {game.seed for game in left}.intersection(game.seed for game in right)
    if duplicate:
        raise ValueError(f"duplicate game seed across replay sources: {min(duplicate)}")


def _check_rules(historical: SourceSplit, fresh: SourceSplit | None) -> None:
    samples = historical.train + (fresh.train if fresh is not None else ())
    rules = {sample.position.state.rules for sample in samples}
    if len(rules) != 1:
        raise ValueError("one learner run cannot mix rulesets")


def _check_network_config(model, expected: NetworkConfig) -> None:
    actual = read_network_config(model)
    if actual != expected and not (
        actual.board_size == expected.board_size
        and actual.architecture == expected.architecture
        and actual.channels == expected.channels
        and actual.residual_blocks == expected.residual_blocks
        and actual.value_hidden == expected.value_hidden
        and (
            actual.architecture == "cnn"
            or actual.attention_heads == expected.attention_heads
        )
        and isclose(
            actual.l2_regularization,
            expected.l2_regularization,
            rel_tol=1e-6,
            abs_tol=1e-9,
        )
    ):
        raise ValueError(f"initial model architecture {actual} does not match {expected}")


def _save_model(model, path: Path) -> dict[str, str]:
    model.save(path)
    return {"path": str(path.resolve()), "sha256": _file_sha256(path)}


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def _seed_everything(seed: int) -> None:
    library_seed = seed % 2**32
    random.seed(seed)
    np.random.seed(library_seed)
    import tensorflow as tf

    tf.keras.utils.set_random_seed(library_seed)
    tf.config.experimental.enable_op_determinism()


if __name__ == "__main__":
    main()
