"""Turn immutable search records into neural-network training batches.

Raw records stay in absolute game coordinates.  This module is the one place
where a randomly selected exact symmetry is applied and dense policy targets
are constructed.  Keeping this boundary independent of TensorFlow makes the
most error-prone transformations cheap to unit test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .data import GameRecord, PositionRecord, Target, transform_position, value_target
from .game import ACTION_SIZE, BOARD_SIZE
from .symmetry import Symmetry, transform_outcome


@dataclass(frozen=True, slots=True)
class PositionSample:
    """One position paired with its game's absolute final result."""

    position: PositionRecord
    outcome: int


@dataclass(frozen=True, slots=True)
class TrainingBatch:
    """Dense arrays consumed by the Keras learner."""

    boards: NDArray[np.float32]
    policies: NDArray[np.float32]
    legal_masks: NDArray[np.bool_]
    values: NDArray[np.float32]
    sample_weights: NDArray[np.float32]

    def __post_init__(self) -> None:
        size = len(self.boards)
        expected = {
            "boards": (size, BOARD_SIZE, BOARD_SIZE, 3),
            "policies": (size, ACTION_SIZE),
            "legal_masks": (size, ACTION_SIZE),
            "values": (size,),
            "sample_weights": (size,),
        }
        for name, shape in expected.items():
            if getattr(self, name).shape != shape:
                raise ValueError(f"{name} has shape {getattr(self, name).shape}, not {shape}")


def samples_from_games(games: Sequence[GameRecord]) -> list[PositionSample]:
    """Flatten games without copying their relatively large position records."""

    return [
        PositionSample(position=position, outcome=game.outcome)
        for game in games
        for position in game.positions
    ]


def make_training_batch(
    samples: Sequence[PositionSample],
    *,
    target: Target,
    rng: np.random.Generator | None = None,
    augment: bool = True,
) -> TrainingBatch:
    """Compile samples, selecting one of four symmetries per draw.

    Passing ``augment=False`` uses the identity transformation and is useful
    for stable validation metrics.  Augmentation is performed on demand rather
    than materializing four copies of every expensive search record.
    """

    if not samples:
        raise ValueError("cannot make an empty training batch")
    if augment and rng is None:
        raise ValueError("augmented batches require an explicit random generator")

    symmetries = tuple(Symmetry)
    boards = np.empty((len(samples), BOARD_SIZE, BOARD_SIZE, 3), dtype=np.float32)
    policies = np.zeros((len(samples), ACTION_SIZE), dtype=np.float32)
    legal_masks = np.zeros((len(samples), ACTION_SIZE), dtype=np.bool_)
    values = np.empty(len(samples), dtype=np.float32)
    sample_weights = np.empty(len(samples), dtype=np.float32)

    for index, sample in enumerate(samples):
        symmetry = (
            symmetries[int(rng.integers(len(symmetries)))]
            if augment and rng is not None
            else Symmetry.IDENTITY
        )
        position = transform_position(sample.position, symmetry)
        outcome = transform_outcome(sample.outcome, symmetry)
        assert outcome is not None

        boards[index] = position.state.encode()
        _write_policy(position, policies[index], legal_masks[index])
        values[index] = value_target(position, outcome, target)
        sample_weights[index] = position.sample_weight

    return TrainingBatch(
        boards=boards,
        policies=policies,
        legal_masks=legal_masks,
        values=values,
        sample_weights=sample_weights,
    )


def _write_policy(
    position: PositionRecord,
    policy: NDArray[np.float32],
    legal_mask: NDArray[np.bool_],
) -> None:
    visits = np.array([action.visits for action in position.actions], dtype=np.float64)
    if visits.sum() > 0:
        weights = visits / visits.sum()
    else:
        priors = np.array(
            [action.network_prior for action in position.actions], dtype=np.float64
        )
        if not np.all(np.isfinite(priors)) or np.any(priors < 0):
            raise ValueError("stored policy priors must be finite and non-negative")
        total = float(priors.sum())
        weights = priors / total if total > 0 else np.full(len(priors), 1 / len(priors))

    for action, weight in zip(position.actions, weights, strict=True):
        policy_index = position.state.policy_index(action.move)
        if legal_mask[policy_index]:
            raise ValueError("two stored moves map to the same policy index")
        legal_mask[policy_index] = True
        policy[policy_index] = weight

    if not np.isclose(float(policy.sum()), 1.0, atol=1e-6):
        raise RuntimeError("policy target does not sum to one")
