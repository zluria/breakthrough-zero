"""Turn immutable search records into neural-network training batches.

Raw records stay in absolute game coordinates.  This module is the one place
where one exact symmetry is applied and dense policy targets
are constructed.  Keeping this boundary independent of TensorFlow makes the
most error-prone transformations cheap to unit test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .data import GameRecord, PositionRecord, Target, transform_position, value_target
from .game import POLICY_PLANES
from .symmetry import Symmetry, transform_outcome


@dataclass(frozen=True, slots=True)
class PositionSample:
    """One position, its absolute result, and an optional replay loss weight."""

    position: PositionRecord
    outcome: int
    loss_weight: float = 1.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.loss_weight) or self.loss_weight <= 0:
            raise ValueError("a position sample needs a positive finite loss weight")


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
        if self.boards.ndim != 4 or self.boards.shape[-1] != 3:
            raise ValueError("boards must have shape (batch, side, side, 3)")
        board_size = self.boards.shape[1]
        if self.boards.shape[2] != board_size:
            raise ValueError("training boards must be square")
        action_size = board_size * board_size * POLICY_PLANES
        expected = {
            "boards": (size, board_size, board_size, 3),
            "policies": (size, action_size),
            "legal_masks": (size, action_size),
            "values": (size,),
            "sample_weights": (size,),
        }
        for name, shape in expected.items():
            if getattr(self, name).shape != shape:
                raise ValueError(f"{name} has shape {getattr(self, name).shape}, not {shape}")


def samples_from_games(
    games: Sequence[GameRecord], *, loss_weight: float = 1.0
) -> list[PositionSample]:
    """Flatten games without copying their relatively large position records."""

    return [
        PositionSample(
            position=position,
            outcome=game.outcome,
            loss_weight=loss_weight,
        )
        for game in games
        for position in game.positions
    ]


def make_training_batch(
    samples: Sequence[PositionSample],
    *,
    target: Target,
    rng: np.random.Generator | None = None,
    symmetry_indices: Sequence[int] | None = None,
    augment: bool = True,
) -> TrainingBatch:
    """Compile samples, selecting one of four symmetries per draw.

    Passing ``augment=False`` uses the identity transformation and is useful
    for stable validation metrics.  Augmentation is performed on demand rather
    than materializing four copies of every expensive search record. A caller
    may supply exact ``symmetry_indices`` to run a balanced four-epoch cycle;
    otherwise an explicit random generator chooses them reproducibly.
    """

    if not samples:
        raise ValueError("cannot make an empty training batch")
    if augment and rng is None and symmetry_indices is None:
        raise ValueError("augmented batches require a generator or symmetry indices")
    if not augment and symmetry_indices is not None:
        raise ValueError("identity-only batches cannot specify symmetries")
    if symmetry_indices is not None and len(symmetry_indices) != len(samples):
        raise ValueError("one symmetry index is required per sample")
    rules = samples[0].position.state.rules
    if any(sample.position.state.rules != rules for sample in samples):
        raise ValueError("one training batch cannot mix rulesets")

    symmetries = tuple(Symmetry)
    board_size = rules.active_size
    action_size = rules.action_size
    boards = np.empty((len(samples), board_size, board_size, 3), dtype=np.float32)
    policies = np.zeros((len(samples), action_size), dtype=np.float32)
    legal_masks = np.zeros((len(samples), action_size), dtype=np.bool_)
    values = np.empty(len(samples), dtype=np.float32)
    sample_weights = np.empty(len(samples), dtype=np.float32)

    for index, sample in enumerate(samples):
        if symmetry_indices is not None:
            symmetry_index = int(symmetry_indices[index])
            if not 0 <= symmetry_index < len(symmetries):
                raise ValueError(f"invalid symmetry index: {symmetry_index}")
            symmetry = symmetries[symmetry_index]
        elif augment and rng is not None:
            symmetry = symmetries[int(rng.integers(len(symmetries)))]
        else:
            symmetry = Symmetry.IDENTITY
        position = transform_position(sample.position, symmetry)
        outcome = transform_outcome(sample.outcome, symmetry)
        assert outcome is not None

        boards[index] = position.state.encode()
        _write_policy(position, policies[index], legal_masks[index])
        values[index] = value_target(position, outcome, target)
        sample_weights[index] = position.sample_weight * sample.loss_weight

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
