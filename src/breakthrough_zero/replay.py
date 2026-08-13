"""Small, explicit replay sampling for the learner.

The sampler draws a fixed number of historical and fresh positions in every
batch.  This is easier to reason about than giving a few old positions enormous
loss weights, and it makes replay consumption measurable before training.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import floor
from typing import Sequence

import numpy as np

from .symmetry import Symmetry
from .training import PositionSample


@dataclass(frozen=True, slots=True)
class ReplayBatch:
    samples: tuple[PositionSample, ...]
    symmetry_indices: tuple[int, ...]
    historical_count: int
    fresh_count: int


class _CyclingPool:
    """Shuffle without replacement, then start a new pass."""

    def __init__(self, samples: Sequence[PositionSample], rng: np.random.Generator):
        if not samples:
            raise ValueError("a replay pool cannot be empty")
        self.samples = tuple(samples)
        self.rng = rng
        self.order = np.arange(len(samples))
        self.rng.shuffle(self.order)
        self.cursor = 0
        self.draw_counts = np.zeros(len(samples), dtype=np.int64)
        self.presentations = 0

    def draw(self, count: int) -> tuple[list[PositionSample], list[int]]:
        if not 0 <= count <= len(self.samples):
            raise ValueError("one batch cannot draw a replay position twice")
        chosen: list[PositionSample] = []
        symmetries: list[int] = []
        chosen_indices: set[int] = set()
        while count:
            available = len(self.order) - self.cursor
            take = min(count, available)
            indices = self.order[self.cursor : self.cursor + take]
            for raw_index in indices:
                index = int(raw_index)
                if index in chosen_indices:
                    raise AssertionError("a replay batch drew one position twice")
                chosen_indices.add(index)
                chosen.append(self.samples[index])
                symmetries.append(
                    int(self.draw_counts[index] % len(tuple(Symmetry)))
                )
                self.draw_counts[index] += 1
            self.cursor += take
            self.presentations += take
            count -= take
            if self.cursor == len(self.order):
                self.rng.shuffle(self.order)
                if chosen_indices:
                    unused = [index for index in self.order if int(index) not in chosen_indices]
                    used = [index for index in self.order if int(index) in chosen_indices]
                    self.order = np.asarray(unused + used, dtype=np.int64)
                self.cursor = 0
        return chosen, symmetries

    @property
    def reuse(self) -> float:
        return self.presentations / len(self.samples)


class ReplaySampler:
    """Make stratified batches from historical and fresh self-play."""

    def __init__(
        self,
        historical: Sequence[PositionSample],
        fresh: Sequence[PositionSample] | None,
        *,
        batch_size: int,
        historical_fraction: float = 0.25,
        seed: int = 0,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch size must be positive")
        if not 0 <= historical_fraction <= 1:
            raise ValueError("historical fraction must be in [0, 1]")

        self.rng = np.random.default_rng(seed)
        self.historical = _CyclingPool(historical, self.rng)
        self.fresh = _CyclingPool(fresh, self.rng) if fresh else None
        if self.fresh is None:
            self.historical_count, self.fresh_count = batch_size, 0
        else:
            self.historical_count = round(batch_size * historical_fraction)
            self.fresh_count = batch_size - self.historical_count
            if self.historical_count == 0 or self.fresh_count == 0:
                raise ValueError("a two-source batch must include both sources")
        if self.historical_count > len(historical):
            raise ValueError("historical training split is smaller than its batch quota")
        if self.fresh is not None and self.fresh_count > len(fresh or ()):
            raise ValueError("fresh training split is smaller than its batch quota")

    def draw(self) -> ReplayBatch:
        samples, symmetries = self.historical.draw(self.historical_count)
        if self.fresh is not None:
            fresh_samples, fresh_symmetries = self.fresh.draw(self.fresh_count)
            samples.extend(fresh_samples)
            symmetries.extend(fresh_symmetries)

        order = self.rng.permutation(len(samples))
        return ReplayBatch(
            samples=tuple(samples[int(index)] for index in order),
            symmetry_indices=tuple(symmetries[int(index)] for index in order),
            historical_count=self.historical_count,
            fresh_count=self.fresh_count,
        )

    @property
    def examples_presented(self) -> int:
        fresh = self.fresh.presentations if self.fresh is not None else 0
        return self.historical.presentations + fresh

    @property
    def replay_consumption(self) -> float:
        """Optimizer examples divided by the amount of newest data."""

        denominator = (
            len(self.fresh.samples) if self.fresh is not None else len(self.historical.samples)
        )
        return self.examples_presented / denominator


def step_limit_for_replay(
    *, batch_size: int, newest_positions: int, maximum_consumption: float
) -> int:
    """Convert a replay-consumption cap into a hard optimizer-step cap."""

    if batch_size < 1 or newest_positions < 1:
        raise ValueError("batch size and newest position count must be positive")
    if maximum_consumption <= 0:
        raise ValueError("maximum replay consumption must be positive")
    steps = floor(maximum_consumption * newest_positions / batch_size)
    if steps < 1:
        raise ValueError("replay cap permits no complete optimizer batch")
    return steps


def phase_name(sample: PositionSample) -> str:
    """Use simple board-scaled ply bands for stable validation panels."""

    ply = sample.position.state.ply
    width = sample.position.state.rules.active_size
    if ply < width:
        return "opening"
    if ply < 2 * width:
        return "middle"
    return "late"
