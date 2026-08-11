"""Simple evaluators used before a neural network is trusted."""

from __future__ import annotations

import random

import numpy as np
from numpy.typing import NDArray

from .game import ACTION_SIZE, GameState


class RandomRolloutEvaluator:
    """Uniform policy weights and an optional tactical random rollout."""

    def __init__(self, seed: int = 0, *, prefer_tactical: bool = False) -> None:
        self.rng = random.Random(seed)
        self.prefer_tactical = prefer_tactical

    def evaluate(self, state: GameState) -> tuple[NDArray[np.float32], float]:
        rollout = state.clone()
        while rollout.outcome is None:
            move = rollout.random_legal_move(
                self.rng, prefer_tactical=self.prefer_tactical
            )
            rollout.make_move(move, validate=False)

        policy = np.ones(ACTION_SIZE, dtype=np.float32)
        return policy, float(rollout.outcome)
