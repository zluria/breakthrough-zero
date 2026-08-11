"""Small, explicit Elo summaries for paired arena games."""

from __future__ import annotations

from dataclasses import dataclass
from math import log10, sqrt
from typing import Sequence

import numpy as np

from .arena import ArenaGame
from .game import PLAYER_1, PLAYER_2


@dataclass(frozen=True, slots=True)
class EloSummary:
    """Head-to-head result from the named first agent's perspective."""

    agent_a: str
    agent_b: str
    games: int
    wins: int
    losses: int
    draws: int
    agent_a_pair_sweeps: int
    agent_b_pair_sweeps: int
    color_split_pairs: int
    draw_affected_pairs: int
    score: float
    regularized_score: float
    elo_difference: float
    elo_95_low: float
    elo_95_high: float


def summarize_paired_games(
    games: Sequence[ArenaGame], agent_a: str, agent_b: str
) -> EloSummary:
    """Validate the pairing and estimate Elo with a small-sample correction.

    An opening pair, not an individual game, is the independent observation.
    Each pair contributes its average score in ``[0, 1]``. One virtual drawn
    pair prevents infinite early Elo, and a Wilson interval uses the number of
    pairs. This fractional-Wilson interval is simple and deliberately more
    conservative than pretending the two color-reversed games are independent.
    """

    if not games:
        raise ValueError("cannot rate an empty match")
    if not agent_a or not agent_b or agent_a == agent_b:
        raise ValueError("ratings require two distinct named agents")
    pairs = _validate_pairs(games, agent_a, agent_b)

    wins = losses = draws = 0
    for game in games:
        a_player = 1 if game.p1_agent == agent_a else -1
        if game.winner == 0:
            draws += 1
        elif game.winner == a_player:
            wins += 1
        else:
            losses += 1

    count = len(games)
    points = wins + 0.5 * draws
    score = points / count
    pair_scores = [
        sum(_agent_score(game, agent_a) for game in pair) / len(pair)
        for pair in pairs.values()
    ]
    agent_a_sweeps = sum(score == 1.0 for score in pair_scores)
    agent_b_sweeps = sum(score == 0.0 for score in pair_scores)
    color_splits = sum(
        score == 0.5 and all(game.winner != 0 for game in pair)
        for score, pair in zip(pair_scores, pairs.values(), strict=True)
    )
    draw_affected = len(pair_scores) - agent_a_sweeps - agent_b_sweeps - color_splits
    regularized = (sum(pair_scores) + 0.5) / (len(pair_scores) + 1)
    low, high = _wilson_interval(regularized, len(pair_scores) + 1)
    return EloSummary(
        agent_a=agent_a,
        agent_b=agent_b,
        games=count,
        wins=wins,
        losses=losses,
        draws=draws,
        agent_a_pair_sweeps=agent_a_sweeps,
        agent_b_pair_sweeps=agent_b_sweeps,
        color_split_pairs=color_splits,
        draw_affected_pairs=draw_affected,
        score=score,
        regularized_score=regularized,
        elo_difference=_elo(regularized),
        elo_95_low=_elo(_inside_unit_interval(low)),
        elo_95_high=_elo(_inside_unit_interval(high)),
    )


def fit_elo_table(
    summaries: Sequence[EloSummary],
    *,
    anchor: str,
    anchor_rating: float = 1000.0,
) -> dict[str, float]:
    """Least-squares Elo table with one fixed, immutable anchor rating."""

    names = sorted(
        {name for summary in summaries for name in (summary.agent_a, summary.agent_b)}
    )
    if anchor not in names:
        raise ValueError("the Elo anchor does not occur in the results")
    variables = [name for name in names if name != anchor]
    if not variables:
        return {anchor: anchor_rating}
    indices = {name: index for index, name in enumerate(variables)}
    rows: list[list[float]] = []
    targets: list[float] = []
    for summary in summaries:
        weight = sqrt(summary.games)
        row = [0.0] * len(variables)
        if summary.agent_a != anchor:
            row[indices[summary.agent_a]] += weight
        if summary.agent_b != anchor:
            row[indices[summary.agent_b]] -= weight
        rows.append(row)
        targets.append(summary.elo_difference * weight)

    solution, _, rank, _ = np.linalg.lstsq(
        np.asarray(rows), np.asarray(targets), rcond=None
    )
    if rank != len(variables):
        raise ValueError("match graph is not connected to the Elo anchor")
    ratings = {anchor: anchor_rating}
    ratings.update(
        {
            name: anchor_rating + float(solution[indices[name]])
            for name in variables
        }
    )
    return ratings


def _validate_pairs(
    games: Sequence[ArenaGame], agent_a: str, agent_b: str
) -> dict[int, list[ArenaGame]]:
    pairs: dict[int, list[ArenaGame]] = {}
    for game in games:
        if {game.p1_agent, game.p2_agent} != {agent_a, agent_b}:
            raise ValueError("a game contains the wrong agents")
        pairs.setdefault(game.pair_id, []).append(game)

    for pair in pairs.values():
        if len(pair) != 2 or {game.game_in_pair for game in pair} != {0, 1}:
            raise ValueError("each opening needs exactly two numbered games")
        if len({game.opening_index for game in pair}) != 1:
            raise ValueError("a pair uses different openings")
        if len({game.opening_seed for game in pair}) != 1:
            raise ValueError("a pair uses different opening seeds")
        a_colors = [game.p1_agent == agent_a for game in pair]
        if sum(a_colors) != 1:
            raise ValueError("a pair does not reverse the agents' colors")
    return pairs


def _agent_score(game: ArenaGame, agent: str) -> float:
    if game.winner == 0:
        return 0.5
    agent_player = PLAYER_1 if game.p1_agent == agent else PLAYER_2
    return float(game.winner == agent_player)


def _elo(score: float) -> float:
    return 400.0 * log10(score / (1.0 - score))


def _wilson_interval(score: float, count: int) -> tuple[float, float]:
    z = 1.959963984540054
    z_squared = z * z
    denominator = 1.0 + z_squared / count
    center = (score + z_squared / (2.0 * count)) / denominator
    radius = z * sqrt(
        score * (1.0 - score) / count + z_squared / (4.0 * count * count)
    ) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def _inside_unit_interval(value: float) -> float:
    return min(1.0 - 1e-12, max(1e-12, value))
