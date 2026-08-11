#!/usr/bin/env python3
"""Run a reproducible paired tournament; the 5x5 rules are the default."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from itertools import combinations
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

from breakthrough_zero.arena import (
    AgentSpec,
    MatchConfig,
    RandomAgent,
    TimedAlphaBetaAgent,
    TimedDummyPUCTAgent,
    TimedPUCTAgent,
    play_paired_match,
    save_match,
)
from breakthrough_zero.game import MINI_RULES, STANDARD_RULES, GameState, Ruleset
from breakthrough_zero.network import KerasEvaluator, load_network
from breakthrough_zero.openings import (
    OpeningConfig,
    generate_opening_suite,
    save_opening_suite,
)
from breakthrough_zero.ratings import fit_elo_table, summarize_paired_games


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--pairs", type=int, default=16)
    parser.add_argument("--opening-plies", type=int, default=4)
    parser.add_argument("--rules", choices=("mini", "standard"), default="mini")
    parser.add_argument("--move-seconds", type=float, default=0.05)
    parser.add_argument("--time-tolerance-seconds", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument(
        "--max-failures",
        type=int,
        default=0,
        help="fail after saving results if this many abnormal games are exceeded",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="add a saved Keras model as a neural PUCT agent",
    )
    args = parser.parse_args()
    if args.max_failures < 0:
        parser.error("--max-failures cannot be negative")
    return args


def agents(model_specs: list[tuple[str, Path]]) -> tuple[AgentSpec, ...]:
    specs = [
        AgentSpec("random", RandomAgent),
        AgentSpec("alpha-beta", TimedAlphaBetaAgent),
        AgentSpec("puct-rollout", TimedDummyPUCTAgent),
        AgentSpec(
            "puct-tactical",
            lambda seed: TimedDummyPUCTAgent(
                seed, prefer_tactical_rollouts=True
            ),
        ),
    ]
    for name, path in model_specs:
        evaluator = KerasEvaluator(load_network(path))
        specs.append(
            AgentSpec(
                name,
                lambda seed, evaluator=evaluator: TimedPUCTAgent(seed, evaluator),
            )
        )
    return tuple(specs)


def main() -> None:
    args = parse_args()
    rules = MINI_RULES if args.rules == "mini" else STANDARD_RULES
    model_specs = parse_model_specs(args.model)
    args.output.mkdir(parents=True, exist_ok=False)
    revision = git_revision()
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": revision,
        "host": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "local"),
        "slurm_node": os.environ.get("SLURMD_NODENAME", "local"),
        "rating_note": "One virtual drawn opening pair regularizes Elo.",
        "models": {name: str(path.resolve()) for name, path in model_specs},
    }
    opening_config = OpeningConfig(
        count=args.pairs,
        plies=args.opening_plies,
    )
    suite = generate_opening_suite(opening_config, rules, seed=args.seed)
    save_opening_suite(
        args.output / "openings.json", suite, metadata=metadata
    )
    match_config = MatchConfig(
        time_limit_seconds=args.move_seconds,
        time_tolerance_seconds=args.time_tolerance_seconds,
    )

    agent_specs = agents(model_specs)
    warm_up(agent_specs, args.move_seconds, rules)
    summaries = []
    all_games = []
    for match_index, (agent_a, agent_b) in enumerate(
        combinations(agent_specs, 2)
    ):
        match_seed = args.seed + match_index + 1
        games = play_paired_match(
            suite,
            agent_a,
            agent_b,
            match_config,
            seed=match_seed,
        )
        all_games.extend(games)
        save_match(
            args.output / f"{agent_a.name}_vs_{agent_b.name}.json",
            suite,
            games,
            match_config,
            match_seed=match_seed,
            metadata=metadata,
        )
        summaries.append(
            summarize_paired_games(games, agent_a.name, agent_b.name)
        )

    ratings = fit_elo_table(summaries, anchor="random", anchor_rating=1000.0)
    termination_counts = Counter(game.termination for game in all_games)
    failure_count = sum(
        count for termination, count in termination_counts.items()
        if termination != "terminal"
    )
    report = {
        "metadata": metadata,
        "rules": rules.name,
        "pairs_per_match": args.pairs,
        "games_per_match": 2 * args.pairs,
        "opening_config": asdict(opening_config),
        "match_config": asdict(match_config),
        "ratings": ratings,
        "matches": [asdict(summary) for summary in summaries],
        "termination_counts": dict(sorted(termination_counts.items())),
        "failure_count": failure_count,
    }
    (args.output / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    (args.output / "README.md").write_text(
        markdown_report(report), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if failure_count > args.max_failures:
        raise RuntimeError(
            f"arena recorded {failure_count} abnormal games; "
            f"allowed {args.max_failures}"
        )


def parse_model_specs(values: list[str]) -> list[tuple[str, Path]]:
    reserved = {"random", "alpha-beta", "puct-rollout", "puct-tactical"}
    models = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"model must use NAME=PATH syntax: {value}")
        name, raw_path = value.split("=", 1)
        path = Path(raw_path)
        duplicate = any(existing == name for existing, _ in models)
        if not name or name in reserved or duplicate:
            raise ValueError(f"model name is empty, reserved, or duplicated: {name}")
        if not path.is_file():
            raise FileNotFoundError(f"model does not exist: {path}")
        models.append((name, path))
    return models


def git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def warm_up(
    agent_specs: tuple[AgentSpec, ...], move_seconds: float, rules: Ruleset
) -> None:
    """Exercise each code path once outside every recorded match clock."""

    state = GameState.initial(rules)
    for index, spec in enumerate(agent_specs):
        agent = spec.factory(10_000 + index)
        agent.select_move(state, min(0.01, move_seconds))


def markdown_report(report: dict[str, object]) -> str:
    lines = [
        "# Breakthrough tournament",
        "",
        "Smoke ratings are anchored at random = 1000. Do not treat a small run",
        "as a stable strength claim. Confidence intervals are head-to-head, not",
        "global-rating intervals.",
        "",
        f"Termination counts: `{report['termination_counts']}`. ",
        f"Abnormal games: **{report['failure_count']}**.",
        "",
        "| Agent | Fitted Elo |",
        "| --- | ---: |",
    ]
    ratings = report["ratings"]
    assert isinstance(ratings, dict)
    for name, rating in sorted(
        ratings.items(), key=lambda item: float(item[1]), reverse=True
    ):
        lines.append(f"| {name} | {float(rating):.1f} |")
    lines.extend(
        [
            "",
            "| Match (first-agent view) | W-L-D | Pair sweeps | "
            "Color splits | Elo | 95% CI |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    matches = report["matches"]
    assert isinstance(matches, list)
    for match in matches:
        lines.append(
            f"| {match['agent_a']} vs {match['agent_b']} | "
            f"{match['wins']}-{match['losses']}-{match['draws']} | "
            f"{match['agent_a_pair_sweeps']}-{match['agent_b_pair_sweeps']} | "
            f"{match['color_split_pairs']} | {match['elo_difference']:+.1f} | "
            f"[{match['elo_95_low']:+.1f}, {match['elo_95_high']:+.1f}] |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
