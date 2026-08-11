#!/usr/bin/env python3
"""Run a reproducible paired tournament; the 5x5 rules are the default."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
from itertools import combinations
import json
from math import isfinite
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
from breakthrough_zero.evaluators import SymmetryEnsembleEvaluator
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
    parser.add_argument(
        "--puct-c-puct",
        type=float,
        default=1.5,
        help="exploration constant for both rollout-PUCT baselines",
    )
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
    parser.add_argument(
        "--ensemble-model",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="add a Keras agent averaged over all four exact symmetries",
    )
    parser.add_argument(
        "--tactical-puct",
        action="append",
        default=[],
        metavar="NAME=C_PUCT",
        help="add a tactical-rollout PUCT agent with its own exploration constant",
    )
    parser.add_argument(
        "--baselines",
        choices=("all", "strong", "none"),
        default="all",
        help="include all baselines, only alpha-beta/tactical PUCT, or none",
    )
    parser.add_argument(
        "--matchups",
        choices=("all", "custom-vs-baselines"),
        default="all",
        help="run every pair or only custom agents against fixed baselines",
    )
    args = parser.parse_args()
    if args.max_failures < 0:
        parser.error("--max-failures cannot be negative")
    if args.puct_c_puct < 0:
        parser.error("--puct-c-puct cannot be negative")
    return args


def agents(
    model_specs: list[tuple[str, Path, bool]],
    tactical_puct_specs: list[tuple[str, float]],
    *,
    baseline_set: str,
    puct_c_puct: float = 1.5,
) -> tuple[AgentSpec, ...]:
    specs = []
    if baseline_set == "all":
        specs.extend(
            [
                AgentSpec("random", RandomAgent),
                AgentSpec("alpha-beta", TimedAlphaBetaAgent),
                AgentSpec(
                    "puct-rollout",
                    lambda seed: TimedDummyPUCTAgent(
                        seed, c_puct=puct_c_puct
                    ),
                ),
                AgentSpec(
                    "puct-tactical",
                    lambda seed: TimedDummyPUCTAgent(
                        seed,
                        c_puct=puct_c_puct,
                        prefer_tactical_rollouts=True,
                    ),
                ),
            ]
        )
    elif baseline_set == "strong":
        specs.extend(
            [
                AgentSpec("alpha-beta", TimedAlphaBetaAgent),
                AgentSpec(
                    "puct-tactical",
                    lambda seed: TimedDummyPUCTAgent(
                        seed,
                        c_puct=puct_c_puct,
                        prefer_tactical_rollouts=True,
                    ),
                ),
            ]
        )
    elif baseline_set != "none":
        raise ValueError(f"unknown baseline set: {baseline_set}")
    for name, c_puct in tactical_puct_specs:
        specs.append(
            AgentSpec(
                name,
                lambda seed, c_puct=c_puct: TimedDummyPUCTAgent(
                    seed,
                    c_puct=c_puct,
                    prefer_tactical_rollouts=True,
                ),
            )
        )
    for name, path, use_ensemble in model_specs:
        base = KerasEvaluator(load_network(path))
        evaluator = SymmetryEnsembleEvaluator(base) if use_ensemble else base
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
    model_specs = parse_model_specs(args.model, args.ensemble_model)
    tactical_puct_specs = parse_tactical_puct_specs(args.tactical_puct)
    _check_unique_names(model_specs, tactical_puct_specs)
    custom_names = {
        name for name, _, _ in model_specs
    } | {name for name, _ in tactical_puct_specs}
    if args.baselines == "none" and len(custom_names) < 2:
        raise ValueError("a custom tournament needs at least two agents")
    if args.matchups == "custom-vs-baselines" and (
        args.baselines == "none" or not custom_names
    ):
        raise ValueError("custom-vs-baselines needs both kinds of agent")
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
        "puct_c_puct": args.puct_c_puct,
        "baseline_set": args.baselines,
        "matchups": args.matchups,
        "tactical_puct_agents": dict(tactical_puct_specs),
        "models": {
            name: {
                "path": str(path.resolve()),
                "sha256": file_sha256(path),
                "symmetry_ensemble": use_ensemble,
            }
            for name, path, use_ensemble in model_specs
        },
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

    agent_specs = agents(
        model_specs,
        tactical_puct_specs,
        baseline_set=args.baselines,
        puct_c_puct=args.puct_c_puct,
    )
    warm_up(agent_specs, args.move_seconds, rules)
    summaries = []
    all_games = []
    pairings = matchup_pairs(agent_specs, custom_names, args.matchups)
    for match_index, (agent_a, agent_b) in enumerate(pairings):
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

    names = {spec.name for spec in agent_specs}
    rating_anchor = (
        "random"
        if "random" in names
        else "alpha-beta" if "alpha-beta" in names else agent_specs[0].name
    )
    anchor_rating = 1000.0 if rating_anchor == "random" else 1500.0
    ratings = fit_elo_table(
        summaries, anchor=rating_anchor, anchor_rating=anchor_rating
    )
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
        "matchup_count": len(pairings),
        "opening_config": asdict(opening_config),
        "match_config": match_config.to_record(rules),
        "ratings": ratings,
        "rating_anchor": rating_anchor,
        "rating_anchor_value": anchor_rating,
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


def parse_model_specs(
    values: list[str], ensemble_values: list[str]
) -> list[tuple[str, Path, bool]]:
    reserved = {"random", "alpha-beta", "puct-rollout", "puct-tactical"}
    models = []
    requested = [(value, False) for value in values] + [
        (value, True) for value in ensemble_values
    ]
    for value, use_ensemble in requested:
        if "=" not in value:
            raise ValueError(f"model must use NAME=PATH syntax: {value}")
        name, raw_path = value.split("=", 1)
        path = Path(raw_path)
        duplicate = any(existing == name for existing, _, _ in models)
        if not name or name in reserved or duplicate:
            raise ValueError(f"model name is empty, reserved, or duplicated: {name}")
        if not path.is_file():
            raise FileNotFoundError(f"model does not exist: {path}")
        models.append((name, path, use_ensemble))
    return models


def matchup_pairs(
    agent_specs: tuple[AgentSpec, ...],
    custom_names: set[str],
    mode: str,
) -> tuple[tuple[AgentSpec, AgentSpec], ...]:
    """Return only the comparisons authorized by the experiment design."""

    pairings = tuple(combinations(agent_specs, 2))
    if mode == "all":
        return pairings
    if mode == "custom-vs-baselines":
        return tuple(
            pair
            for pair in pairings
            if (pair[0].name in custom_names) != (pair[1].name in custom_names)
        )
    raise ValueError(f"unknown matchup mode: {mode}")


def parse_tactical_puct_specs(values: list[str]) -> list[tuple[str, float]]:
    """Parse readable named search variants for direct paired comparisons."""

    specs = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"tactical PUCT must use NAME=C_PUCT syntax: {value}")
        name, raw_c_puct = value.split("=", 1)
        try:
            c_puct = float(raw_c_puct)
        except ValueError as error:
            raise ValueError(
                f"invalid c_puct in tactical PUCT spec: {value}"
            ) from error
        if not name or not isfinite(c_puct) or c_puct < 0:
            raise ValueError(f"invalid tactical PUCT spec: {value}")
        if any(existing == name for existing, _ in specs):
            raise ValueError(f"duplicated tactical PUCT name: {name}")
        specs.append((name, c_puct))
    return specs


def _check_unique_names(
    model_specs: list[tuple[str, Path, bool]],
    tactical_puct_specs: list[tuple[str, float]],
) -> None:
    reserved = {"random", "alpha-beta", "puct-rollout", "puct-tactical"}
    names = [name for name, _, _ in model_specs] + [
        name for name, _ in tactical_puct_specs
    ]
    if any(name in reserved for name in names) or len(names) != len(set(names)):
        raise ValueError("custom agent names must be unique and not reserved")


def file_sha256(path: Path) -> str:
    """Identify model contents independently of a mutable filesystem path."""

    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
        f"Smoke ratings are anchored at {report['rating_anchor']} = "
        f"{float(report['rating_anchor_value']):.0f}. Do not treat a small run",
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
