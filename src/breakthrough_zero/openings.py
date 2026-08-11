"""Saved random opening prefixes for diverse, noise-free rated matches."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import random
import tempfile
from typing import Any

from .game import RULESETS_BY_NAME, GameState, Move, Ruleset


OPENING_SCHEMA_VERSION = 2
RANDOM_GENERATOR = "uniform-random-v1"
LEGACY_GENERATOR = "legacy-noisy-puct-v1"


@dataclass(frozen=True, slots=True)
class OpeningConfig:
    count: int
    plies: int = 4
    generator: str = RANDOM_GENERATOR
    reject_immediate_wins: bool = True

    def __post_init__(self) -> None:
        if self.count < 1:
            raise ValueError("opening count must be positive")
        if self.generator == RANDOM_GENERATOR:
            if not 2 <= self.plies <= 10 or self.plies % 2:
                raise ValueError("random opening plies must be even and in [2, 10]")
        elif self.generator == LEGACY_GENERATOR:
            if not 5 <= self.plies <= 10:
                raise ValueError("legacy noisy opening plies must be in [5, 10]")
        else:
            raise ValueError(f"unknown opening generator: {self.generator}")


@dataclass(frozen=True, slots=True)
class Opening:
    state: GameState
    moves: tuple[Move, ...]
    seed: int

    def __post_init__(self) -> None:
        if not 0 <= self.seed < 2**64:
            raise ValueError("opening seed must fit in an unsigned 64-bit integer")
        if self.state.outcome is not None:
            raise ValueError("a rated opening must be non-terminal")

        replay = GameState.initial(self.state.rules)
        for move in self.moves:
            if replay.outcome is not None or not replay.is_legal(move):
                raise ValueError("opening prefix contains an illegal move")
            replay.make_move(move, validate=False)
        if replay != self.state:
            raise ValueError("opening state does not match its move prefix")


@dataclass(frozen=True, slots=True)
class OpeningSuite:
    rules: Ruleset
    config: OpeningConfig
    master_seed: int
    openings: tuple[Opening, ...]

    def __post_init__(self) -> None:
        if not 0 <= self.master_seed < 2**64:
            raise ValueError("master seed must fit in an unsigned 64-bit integer")
        if len(self.openings) != self.config.count:
            raise ValueError("opening count does not match the configuration")
        if any(opening.state.rules != self.rules for opening in self.openings):
            raise ValueError("an opening uses the wrong ruleset")
        if any(len(opening.moves) != self.config.plies for opening in self.openings):
            raise ValueError("an opening has the wrong prefix length")
        if self.config.reject_immediate_wins and any(
            opening.state.has_immediate_winning_move()
            for opening in self.openings
        ):
            raise ValueError("an opening gives the mover an immediate win")
        keys = {
            (opening.state.p1, opening.state.p2, opening.state.to_move)
            for opening in self.openings
        }
        if len(keys) != len(self.openings):
            raise ValueError("opening suite contains duplicate positions")


def generate_opening_suite(
    config: OpeningConfig, rules: Ruleset, *, seed: int
) -> OpeningSuite:
    """Generate candidate-independent prefixes with uniform random moves."""

    if not 0 <= seed < 2**64:
        raise ValueError("master seed must fit in an unsigned 64-bit integer")
    if config.generator != RANDOM_GENERATOR:
        raise ValueError("only the uniform-random opening generator is current")
    seeds = random.Random(seed)
    openings: list[Opening] = []
    seen: set[tuple[int, int, int]] = set()
    attempts = 0
    maximum_attempts = config.count * 100

    while len(openings) < config.count and attempts < maximum_attempts:
        attempts += 1
        opening_seed = seeds.getrandbits(64)
        move_rng = random.Random(opening_seed)
        state = GameState.initial(rules)
        moves: list[Move] = []

        for _ in range(config.plies):
            move = state.random_legal_move(move_rng)
            moves.append(move)
            state.make_move(move, validate=False)
            if state.outcome is not None:
                break

        key = (state.p1, state.p2, state.to_move)
        acceptable = not (
            config.reject_immediate_wins
            and state.has_immediate_winning_move()
        )
        if (
            state.outcome is None
            and len(moves) == config.plies
            and key not in seen
            and acceptable
        ):
            openings.append(Opening(state.clone(), tuple(moves), opening_seed))
            seen.add(key)

    if len(openings) != config.count:
        raise RuntimeError("could not generate enough distinct non-terminal openings")
    return OpeningSuite(rules, config, seed, tuple(openings))


def save_opening_suite(
    path: str | Path,
    suite: OpeningSuite,
    *,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Atomically publish one immutable JSON opening suite."""

    output = Path(path)
    if output.suffix != ".json":
        raise ValueError("opening suite path must end in .json")
    if output.exists():
        raise FileExistsError(f"opening suite already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": OPENING_SCHEMA_VERSION,
        "rules": suite.rules.name,
        "config": asdict(suite.config),
        "master_seed": suite.master_seed,
        "metadata": metadata or {},
        "openings": [
            {
                "seed": opening.seed,
                "p1": hex(opening.state.p1),
                "p2": hex(opening.state.p2),
                "to_move": opening.state.to_move,
                "ply": opening.state.ply,
                "moves": [[move.source, move.target] for move in opening.moves],
            }
            for opening in suite.openings
        ],
    }

    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{output.stem}-", suffix=".json", dir=output.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary)
    try:
        temporary_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        with temporary_path.open("r+b") as file:
            os.fsync(file.fileno())
        os.replace(temporary_path, output)
    finally:
        temporary_path.unlink(missing_ok=True)
    return output


def load_opening_suite(path: str | Path) -> tuple[OpeningSuite, dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    schema_version = payload.get("schema_version")
    if schema_version not in (1, OPENING_SCHEMA_VERSION):
        raise ValueError("unsupported opening-suite schema")
    try:
        rules = RULESETS_BY_NAME[payload["rules"]]
    except KeyError as error:
        raise ValueError("unknown opening-suite ruleset") from error
    if schema_version == 1:
        legacy = payload["config"]
        config = OpeningConfig(
            count=int(legacy["count"]),
            plies=int(legacy["plies"]),
            generator=LEGACY_GENERATOR,
            reject_immediate_wins=False,
        )
    else:
        config = OpeningConfig(**payload["config"])
    openings = tuple(
        Opening(
            state=GameState(
                p1=int(item["p1"], 16),
                p2=int(item["p2"], 16),
                to_move=int(item["to_move"]),
                ply=int(item["ply"]),
                rules=rules,
            ),
            moves=tuple(
                Move(int(source), int(target))
                for source, target in item["moves"]
            ),
            seed=int(item["seed"]),
        )
        for item in payload["openings"]
    )
    suite = OpeningSuite(
        rules=rules,
        config=config,
        master_seed=int(payload["master_seed"]),
        openings=openings,
    )
    return suite, payload.get("metadata", {})
