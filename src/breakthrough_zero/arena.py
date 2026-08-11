"""Paired, wall-clock game evaluation with explicit failure records."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
import json
import os
from pathlib import Path
import random
import tempfile
from time import perf_counter
from typing import Any, Protocol

from .alphabeta import AlphaBetaAgent, AlphaBetaConfig
from .evaluators import RandomRolloutEvaluator
from .game import PLAYER_1, PLAYER_2, GameState, Move, Ruleset
from .openings import Opening, OpeningSuite
from .search import Evaluator, PUCTSearch, SearchConfig, best_move


Clock = Callable[[], float]
ARENA_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class Decision:
    """One agent decision plus an implementation-independent work count."""

    move: Move
    work_units: int = 1
    details: dict[str, int | float | str] | None = None

    def __post_init__(self) -> None:
        if self.work_units < 0:
            raise ValueError("work units cannot be negative")


class Player(Protocol):
    def select_move(
        self, state: GameState, time_limit_seconds: float
    ) -> Decision:
        """Choose a move without mutating the supplied state."""


@dataclass(frozen=True, slots=True)
class AgentSpec:
    """A named factory; a fresh seeded agent is made for every game."""

    name: str
    factory: Callable[[int], Player]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("an agent needs a non-empty name")


class RandomAgent:
    """Uniform random legal play: the fixed lowest-complexity baseline."""

    def __init__(self, seed: int) -> None:
        self.rng = random.Random(seed)

    def select_move(
        self, state: GameState, time_limit_seconds: float
    ) -> Decision:
        del time_limit_seconds
        return Decision(state.random_legal_move(self.rng))


class TimedAlphaBetaAgent:
    """Adapter from the baseline's rich search result to an arena decision."""

    def __init__(
        self, seed: int, config: AlphaBetaConfig = AlphaBetaConfig()
    ) -> None:
        del seed  # The baseline is deterministic; the common factory still seeds it.
        self.agent = AlphaBetaAgent(config)

    def select_move(
        self, state: GameState, time_limit_seconds: float
    ) -> Decision:
        result = self.agent.search(state, time_limit_seconds)
        return Decision(
            move=result.move,
            work_units=result.nodes,
            details={"depth": result.depth, "value": result.value},
        )


class TimedPUCTAgent:
    """PUCT around any evaluator, with no rated-search noise."""

    def __init__(
        self,
        seed: int,
        evaluator: Evaluator,
        *,
        c_puct: float = 1.5,
        min_simulations: int = 2,
    ) -> None:
        self.search = PUCTSearch(
            evaluator,
            SearchConfig(simulations=1, c_puct=c_puct),
            seed=seed,
        )
        self.min_simulations = min_simulations

    def select_move(
        self, state: GameState, time_limit_seconds: float
    ) -> Decision:
        root = self.search.run_for_time(
            state,
            time_limit_seconds,
            min_simulations=self.min_simulations,
        )
        return Decision(
            best_move(root),
            work_units=root.visits,
            details={"root_q": root.q, "simulations": root.visits},
        )


class TimedDummyPUCTAgent(TimedPUCTAgent):
    """Uniform-policy PUCT with a rollout value and no rated-search noise."""

    def __init__(
        self,
        seed: int,
        *,
        c_puct: float = 1.5,
        prefer_tactical_rollouts: bool = False,
        min_simulations: int = 2,
    ) -> None:
        streams = random.Random(seed)
        evaluator = RandomRolloutEvaluator(
            streams.getrandbits(64),
            prefer_tactical=prefer_tactical_rollouts,
        )
        super().__init__(
            streams.getrandbits(64),
            evaluator,
            c_puct=c_puct,
            min_simulations=min_simulations,
        )


@dataclass(frozen=True, slots=True)
class MatchConfig:
    time_limit_seconds: float
    max_rated_plies: int | None = None
    time_tolerance_seconds: float = 0.01

    def __post_init__(self) -> None:
        if self.time_limit_seconds <= 0:
            raise ValueError("move time limit must be positive")
        if self.max_rated_plies is not None and self.max_rated_plies < 1:
            raise ValueError("rated ply limit must be positive")
        if self.time_tolerance_seconds < 0:
            raise ValueError("time tolerance cannot be negative")

    def ply_limit(self, rules: Ruleset) -> int:
        """Return an explicit test cap or the rules-derived safe bound."""

        return self.max_rated_plies or rules.maximum_game_plies

    def to_record(self, rules: Ruleset) -> dict[str, int | float | None]:
        """Serialize both the requested override and effective safety bound."""

        return {
            **asdict(self),
            "effective_max_rated_plies": self.ply_limit(rules),
        }


@dataclass(frozen=True, slots=True)
class MoveRecord:
    ply: int
    player: int
    agent: str
    move: Move | None
    elapsed_seconds: float
    work_units: int
    note: str = ""


@dataclass(frozen=True, slots=True)
class ArenaGame:
    pair_id: int
    game_in_pair: int
    opening_index: int
    opening_seed: int
    p1_agent: str
    p2_agent: str
    p1_seed: int
    p2_seed: int
    records: tuple[MoveRecord, ...]
    winner: int
    termination: str

    def __post_init__(self) -> None:
        if self.game_in_pair not in (0, 1):
            raise ValueError("game_in_pair must be zero or one")
        if self.winner not in (PLAYER_1, 0, PLAYER_2):
            raise ValueError("winner must be Player 1, Player 2, or draw")


def play_game(
    opening: Opening,
    *,
    opening_index: int,
    pair_id: int,
    game_in_pair: int,
    p1: tuple[str, Player, int],
    p2: tuple[str, Player, int],
    config: MatchConfig,
    clock: Clock = perf_counter,
) -> ArenaGame:
    """Play one rated game; an illegal, failed, or very late move forfeits."""

    state = opening.state.clone()
    players = {PLAYER_1: p1, PLAYER_2: p2}
    records: list[MoveRecord] = []
    winner = 0
    termination = "ply_limit_draw"

    for _ in range(config.ply_limit(state.rules)):
        player = state.to_move
        name, agent, _ = players[player]
        started = clock()
        try:
            decision = agent.select_move(state.clone(), config.time_limit_seconds)
        except Exception as error:  # The match must preserve the failure as data.
            elapsed = max(0.0, clock() - started)
            records.append(
                MoveRecord(
                    state.ply,
                    player,
                    name,
                    None,
                    elapsed,
                    0,
                    f"agent error: {type(error).__name__}: {error}",
                )
            )
            winner = -player
            termination = "agent_error_forfeit"
            break

        elapsed = max(0.0, clock() - started)
        if not isinstance(decision, Decision) or not isinstance(decision.move, Move):
            records.append(
                MoveRecord(
                    state.ply,
                    player,
                    name,
                    None,
                    elapsed,
                    0,
                    "agent returned an invalid decision",
                )
            )
            winner = -player
            termination = "agent_error_forfeit"
            break
        record = MoveRecord(
            state.ply,
            player,
            name,
            decision.move,
            elapsed,
            decision.work_units,
        )
        if elapsed > config.time_limit_seconds + config.time_tolerance_seconds:
            records.append(replace(record, note="time limit exceeded"))
            winner = -player
            termination = "time_forfeit"
            break
        if not state.is_legal(decision.move):
            records.append(replace(record, note="illegal move"))
            winner = -player
            termination = "illegal_move_forfeit"
            break

        records.append(record)
        state.make_move(decision.move, validate=False)
        if state.outcome is not None:
            winner = int(state.outcome)
            termination = "terminal"
            break

    return ArenaGame(
        pair_id=pair_id,
        game_in_pair=game_in_pair,
        opening_index=opening_index,
        opening_seed=opening.seed,
        p1_agent=p1[0],
        p2_agent=p2[0],
        p1_seed=p1[2],
        p2_seed=p2[2],
        records=tuple(records),
        winner=winner,
        termination=termination,
    )


def play_paired_match(
    suite: OpeningSuite,
    agent_a: AgentSpec,
    agent_b: AgentSpec,
    config: MatchConfig,
    *,
    seed: int,
    clock: Clock = perf_counter,
) -> tuple[ArenaGame, ...]:
    """Play every saved opening twice, reversing colors within each pair."""

    if agent_a.name == agent_b.name:
        raise ValueError("paired agents need distinct names")
    seeds = random.Random(seed)
    games: list[ArenaGame] = []
    for pair_id, opening in enumerate(suite.openings):
        a_seed, b_seed = seeds.getrandbits(64), seeds.getrandbits(64)
        games.append(
            play_game(
                opening,
                opening_index=pair_id,
                pair_id=pair_id,
                game_in_pair=0,
                p1=(agent_a.name, agent_a.factory(a_seed), a_seed),
                p2=(agent_b.name, agent_b.factory(b_seed), b_seed),
                config=config,
                clock=clock,
            )
        )
        games.append(
            play_game(
                opening,
                opening_index=pair_id,
                pair_id=pair_id,
                game_in_pair=1,
                p1=(agent_b.name, agent_b.factory(b_seed), b_seed),
                p2=(agent_a.name, agent_a.factory(a_seed), a_seed),
                config=config,
                clock=clock,
            )
        )
    return tuple(games)


def validate_game(game: ArenaGame, opening: Opening) -> None:
    """Replay all accepted moves and verify terminal games exactly."""

    if game.opening_seed != opening.seed:
        raise ValueError("arena game references the wrong opening seed")
    state = opening.state.clone()
    for record in game.records:
        if record.ply != state.ply or record.player != state.to_move:
            raise ValueError("arena record has inconsistent ply or player")
        expected_agent = game.p1_agent if record.player == PLAYER_1 else game.p2_agent
        if record.agent != expected_agent:
            raise ValueError("arena record names the wrong moving agent")
        if record.note:
            break
        if record.move is None or not state.is_legal(record.move):
            raise ValueError("arena record contains an illegal accepted move")
        state.make_move(record.move, validate=False)
    if game.termination == "terminal" and state.outcome != game.winner:
        raise ValueError("terminal arena result does not match its trajectory")
    if game.termination == "ply_limit_draw":
        if state.outcome is not None or game.winner != 0:
            raise ValueError("a ply-limit draw has an inconsistent result")
    if game.termination.endswith("_forfeit"):
        if not game.records or not game.records[-1].note:
            raise ValueError("a forfeit needs a final diagnostic record")
        if state.outcome is not None or game.winner != -game.records[-1].player:
            raise ValueError("a forfeit has an inconsistent winner")


def save_match(
    path: str | Path,
    suite: OpeningSuite,
    games: Sequence[ArenaGame],
    config: MatchConfig,
    *,
    match_seed: int,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Atomically publish an immutable, replayable JSON match artifact."""

    output = Path(path)
    if output.suffix != ".json":
        raise ValueError("match path must end in .json")
    if output.exists():
        raise FileExistsError(f"match already exists: {output}")
    for game in games:
        validate_game(game, suite.openings[game.opening_index])

    payload = {
        "schema_version": ARENA_SCHEMA_VERSION,
        "rules": suite.rules.name,
        "opening_master_seed": suite.master_seed,
        "match_seed": match_seed,
        "config": config.to_record(suite.rules),
        "metadata": metadata or {},
        "games": [_game_payload(game) for game in games],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
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


def _game_payload(game: ArenaGame) -> dict[str, Any]:
    payload = asdict(game)
    for record in payload["records"]:
        move = record["move"]
        record["move"] = None if move is None else [move["source"], move["target"]]
    return payload
