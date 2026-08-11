"""Architecture-independent storage for expensive self-play search data."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import random
import tempfile
from typing import Any, Literal

import numpy as np

from .game import RULESETS_BY_NAME, GameState, Move
from .search import Node, greedy_leaf_value
from .symmetry import Symmetry, transform_move, transform_outcome, transform_state

SCHEMA_VERSION = 3
Target = Literal["outcome", "soft_z", "a0c", "played_q", "greedy_backup"]


@dataclass(frozen=True, slots=True)
class ActionStatistics:
    move: Move
    prior: float
    network_prior: float
    visits: int
    value_sum: float
    value_square_sum: float

    def q(self, parent_q: float) -> float:
        return self.value_sum / self.visits if self.visits else parent_q


@dataclass(frozen=True, slots=True)
class PositionRecord:
    state: GameState
    actions: tuple[ActionStatistics, ...]
    selected_move: Move
    root_visits: int
    root_value_sum: float
    root_value_square_sum: float
    root_evaluation: float
    greedy_backup: float
    full_search: bool = True
    sample_weight: float = 1.0

    @property
    def root_q(self) -> float:
        return self.root_value_sum / self.root_visits

    @classmethod
    def from_search(
        cls,
        state: GameState,
        root: Node,
        selected_move: Move,
        *,
        full_search: bool = True,
        sample_weight: float = 1.0,
    ) -> PositionRecord:
        if root.visits == 0 or root.evaluation is None:
            raise ValueError("search root has not been evaluated")
        if selected_move not in root.children:
            raise ValueError("selected move is not a root child")
        if root.state != state:
            raise ValueError("search root and recorded state disagree")
        if set(root.children) != set(state.legal_moves()):
            raise ValueError("search root children do not equal the legal moves")

        actions = tuple(
            ActionStatistics(
                move=move,
                prior=child.prior,
                network_prior=child.network_prior,
                visits=child.visits,
                value_sum=child.value_sum,
                value_square_sum=child.value_square_sum,
            )
            for move, child in root.children.items()
        )
        return cls(
            state=state.clone(),
            actions=actions,
            selected_move=selected_move,
            root_visits=root.visits,
            root_value_sum=root.value_sum,
            root_value_square_sum=root.value_square_sum,
            root_evaluation=root.evaluation,
            greedy_backup=greedy_leaf_value(root),
            full_search=full_search,
            sample_weight=sample_weight,
        )


@dataclass(frozen=True, slots=True)
class GameRecord:
    positions: tuple[PositionRecord, ...]
    outcome: int
    seed: int

    def __post_init__(self) -> None:
        if self.outcome not in (-1, 1):
            raise ValueError("game outcome must be -1 or +1")
        if not self.positions:
            raise ValueError("a game record must contain positions")

        for index, position in enumerate(self.positions):
            state = position.state
            if state.outcome is not None:
                raise ValueError("terminal states must not be stored as positions")

            legal_moves = state.legal_moves()
            action_moves = [action.move for action in position.actions]
            if len(action_moves) != len(set(action_moves)):
                raise ValueError("a position contains duplicate action records")
            if set(action_moves) != set(legal_moves):
                raise ValueError("stored actions do not equal the legal moves")
            if position.selected_move not in action_moves:
                raise ValueError("selected move is absent from the action records")

            next_state = state.clone()
            next_state.make_move(position.selected_move, validate=False)
            if index + 1 < len(self.positions):
                if next_state.outcome is not None:
                    raise ValueError("a game continues after a terminal move")
                if next_state != self.positions[index + 1].state:
                    raise ValueError("consecutive stored positions are inconsistent")
            elif next_state.outcome != self.outcome:
                raise ValueError("the final selected move does not produce the outcome")


def value_target(position: PositionRecord, outcome: int, target: Target) -> float:
    """Derive one absolute value target from retained search statistics."""

    if target == "outcome":
        return float(outcome)
    if target == "soft_z":
        return position.root_q
    if target == "greedy_backup":
        return position.greedy_backup

    if target == "a0c":
        action = max(position.actions, key=lambda item: item.visits)
    elif target == "played_q":
        action = next(
            item for item in position.actions if item.move == position.selected_move
        )
    else:
        raise ValueError(f"unknown value target: {target}")
    return action.q(position.root_q)


def transform_position(
    position: PositionRecord, symmetry: Symmetry
) -> PositionRecord:
    """Apply one exact symmetry, including all absolute value signs."""

    sign = -1.0 if symmetry.swap_players else 1.0
    return PositionRecord(
        state=transform_state(position.state, symmetry),
        actions=tuple(
            ActionStatistics(
                move=transform_move(action.move, symmetry, position.state.rules),
                prior=action.prior,
                network_prior=action.network_prior,
                visits=action.visits,
                value_sum=sign * action.value_sum,
                value_square_sum=action.value_square_sum,
            )
            for action in position.actions
        ),
        selected_move=transform_move(
            position.selected_move, symmetry, position.state.rules
        ),
        root_visits=position.root_visits,
        root_value_sum=sign * position.root_value_sum,
        root_value_square_sum=position.root_value_square_sum,
        root_evaluation=sign * position.root_evaluation,
        greedy_backup=sign * position.greedy_backup,
        full_search=position.full_search,
        sample_weight=position.sample_weight,
    )


def transform_game(game: GameRecord, symmetry: Symmetry) -> GameRecord:
    outcome = transform_outcome(game.outcome, symmetry)
    assert outcome is not None
    return GameRecord(
        positions=tuple(transform_position(p, symmetry) for p in game.positions),
        outcome=outcome,
        seed=game.seed,
    )


def split_game_indices(
    game_count: int, validation_fraction: float, *, seed: int
) -> tuple[list[int], list[int]]:
    """Make a reproducible split without leaking positions between sets."""

    if game_count < 2:
        raise ValueError("at least two games are needed for a split")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    indices = list(range(game_count))
    random.Random(seed).shuffle(indices)
    validation_count = min(
        game_count - 1, max(1, round(game_count * validation_fraction))
    )
    return indices[validation_count:], indices[:validation_count]


def save_chunk(
    path: str | Path,
    games: tuple[GameRecord, ...],
    *,
    metadata: dict[str, Any],
) -> tuple[Path, Path]:
    """Atomically publish one NPZ chunk and its checksummed manifest.

    Chunks are immutable. The data file is moved into place first and the
    manifest second, so the manifest acts as the commit marker. A crash can
    leave an obvious orphan ``.npz`` but never a manifest that blesses partial
    data.
    """

    data_path = Path(path)
    if data_path.suffix != ".npz":
        raise ValueError("chunk path must end in .npz")
    if not games:
        raise ValueError("cannot save an empty chunk")
    data_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = data_path.with_suffix(".json")
    if data_path.exists() or manifest_path.exists():
        raise FileExistsError(f"self-play chunk already exists: {data_path}")

    positions = [position for game in games for position in game.positions]
    actions = [action for position in positions for action in position.actions]
    game_offsets = _offsets([len(game.positions) for game in games])
    action_offsets = _offsets([len(position.actions) for position in positions])

    arrays = dict(
        game_offsets=game_offsets,
        game_outcomes=np.array([game.outcome for game in games], dtype=np.int8),
        game_seeds=np.array([game.seed for game in games], dtype=np.uint64),
        action_offsets=action_offsets,
        p1=np.array([p.state.p1 for p in positions], dtype=np.uint64),
        p2=np.array([p.state.p2 for p in positions], dtype=np.uint64),
        to_move=np.array([p.state.to_move for p in positions], dtype=np.int8),
        ply=np.array([p.state.ply for p in positions], dtype=np.uint16),
        rules=np.array([p.state.rules.name for p in positions], dtype=np.str_),
        selected_source=np.array(
            [p.selected_move.source for p in positions], dtype=np.uint8
        ),
        selected_target=np.array(
            [p.selected_move.target for p in positions], dtype=np.uint8
        ),
        root_visits=np.array([p.root_visits for p in positions], dtype=np.uint32),
        root_value_sum=np.array(
            [p.root_value_sum for p in positions], dtype=np.float32
        ),
        root_value_square_sum=np.array(
            [p.root_value_square_sum for p in positions], dtype=np.float32
        ),
        root_evaluation=np.array(
            [p.root_evaluation for p in positions], dtype=np.float32
        ),
        greedy_backup=np.array([p.greedy_backup for p in positions], dtype=np.float32),
        full_search=np.array([p.full_search for p in positions], dtype=np.bool_),
        sample_weight=np.array([p.sample_weight for p in positions], dtype=np.float32),
        source=np.array([a.move.source for a in actions], dtype=np.uint8),
        target=np.array([a.move.target for a in actions], dtype=np.uint8),
        prior=np.array([a.prior for a in actions], dtype=np.float32),
        network_prior=np.array(
            [a.network_prior for a in actions], dtype=np.float32
        ),
        visits=np.array([a.visits for a in actions], dtype=np.uint32),
        value_sum=np.array([a.value_sum for a in actions], dtype=np.float32),
        value_square_sum=np.array([a.value_square_sum for a in actions], dtype=np.float32),
    )

    data_file, temporary_data = tempfile.mkstemp(
        prefix=f".{data_path.stem}-", suffix=".npz", dir=data_path.parent
    )
    os.close(data_file)
    manifest_file, temporary_manifest = tempfile.mkstemp(
        prefix=f".{data_path.stem}-", suffix=".json", dir=data_path.parent
    )
    os.close(manifest_file)
    temporary_data_path = Path(temporary_data)
    temporary_manifest_path = Path(temporary_manifest)

    try:
        np.savez(temporary_data_path, **arrays)
        _flush_file(temporary_data_path)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "game_count": len(games),
            "position_count": len(positions),
            "action_count": len(actions),
            "sha256": _sha256(temporary_data_path),
            "metadata": metadata,
        }
        temporary_manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        _flush_file(temporary_manifest_path)

        os.replace(temporary_data_path, data_path)
        os.replace(temporary_manifest_path, manifest_path)
    finally:
        temporary_data_path.unlink(missing_ok=True)
        temporary_manifest_path.unlink(missing_ok=True)
    return data_path, manifest_path


def load_chunk(path: str | Path) -> tuple[tuple[GameRecord, ...], dict[str, Any]]:
    """Load and verify a chunk written by :func:`save_chunk`."""

    data_path = Path(path)
    manifest_path = data_path.with_suffix(".json")
    if not manifest_path.exists():
        raise ValueError("self-play chunk is incomplete: manifest is missing")
    if not data_path.exists():
        raise ValueError("self-play chunk is incomplete: data file is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported self-play schema")
    if manifest["sha256"] != _sha256(data_path):
        raise ValueError("self-play chunk checksum does not match its manifest")

    with np.load(data_path, allow_pickle=False) as archive:
        # NpzFile loads one compressed member on every ``archive[name]``
        # access. Materialize each member once before the nested position/action
        # loops; otherwise a small chunk can be decompressed thousands of times.
        arrays = {name: archive[name] for name in archive.files}

    # Reconstruct outside the archive context from the in-memory arrays.
    positions: list[PositionRecord] = []
    for index in range(len(arrays["p1"])):
        start, end = arrays["action_offsets"][index : index + 2]
        actions = tuple(
            ActionStatistics(
                move=Move(int(arrays["source"][j]), int(arrays["target"][j])),
                prior=float(arrays["prior"][j]),
                network_prior=float(arrays["network_prior"][j]),
                visits=int(arrays["visits"][j]),
                value_sum=float(arrays["value_sum"][j]),
                value_square_sum=float(arrays["value_square_sum"][j]),
            )
            for j in range(int(start), int(end))
        )
        positions.append(
            PositionRecord(
                state=GameState(
                    p1=int(arrays["p1"][index]),
                    p2=int(arrays["p2"][index]),
                    to_move=int(arrays["to_move"][index]),
                    ply=int(arrays["ply"][index]),
                    rules=_ruleset(str(arrays["rules"][index])),
                ),
                actions=actions,
                selected_move=Move(
                    int(arrays["selected_source"][index]),
                    int(arrays["selected_target"][index]),
                ),
                root_visits=int(arrays["root_visits"][index]),
                root_value_sum=float(arrays["root_value_sum"][index]),
                root_value_square_sum=float(
                    arrays["root_value_square_sum"][index]
                ),
                root_evaluation=float(arrays["root_evaluation"][index]),
                greedy_backup=float(arrays["greedy_backup"][index]),
                full_search=bool(arrays["full_search"][index]),
                sample_weight=float(arrays["sample_weight"][index]),
            )
        )

    games = []
    for index, outcome in enumerate(arrays["game_outcomes"]):
        start, end = arrays["game_offsets"][index : index + 2]
        games.append(
            GameRecord(
                positions=tuple(positions[int(start) : int(end)]),
                outcome=int(outcome),
                seed=int(arrays["game_seeds"][index]),
            )
        )

    counts_match = (
        len(games) == manifest["game_count"]
        and len(positions) == manifest["position_count"]
        and sum(len(position.actions) for position in positions)
        == manifest["action_count"]
    )
    if not counts_match:
        raise ValueError("self-play counts do not match the manifest")
    return tuple(games), manifest


def _offsets(lengths: list[int]) -> np.ndarray:
    return np.concatenate(([0], np.cumsum(lengths, dtype=np.uint64)))


def _ruleset(name: str):
    try:
        return RULESETS_BY_NAME[name]
    except KeyError as error:
        raise ValueError(f"unknown stored ruleset: {name}") from error


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _flush_file(path: Path) -> None:
    """Ask the operating system to flush one completed temporary file."""

    # Windows requires a writable handle for FlushFileBuffers, which is what
    # Python's fsync uses here.  The file contents are already complete; r+b
    # merely supplies the required handle permissions.
    with path.open("r+b") as file:
        os.fsync(file.fileno())
