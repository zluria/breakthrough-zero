from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from breakthrough_zero.arena import (
    AgentSpec,
    ArenaGame,
    Decision,
    MatchConfig,
    RandomAgent,
    play_game,
    play_paired_match,
    save_match,
    validate_game,
)
from breakthrough_zero.game import MINI_RULES, GameState, Move
from breakthrough_zero.openings import Opening, OpeningConfig, OpeningSuite


def short_opening(seed: int = 7) -> Opening:
    state = GameState.initial(MINI_RULES)
    moves = []
    for _ in range(5):
        move = state.legal_moves()[0]
        moves.append(move)
        state.make_move(move, validate=False)
    return Opening(state, tuple(moves), seed)


def one_opening_suite() -> OpeningSuite:
    return OpeningSuite(
        MINI_RULES,
        OpeningConfig(count=1, plies=5, simulations=2),
        11,
        (short_opening(),),
    )


class FirstLegalAgent:
    def __init__(self, budgets: list[float] | None = None) -> None:
        self.budgets = budgets

    def select_move(self, state: GameState, budget: float) -> Decision:
        if self.budgets is not None:
            self.budgets.append(budget)
        return Decision(state.legal_moves()[0], work_units=3)


class SequenceClock:
    def __init__(self, values: list[float]) -> None:
        self.values = iter(values)

    def __call__(self) -> float:
        return next(self.values)


class ArenaTests(unittest.TestCase):
    def test_paired_match_reverses_colors_and_passes_equal_budgets(self) -> None:
        budgets: list[float] = []
        first = AgentSpec("first", lambda seed: FirstLegalAgent(budgets))
        second = AgentSpec("second", lambda seed: FirstLegalAgent(budgets))
        games = play_paired_match(
            one_opening_suite(),
            first,
            second,
            MatchConfig(0.25, max_rated_plies=1),
            seed=19,
        )
        self.assertEqual(
            [(game.p1_agent, game.p2_agent) for game in games],
            [("first", "second"), ("second", "first")],
        )
        self.assertEqual(budgets, [0.25, 0.25])
        self.assertEqual(games[0].p1_seed, games[1].p2_seed)
        self.assertEqual(games[0].p2_seed, games[1].p1_seed)

    def test_late_and_illegal_moves_are_explicit_forfeits(self) -> None:
        opening = short_opening()
        late = play_game(
            opening,
            opening_index=0,
            pair_id=0,
            game_in_pair=0,
            p1=("late", FirstLegalAgent(), 1),
            p2=("other", FirstLegalAgent(), 2),
            config=MatchConfig(0.01, time_tolerance_seconds=0),
            clock=SequenceClock([0.0, 0.02]),
        )
        self.assertEqual(late.termination, "time_forfeit")
        self.assertEqual(late.records[-1].note, "time limit exceeded")

        class IllegalAgent:
            def select_move(self, state: GameState, budget: float) -> Decision:
                return Decision(Move(0, 0))

        illegal = play_game(
            opening,
            opening_index=0,
            pair_id=0,
            game_in_pair=0,
            p1=("illegal", IllegalAgent(), 1),
            p2=("other", FirstLegalAgent(), 2),
            config=MatchConfig(1.0),
        )
        self.assertEqual(illegal.termination, "illegal_move_forfeit")
        self.assertEqual(illegal.winner, -illegal.records[-1].player)

    def test_complete_game_replays_and_keeps_terminal_last_mover(self) -> None:
        opening = short_opening()
        game = play_game(
            opening,
            opening_index=0,
            pair_id=0,
            game_in_pair=0,
            p1=("p1", RandomAgent(3), 3),
            p2=("p2", RandomAgent(5), 5),
            config=MatchConfig(0.1),
        )
        self.assertEqual(game.termination, "terminal")
        validate_game(game, opening)
        self.assertEqual(game.records[-1].player, game.winner)

    def test_match_save_is_atomic_and_immutable(self) -> None:
        suite = one_opening_suite()
        specs = (
            AgentSpec("a", lambda seed: FirstLegalAgent()),
            AgentSpec("b", lambda seed: FirstLegalAgent()),
        )
        games = play_paired_match(
            suite, *specs, MatchConfig(0.1, max_rated_plies=1), seed=4
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "match.json"
            save_match(path, suite, games, MatchConfig(0.1), match_seed=4)
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(len(payload["games"]), 2)
            with self.assertRaises(FileExistsError):
                save_match(path, suite, games, MatchConfig(0.1), match_seed=4)


if __name__ == "__main__":
    unittest.main()
