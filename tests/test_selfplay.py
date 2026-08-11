from __future__ import annotations

import unittest

import numpy as np

from breakthrough_zero.game import MINI_RULES, PLAYER_1, GameState, Move
from breakthrough_zero.search import Node, SearchConfig
from breakthrough_zero.selfplay import (
    SelfPlayConfig,
    generate_dummy_games,
    play_dummy_game,
    sample_move,
)


class SelfPlayTests(unittest.TestCase):
    def test_mini_games_use_the_same_reproducible_pipeline(self) -> None:
        config = SelfPlayConfig(
            search=SearchConfig(simulations=4), sample_until_ply=4, max_plies=40
        )
        game = next(
            generate_dummy_games(1, config, seed=29, rules=MINI_RULES)
        )
        replay = play_dummy_game(
            config,
            seed=game.seed,
            initial_state=GameState.initial(MINI_RULES),
        )
        self.assertEqual(game, replay)
        self.assertTrue(
            all(position.state.rules == MINI_RULES for position in game.positions)
        )

    def test_terminal_move_is_not_recorded_as_an_opponent_position(self) -> None:
        initial = GameState(p1=1 << 54, p2=1 << 63, ply=20)
        game = play_dummy_game(
            SelfPlayConfig(
                search=SearchConfig(simulations=4), sample_until_ply=99
            ),
            seed=11,
            initial_state=initial,
        )

        self.assertEqual(len(game.positions), 1)
        self.assertIsNone(game.positions[0].state.outcome)
        terminal = game.positions[0].state.clone()
        terminal.make_move(game.positions[0].selected_move)
        self.assertEqual(terminal.outcome, PLAYER_1)
        self.assertEqual(terminal.to_move, PLAYER_1)

    def test_dummy_game_is_reproducible_from_its_stored_seed(self) -> None:
        config = SelfPlayConfig(
            search=SearchConfig(simulations=3, c_puct=1.2),
            sample_until_ply=8,
        )
        first = play_dummy_game(config, seed=1234)
        second = play_dummy_game(config, seed=first.seed)
        self.assertEqual(first, second)

    def test_visit_sampling_and_deterministic_selection(self) -> None:
        first = Move(8, 16)
        second = Move(9, 17)
        root = Node(
            children={
                first: Node(prior=0.5, visits=0),
                second: Node(prior=0.5, visits=10),
            }
        )
        rng = np.random.default_rng(5)
        self.assertEqual(sample_move(root, rng, temperature=1.0), second)
        self.assertEqual(sample_move(root, rng, temperature=0.0), second)

    def test_generated_games_have_distinct_replayable_seeds(self) -> None:
        config = SelfPlayConfig(search=SearchConfig(simulations=2))
        games = list(generate_dummy_games(2, config, seed=99))
        self.assertNotEqual(games[0].seed, games[1].seed)
        for game in games:
            self.assertEqual(game, play_dummy_game(config, seed=game.seed))

    def test_ply_limit_fails_loudly(self) -> None:
        config = SelfPlayConfig(search=SearchConfig(simulations=1), max_plies=1)
        with self.assertRaisesRegex(RuntimeError, "ply limit"):
            play_dummy_game(config, seed=3)


if __name__ == "__main__":
    unittest.main()
