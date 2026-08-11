from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from breakthrough_zero.data import (
    GameRecord,
    PositionRecord,
    load_chunk,
    save_chunk,
    split_game_indices,
    transform_game,
    value_target,
)
from breakthrough_zero.game import PLAYER_1, GameState
from breakthrough_zero.search import PUCTSearch, SearchConfig, best_move
from breakthrough_zero.symmetry import Symmetry, transform_move
from test_search import ZeroEvaluator


def sample_game(seed: int = 5) -> GameRecord:
    state = GameState()
    root = PUCTSearch(
        ZeroEvaluator(), SearchConfig(simulations=12, c_puct=1.0), seed=seed
    ).run(state)
    position = PositionRecord.from_search(state, root, best_move(root))
    return GameRecord(positions=(position,), outcome=PLAYER_1, seed=seed)


class DataTests(unittest.TestCase):
    def test_all_main_value_targets_are_reconstructible(self) -> None:
        game = sample_game()
        position = game.positions[0]
        for target in ("outcome", "soft_z", "a0c", "played_q", "greedy_backup"):
            self.assertTrue(-1 <= value_target(position, game.outcome, target) <= 1)
        self.assertEqual(value_target(position, game.outcome, "outcome"), 1)

    def test_all_symmetries_transform_moves_and_absolute_values(self) -> None:
        game = sample_game()
        original = game.positions[0]
        for symmetry in Symmetry:
            transformed = transform_game(game, symmetry)
            position = transformed.positions[0]
            sign = -1 if symmetry.swap_players else 1
            self.assertEqual(transformed.outcome, sign * game.outcome)
            self.assertAlmostEqual(position.root_value_sum, sign * original.root_value_sum)
            self.assertEqual(position.root_value_square_sum, original.root_value_square_sum)
            self.assertEqual(
                position.selected_move,
                transform_move(original.selected_move, symmetry),
            )
            self.assertEqual(
                {action.move for action in position.actions},
                {transform_move(action.move, symmetry) for action in original.actions},
            )

    def test_chunk_round_trip_preserves_high_bit_and_search_statistics(self) -> None:
        game = sample_game()
        self.assertTrue(game.positions[0].state.p2 & (1 << 63))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "chunk_000.npz"
            save_chunk(path, (game,), metadata={"rules": "breakthrough-8x8-v1"})
            loaded, manifest = load_chunk(path)

        self.assertEqual(loaded[0].outcome, game.outcome)
        self.assertEqual(loaded[0].seed, game.seed)
        self.assertEqual(loaded[0].positions[0].state, game.positions[0].state)
        self.assertEqual(
            loaded[0].positions[0].selected_move,
            game.positions[0].selected_move,
        )
        self.assertEqual(len(loaded[0].positions[0].actions), 22)
        self.assertEqual(manifest["metadata"]["rules"], "breakthrough-8x8-v1")

    def test_validation_split_is_by_disjoint_games(self) -> None:
        train, validation = split_game_indices(20, 0.2, seed=3)
        self.assertEqual(len(validation), 4)
        self.assertFalse(set(train) & set(validation))
        self.assertEqual(set(train) | set(validation), set(range(20)))


if __name__ == "__main__":
    unittest.main()
