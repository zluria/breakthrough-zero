from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from breakthrough_zero.data import (
    GameRecord,
    PositionRecord,
    load_chunk,
    save_chunk,
    split_game_indices,
    transform_game,
    value_target,
)
from breakthrough_zero.game import MINI_RULES, PLAYER_1, GameState
from breakthrough_zero.search import PUCTSearch, SearchConfig, best_move
from breakthrough_zero.symmetry import Symmetry, transform_move
from test_search import ZeroEvaluator


def sample_game(seed: int = 5) -> GameRecord:
    # Every legal P1 move reaches the goal row. Keeping a P2 piece on square
    # 63 also exercises unsigned bitboard storage without making the state
    # terminal.
    state = GameState(p1=1 << 54, p2=1 << 63, to_move=PLAYER_1, ply=20)
    root = PUCTSearch(
        ZeroEvaluator(), SearchConfig(simulations=12, c_puct=1.0), seed=seed
    ).run(state)
    move = best_move(root)
    position = PositionRecord.from_search(state, root, move)
    terminal = state.clone()
    terminal.make_move(move)
    assert terminal.outcome == PLAYER_1
    return GameRecord(positions=(position,), outcome=terminal.outcome, seed=seed)


class DataTests(unittest.TestCase):
    def test_all_main_value_targets_are_reconstructible(self) -> None:
        game = sample_game()
        position = game.positions[0]
        for target in (
            "outcome",
            "soft_z",
            "mixed_z_q",
            "a0c",
            "played_q",
            "greedy_backup",
        ):
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
                transform_move(
                    original.selected_move, symmetry, original.state.rules
                ),
            )
            self.assertEqual(
                {action.move for action in position.actions},
                {
                    transform_move(action.move, symmetry, original.state.rules)
                    for action in original.actions
                },
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
        self.assertEqual(len(loaded[0].positions[0].actions), 3)
        self.assertEqual(manifest["metadata"]["rules"], "breakthrough-8x8-v1")

    def test_chunk_round_trip_preserves_mini_ruleset(self) -> None:
        state = GameState(
            p1=1 << 24,
            p2=1 << 4,
            to_move=PLAYER_1,
            rules=MINI_RULES,
        )
        root = PUCTSearch(
            ZeroEvaluator(), SearchConfig(simulations=4), seed=13
        ).run(state)
        move = best_move(root)
        game = GameRecord(
            positions=(PositionRecord.from_search(state, root, move),),
            outcome=PLAYER_1,
            seed=13,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mini.npz"
            save_chunk(path, (game,), metadata={})
            loaded, _ = load_chunk(path)

        self.assertEqual(loaded[0].positions[0].state.rules, MINI_RULES)
        self.assertEqual(loaded[0], game)

    def test_validation_split_is_by_disjoint_games(self) -> None:
        train, validation = split_game_indices(20, 0.2, seed=3)
        self.assertEqual(len(validation), 4)
        self.assertFalse(set(train) & set(validation))
        self.assertEqual(set(train) | set(validation), set(range(20)))

    def test_game_record_rejects_a_false_terminal_result(self) -> None:
        state = GameState()
        root = PUCTSearch(
            ZeroEvaluator(), SearchConfig(simulations=4), seed=7
        ).run(state)
        position = PositionRecord.from_search(state, root, best_move(root))
        with self.assertRaisesRegex(ValueError, "final selected move"):
            GameRecord(positions=(position,), outcome=PLAYER_1, seed=7)

    def test_chunks_are_immutable_and_temporary_files_are_cleaned(self) -> None:
        game = sample_game()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "chunk_000.npz"
            save_chunk(path, (game,), metadata={})
            with self.assertRaises(FileExistsError):
                save_chunk(path, (game,), metadata={})
            self.assertFalse(list(Path(directory).glob(".chunk_000-*")))

    def test_failed_publication_never_creates_a_manifest(self) -> None:
        game = sample_game()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "chunk_000.npz"
            with patch("breakthrough_zero.data.os.replace", side_effect=OSError):
                with self.assertRaises(OSError):
                    save_chunk(path, (game,), metadata={})
            self.assertFalse(path.exists())
            self.assertFalse(path.with_suffix(".json").exists())
            self.assertFalse(list(Path(directory).glob(".chunk_000-*")))

    def test_orphan_data_file_is_rejected_as_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "chunk_000.npz"
            path.write_bytes(b"incomplete")
            with self.assertRaisesRegex(ValueError, "manifest is missing"):
                load_chunk(path)

    def test_orphan_manifest_is_rejected_as_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "chunk_000.npz"
            path.with_suffix(".json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "data file is missing"):
                load_chunk(path)


if __name__ == "__main__":
    unittest.main()
