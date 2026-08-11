from __future__ import annotations

import unittest

import numpy as np

from breakthrough_zero.evaluators import RandomRolloutEvaluator
from breakthrough_zero.game import ACTION_SIZE, PLAYER_1, PLAYER_2, GameState, Move
from breakthrough_zero.search import (
    Node,
    PUCTSearch,
    RootNoiseConfig,
    SearchConfig,
    backup,
    best_move,
    select_child,
)


class ZeroEvaluator:
    def __init__(self, policy: np.ndarray | None = None) -> None:
        self.policy = (
            np.ones(ACTION_SIZE, dtype=np.float32) if policy is None else policy
        )

    def evaluate(self, state: GameState) -> tuple[np.ndarray, float]:
        return self.policy, 0.0


def searched(state: GameState, simulations: int = 2) -> Node:
    return PUCTSearch(
        ZeroEvaluator(), SearchConfig(simulations=simulations, c_puct=1.0)
    ).run(state)


class SearchTests(unittest.TestCase):
    def test_timed_search_completes_whole_simulations_without_noise(self) -> None:
        class StepClock:
            def __init__(self) -> None:
                self.value = 0.0

            def __call__(self) -> float:
                current = self.value
                self.value += 0.01
                return current

        state = GameState()
        root = PUCTSearch(
            ZeroEvaluator(), SearchConfig(simulations=99), seed=17
        ).run_for_time(
            state,
            0.015,
            min_simulations=2,
            clock=StepClock(),
        )
        self.assertEqual(root.visits, 3)
        self.assertEqual(
            [child.prior for child in root.children.values()],
            [child.network_prior for child in root.children.values()],
        )
        self.assertEqual(state, GameState())

    def test_search_does_not_mutate_root_state(self) -> None:
        state = GameState()
        original = state.clone()
        searched(state, simulations=12)
        self.assertEqual(state, original)

    def test_expansion_masks_illegal_policy_and_normalizes_legal_priors(self) -> None:
        raw = np.zeros(ACTION_SIZE, dtype=np.float32)
        raw[-1] = 1000.0  # Illegal in the initial position.
        root = PUCTSearch(ZeroEvaluator(raw), SearchConfig(simulations=1)).run(
            GameState()
        )
        self.assertEqual(set(root.children), set(GameState().legal_moves()))
        self.assertAlmostEqual(sum(child.prior for child in root.children.values()), 1)
        self.assertEqual(
            {round(child.prior, 12) for child in root.children.values()}, {round(1 / 22, 12)}
        )

    def test_visit_accounting(self) -> None:
        root = searched(GameState(), simulations=9)
        self.assertEqual(root.visits, 9)
        self.assertEqual(sum(child.visits for child in root.children.values()), 8)

    def test_root_noise_does_not_destroy_the_network_prior(self) -> None:
        root = PUCTSearch(
            ZeroEvaluator(),
            SearchConfig(simulations=1),
            seed=4,
        ).run(
            GameState(),
            root_noise=RootNoiseConfig(fraction=0.5, total_concentration=10),
        )
        network_priors = [child.network_prior for child in root.children.values()]
        search_priors = [child.prior for child in root.children.values()]
        self.assertTrue(all(prior == 1 / 22 for prior in network_priors))
        self.assertNotEqual(network_priors, search_priors)
        self.assertAlmostEqual(sum(search_priors), 1)

    def test_backup_never_changes_absolute_sign(self) -> None:
        parent = Node(state=GameState(to_move=PLAYER_1))
        child = Node(parent=parent)
        backup([parent, child], -0.75)
        self.assertEqual(parent.q, -0.75)
        self.assertEqual(child.q, -0.75)

    def test_terminal_child_keeps_last_mover_and_absolute_value(self) -> None:
        state = GameState(p1=1 << 48, p2=1 << 15, to_move=PLAYER_2)
        root = searched(state, simulations=20)
        terminal_children = [
            child
            for move, child in root.children.items()
            if move.target // 8 == 0 and child.visits
        ]
        self.assertTrue(terminal_children)
        self.assertTrue(
            all(
                child.state is not None and child.state.to_move == PLAYER_2
                for child in terminal_children
            )
        )
        self.assertTrue(all(child.q == -1 for child in terminal_children))

    def test_unvisited_child_uses_parent_q(self) -> None:
        parent = Node(
            state=GameState(to_move=PLAYER_1), visits=4, value_sum=1.0
        )
        child = Node(parent=parent)
        self.assertEqual(child.q, 0.25)

    def test_selection_maximizes_for_p1_and_minimizes_for_p2(self) -> None:
        moves = [Move(8, 16), Move(9, 17)]
        for player, expected in ((PLAYER_1, moves[0]), (PLAYER_2, moves[1])):
            parent = Node(state=GameState(to_move=player), visits=10)
            parent.children = {
                moves[0]: Node(parent=parent, visits=2, value_sum=1.6),
                moves[1]: Node(parent=parent, visits=2, value_sum=-1.2),
            }
            self.assertEqual(select_child(parent, c_puct=0)[0], expected)

    def test_states_are_cached_only_for_visited_nodes(self) -> None:
        state = GameState()
        root = searched(state, simulations=9)
        self.assertIsNot(root.state, state)
        self.assertEqual(root.state, state)
        cached_children = [
            child for child in root.children.values() if child.state is not None
        ]
        self.assertEqual(
            len(cached_children),
            sum(child.visits > 0 for child in root.children.values()),
        )

    def test_search_finds_immediate_win_for_either_player(self) -> None:
        positions = [
            GameState(p1=1 << 48, p2=1 << 15, to_move=PLAYER_1),
            GameState(p1=1 << 48, p2=1 << 15, to_move=PLAYER_2),
        ]
        for state in positions:
            mover = state.to_move
            root = searched(state, simulations=40)
            state.make_move(best_move(root))
            self.assertEqual(state.outcome, mover)

    def test_random_rollout_is_seeded_and_absolute(self) -> None:
        state = GameState()
        first = RandomRolloutEvaluator(seed=17).evaluate(state)
        second = RandomRolloutEvaluator(seed=17).evaluate(state)
        np.testing.assert_array_equal(first[0], second[0])
        self.assertEqual(first[1], second[1])
        self.assertIn(first[1], (-1.0, 1.0))

    def test_tactical_rollout_move_prefers_a_win_then_a_capture(self) -> None:
        import random

        winning = GameState(
            p1=(1 << 48) | (1 << 8), p2=1 << 15, to_move=PLAYER_1
        )
        move = winning.random_legal_move(
            random.Random(1), prefer_tactical=True
        )
        self.assertEqual(move.target // 8, 7)

        capturing = GameState(
            p1=(1 << 8) | (1 << 9), p2=1 << 17, to_move=PLAYER_1
        )
        move = capturing.random_legal_move(
            random.Random(1), prefer_tactical=True
        )
        self.assertEqual(move.target, 17)


if __name__ == "__main__":
    unittest.main()
