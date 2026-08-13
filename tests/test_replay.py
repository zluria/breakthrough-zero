from __future__ import annotations

import unittest
from types import SimpleNamespace

from breakthrough_zero.game import MINI_RULES, GameState
from breakthrough_zero.replay import ReplaySampler, phase_name, step_limit_for_replay
from breakthrough_zero.search import SearchConfig
from breakthrough_zero.selfplay import SelfPlayConfig, play_dummy_game
from breakthrough_zero.training import samples_from_games
from scripts.train_replay import (
    SourceSplit,
    _actor_manifest,
    _policy_surprise,
    _stable_split_indices,
    _weight_training_surprise,
)


class ReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        game = play_dummy_game(
            SelfPlayConfig(search=SearchConfig(simulations=4)),
            seed=17,
            initial_state=GameState.initial(MINI_RULES),
        )
        cls.samples = samples_from_games([game])

    def test_two_source_batches_have_exact_unweighted_quotas(self) -> None:
        sampler = ReplaySampler(
            self.samples,
            self.samples,
            batch_size=8,
            historical_fraction=0.25,
            seed=9,
        )
        batch = sampler.draw()

        self.assertEqual(batch.historical_count, 2)
        self.assertEqual(batch.fresh_count, 6)
        self.assertEqual(len(batch.samples), 8)
        self.assertEqual(sampler.examples_presented, 8)
        self.assertAlmostEqual(sampler.replay_consumption, 8 / len(self.samples))

    def test_symmetry_cycle_covers_all_four_transforms(self) -> None:
        single = self.samples[:1]
        sampler = ReplaySampler(single, None, batch_size=1, seed=3)
        observed = [sampler.draw().symmetry_indices[0] for _ in range(8)]

        self.assertEqual(observed, [0, 1, 2, 3, 0, 1, 2, 3])

    def test_batch_stays_unique_when_it_crosses_a_shuffle_boundary(self) -> None:
        pool = self.samples[:5]
        sampler = ReplaySampler(pool, None, batch_size=4, seed=5)
        sampler.draw()
        crossed = sampler.draw()

        self.assertEqual(len({id(sample) for sample in crossed.samples}), 4)

    def test_replay_cap_is_an_optimizer_step_cap(self) -> None:
        self.assertEqual(
            step_limit_for_replay(
                batch_size=256,
                newest_positions=10_000,
                maximum_consumption=4,
            ),
            156,
        )
        with self.assertRaisesRegex(ValueError, "no complete"):
            step_limit_for_replay(
                batch_size=256,
                newest_positions=10,
                maximum_consumption=1,
            )

    def test_surprise_uses_visits_not_the_unchanged_search_prior(self) -> None:
        surprises = [_policy_surprise(sample) for sample in self.samples]
        self.assertTrue(any(value > 0 for value in surprises))

        source = SourceSplit(
            name="fresh",
            train=tuple(self.samples),
            validation=tuple(self.samples[:1]),
            train_games=1,
            validation_games=1,
            inputs=(),
        )
        weighted = _weight_training_surprise(source, strength=1.0, cap=3.0)
        weights = [sample.loss_weight for sample in weighted.train]
        self.assertAlmostEqual(sum(weights) / len(weights), 1.0)
        self.assertGreater(max(weights), min(weights))
        self.assertEqual(weighted.validation, source.validation)

    def test_phase_bands_are_board_scaled(self) -> None:
        names = [phase_name(sample) for sample in self.samples]
        self.assertEqual(names[0], "opening")
        self.assertTrue(set(names) <= {"opening", "middle", "late"})

    def test_latest_checkpoint_is_actor_even_when_validation_is_worse(self) -> None:
        def record(step: int, objective: float) -> dict:
            return {
                "step": step,
                "optimizer_step": 100 + step,
                "model": {"path": f"step-{step}.keras", "sha256": str(step)},
                "training_state": f"state-{step}",
                "validation": {"monitoring_objective": objective},
            }

        manifest = _actor_manifest(
            [record(0, 1.0), record(10, 0.5), record(20, 0.8)]
        )

        self.assertEqual(manifest["actor_key"], "latest")
        self.assertEqual(manifest["models"]["latest"]["run_step"], 20)
        self.assertEqual(manifest["models"]["latest"]["optimizer_step"], 120)
        self.assertEqual(manifest["models"]["best_validation"]["run_step"], 10)
        self.assertEqual(manifest["evaluation_role"], "diagnostic_only")
        self.assertNotIn("authorized", repr(manifest).lower())
        self.assertNotIn("candidate", repr(manifest).lower())

    def test_game_id_split_is_stable_when_replay_window_grows(self) -> None:
        first = [SimpleNamespace(seed=index) for index in range(100)]
        extended = [SimpleNamespace(seed=index) for index in range(200)]
        first_train, first_validation = _stable_split_indices(
            first, 0.2, seed=47
        )
        extended_train, extended_validation = _stable_split_indices(
            extended, 0.2, seed=47
        )

        self.assertEqual(first_train, [i for i in extended_train if i < 100])
        self.assertEqual(
            first_validation, [i for i in extended_validation if i < 100]
        )


if __name__ == "__main__":
    unittest.main()
