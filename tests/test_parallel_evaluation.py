"""Manual-only frozen evaluation checks. No tests run on import.

Explicit command: python -m unittest tests.test_parallel_evaluation -v
"""
import copy
import unittest

from maternal.config import ExperimentConfig
from maternal.environment import Body


class ParallelEvaluation(unittest.TestCase):
    @staticmethod
    def config():
        config = ExperimentConfig(device="cpu", execution_backend="processes", family_workers=2,
                                  eval_candidates=1, eval_seeds=1, fitness_seeds=1, fitness_steps=4)
        config.world.days, config.world.steps_per_day = 1, 8
        config.ppo.hidden = 8
        config.validate()
        return config

    def test_actor_only_actions_and_rng_match_full_choose_without_critic(self):
        import torch
        from unittest.mock import patch
        from maternal.agent import Learner
        from maternal.environment import FamilyWorld

        config = self.config()
        full = Learner(config, 14)
        fast = Learner.restore(config, full.state(False))
        observation = FamilyWorld(config.world, 15).observation(0)
        with patch.object(fast.model.policy, "predict_values", side_effect=AssertionError("Unused critic called")):
            for intervention in ("normal", "z_zero", "z_flip", "direct_cue_block", "normal"):
                self.assertEqual(full.choose(observation, intervention)[0], fast.act(observation, intervention))
                self.assertTrue((full.z == fast.z).all())
                self.assertTrue(torch.equal(full.generator.get_state(), fast.generator.get_state()))

    def test_reused_frozen_actor_resets_to_fresh_copy_for_every_case(self):
        import torch
        from maternal.agent import Learner
        from maternal.environment import FamilyWorld

        config = self.config()
        source = Learner(config, 16)
        with torch.no_grad():
            source.model.policy.action_net.bias.add_(0.5)
        observation = FamilyWorld(config.world, 17).observation(0)
        with self.assertRaises(RuntimeError):
            source.reset_evaluation_context(18)
        for naive in (True, False):
            reused = source.evaluation_copy(naive=naive, seed=18)
            for seed in (18, 19, 18):
                fresh = source.evaluation_copy(naive=naive, seed=seed)
                reused.reset_evaluation_context(seed)
                for _ in range(5):
                    self.assertEqual(fresh.act(observation), reused.act(observation))
                    self.assertTrue((fresh.z == reused.z).all())
                self.assertEqual(reused.updates, 0)
                self.assertEqual(reused.transitions, 0)

    def test_serial_and_process_rows_order_step_counts_and_frozen_parent_match(self):
        import torch
        from maternal.agent import Learner
        from maternal.evaluation import evaluate_phase, SCENARIOS

        config = self.config()
        learner = Learner(config, 41)
        records = [(learner, Body("subject", role="mother"))]
        original = learner.state(False)
        original_body = copy.deepcopy(records[0][1])
        counts = [0, 0]

        def count(slot):
            counts[slot] += 1

        serial = list(evaluate_phase(records, config, 0, "A0", sample_indices=[0], parallel=False,
                                     on_step=lambda: count(0)))
        frames = []
        parallel = list(evaluate_phase(records, config, 0, "A0", sample_indices=[0],
                                       progress=lambda world, fraction: frames.append((world.t, fraction)),
                                       on_step=lambda: count(1)))
        self.assertEqual(serial, parallel)
        self.assertEqual(len(parallel), len(SCENARIOS) * 4)
        self.assertEqual(counts, [len(parallel) * 8 + 4] * 2)
        self.assertEqual(frames[-1], (8, 1.0))
        self.assertTrue(all(row["memory_reset"] and not row["learning_enabled"] for row in parallel))
        self.assertEqual(records[0][1], original_body)
        self.assertEqual(learner.updates, original["updates"])
        self.assertEqual(learner.shuffle_rng.bit_generator.state, original["shuffle_rng"])
        self.assertTrue(torch.equal(learner.generator.get_state(), original["generator"]))
        for name, tensor in learner.model.policy.state_dict().items():
            self.assertTrue(torch.equal(tensor, original["policy"][name]))

    def test_stop_yields_no_extra_rows_and_propagates(self):
        from maternal.agent import Learner
        from maternal.evaluation import evaluate_phase
        config = self.config()
        rows = []

        class StopTest(Exception):
            pass

        def control():
            if rows:
                raise StopTest()

        with self.assertRaises(StopTest):
            for row in evaluate_phase([(Learner(config, 42), Body("subject"))], config, 1, "A2",
                                      sample_indices=[0], control=control):
                rows.append(row)
        self.assertEqual(len(rows), 1)


if __name__ == "__main__":
    unittest.main()
