"""Manual regression checks only; these tests perform short PPO updates.

Run explicitly: python -m unittest tests.test_parallel -v
No test is run on import or by opening the notebook.
"""
import copy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from maternal.config import ExperimentConfig


class ThreadConfig(unittest.TestCase):
    def test_legacy_default_and_thread_rng_selection(self):
        config = ExperimentConfig()
        config.validate()
        self.assertEqual((config.family_workers, config.ppo_shuffle_rng), (1, "legacy"))
        config.family_workers = 2
        config.validate()
        self.assertEqual(config.ppo_shuffle_rng, "per_agent")
        self.assertEqual(ExperimentConfig.from_dict(config.to_dict()), config)
        for bad in (0, -1, 65, 1.5, True):
            config.family_workers = bad
            with self.assertRaises(ValueError):
                config.validate()


class ThreadedCare(unittest.TestCase):
    @staticmethod
    def config(workers):
        config = ExperimentConfig(device="cpu", population=2, generations=1,
                                  family_workers=workers, ppo_shuffle_rng="per_agent")
        config.world.days, config.world.steps_per_day = 1, 32
        config.ppo.rollout, config.ppo.batch_size, config.ppo.epochs = 16, 8, 1
        config.ppo.hidden = 8
        config.validate()
        return config

    def assert_same_families(self, left, right):
        import torch
        self.assertEqual(left.joint_steps, right.joint_steps)
        self.assertEqual(left.agent_transitions, right.agent_transitions)
        for first, second in zip(left.families, right.families):
            self.assertEqual(first.world.snapshot(), second.world.snapshot())
            self.assertEqual(first.world.summary(), second.world.summary())
            for a, b in zip((first.mother, first.child), (second.mother, second.child)):
                self.assertEqual((a.updates, a.transitions), (b.updates, b.transitions))
                self.assertEqual(a.shuffle_rng.bit_generator.state, b.shuffle_rng.bit_generator.state)
                self.assertTrue(torch.equal(a.generator.get_state(), b.generator.get_state()))
                for key, value in a.model.policy.state_dict().items():
                    self.assertTrue(torch.equal(value, b.model.policy.state_dict()[key]), key)

    def test_serial_and_threads_with_same_rng_and_exact_step_budget(self):
        import numpy as np
        from maternal.evolution import Experiment
        with TemporaryDirectory() as directory:
            serial = Experiment(self.config(1), Path(directory)/"serial")
            serial._new_generation(lambda: None)
            serial._care_serial(lambda frame: None, lambda: None)
            threaded = Experiment(self.config(2), Path(directory)/"threaded")
            threaded._new_generation(lambda: None)
            global_rng = copy.deepcopy(np.random.get_state())
            threaded._care_parallel(lambda frame: None, lambda: None)
            self.assertTrue(np.array_equal(global_rng[1], np.random.get_state()[1]))
            self.assertEqual(global_rng[2:], np.random.get_state()[2:])
            self.assertEqual(threaded.joint_steps, 64)
            self.assertEqual(threaded.agent_transitions, 128)
            self.assert_same_families(serial, threaded)

    def test_stop_at_barrier_checkpoint_and_resume(self):
        from maternal.evolution import Experiment, StopRequested
        with TemporaryDirectory() as directory:
            reference = Experiment(self.config(2), Path(directory)/"reference")
            reference._new_generation(lambda: None)
            reference._care_parallel(lambda frame: None, lambda: None)
            interrupted = Experiment(self.config(2), Path(directory)/"interrupted")
            interrupted._new_generation(lambda: None)

            def control():
                if interrupted.joint_steps >= 32:
                    raise StopRequested()

            with self.assertRaises(StopRequested):
                interrupted._care_parallel(lambda frame: None, control)
            self.assertEqual([f.world.t for f in interrupted.families], [16, 16])
            interrupted.save("interrupted.pt")
            resumed = Experiment.resume(interrupted.output/"models"/"interrupted.pt")
            resumed._care_parallel(lambda frame: None, lambda: None)
            self.assert_same_families(reference, resumed)

    def test_worker_failure_propagates_without_entering_assessment(self):
        from maternal.evolution import Experiment
        with TemporaryDirectory() as directory:
            engine = Experiment(self.config(2), Path(directory)/"failure")
            engine._new_generation(lambda: None)

            def fail(index):
                raise RuntimeError("worker failed")

            engine._family_chunk = fail
            with self.assertRaisesRegex(RuntimeError, "worker failed"):
                engine._care_parallel(lambda frame: None, lambda: None)
            self.assertEqual(engine.phase, "care")
            self.assertFalse((engine.output/"models"/"interrupted.pt").exists())


if __name__ == "__main__":
    unittest.main()
