"""Manual-only spawn/CPU regression checks, including short PPO updates.

Run explicitly: python -m unittest tests.test_processes -v
"""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest

from maternal.config import ExperimentConfig, recommended_process_workers
from tests.test_parallel import ThreadedCare


class ProcessConfig(unittest.TestCase):
    def test_155h_starting_point_respects_smaller_cpu_and_population_limits(self):
        with patch("maternal.config.available_cpu_count", return_value=22):
            self.assertEqual(recommended_process_workers(16), 4)
            self.assertEqual(recommended_process_workers(2), 2)
        with patch("maternal.config.available_cpu_count", return_value=2):
            self.assertEqual(recommended_process_workers(16), 1)
        config = ExperimentConfig.profile("pilot")
        self.assertEqual(config.execution_backend, "processes")
        self.assertEqual(config.ppo_shuffle_rng, "per_agent")
        self.assertEqual(ExperimentConfig.from_dict(config.to_dict()), config)


class ProcessCare(unittest.TestCase):
    @staticmethod
    def config():
        config = ThreadedCare.config(2)
        config.execution_backend = "processes"
        config.process_chunk_steps = 8
        config.validate()
        return config

    def test_spawn_matches_serial_cpu_with_same_per_agent_rng(self):
        from maternal.evolution import Experiment
        with TemporaryDirectory() as directory:
            config = self.config()
            config.execution_backend = "serial"
            config.family_workers = 1
            serial = Experiment(config, Path(directory)/"serial")
            serial._new_generation(lambda: None)
            serial._care_serial(lambda frame: None, lambda: None)
            processes = Experiment(self.config(), Path(directory)/"processes")
            processes._new_generation(lambda: None)
            frames = []
            processes._care_processes(frames.append, lambda: None)
            self.assertEqual(processes.config.device, "cpu")
            self.assertEqual(processes.joint_steps, 64)
            self.assertEqual(processes.agent_transitions, 128)
            self.assertTrue(frames)
            self.assertEqual(frames[-1]["world"]["step"], 32)
            ThreadedCare.assert_same_families(self, serial, processes)

    def test_stop_preserves_partial_rollouts_and_resume_matches_continuous(self):
        from maternal.evolution import Experiment, StopRequested
        with TemporaryDirectory() as directory:
            reference = Experiment(self.config(), Path(directory)/"reference")
            reference._new_generation(lambda: None)
            reference._care_processes(lambda frame: None, lambda: None)
            stopped = Experiment(self.config(), Path(directory)/"stopped")
            stopped._new_generation(lambda: None)

            def control():
                if stopped.joint_steps >= 16:
                    raise StopRequested()

            with self.assertRaises(StopRequested):
                stopped._care_processes(lambda frame: None, control)
            for family in stopped.families:
                self.assertEqual(family.world.t, 8)
                self.assertEqual(family.mother.model.rollout_buffer.pos, 8)
                self.assertEqual(family.child.model.rollout_buffer.pos, 8)
            stopped.save("interrupted.pt")
            resumed = Experiment.resume(stopped.output/"models"/"interrupted.pt")
            self.assertEqual(resumed.config.execution_backend, "processes")
            resumed._care_processes(lambda frame: None, lambda: None)
            ThreadedCare.assert_same_families(self, reference, resumed)

    def test_frozen_policies_do_not_gain_ppo_updates_in_workers(self):
        from maternal.evolution import Experiment
        with TemporaryDirectory() as directory:
            config = self.config()
            config.arm = "C"
            engine = Experiment(config, Path(directory)/"frozen")
            engine._new_generation(lambda: None)
            engine._care_processes(lambda frame: None, lambda: None)
            for family in engine.families:
                self.assertEqual(family.mother.updates, 0)
                self.assertEqual(family.child.updates, 0)

    def test_d_reuses_every_mother_and_optimizer_across_rounds(self):
        import copy
        import torch
        from maternal.evolution import Experiment

        with TemporaryDirectory() as directory:
            config = self.config()
            config.arm = "D"
            engine = Experiment(config, Path(directory)/"d_reuse")
            engine._new_generation(lambda: None)
            mothers = [family.mother for family in engine.families]
            states = [mother.state(False) for mother in mothers]
            engine.next_adults = [{"learner": state, "body": copy.deepcopy(family.world.mother)}
                                  for state, family in zip(states, engine.families)]
            engine.generation, engine.phase = 1, "initialize"
            engine._new_generation(lambda: None)
            for index, family in enumerate(engine.families):
                self.assertIs(family.mother, mothers[index])
                self.assertTrue(torch.equal(family.mother.generator.get_state(), states[index]["generator"]))
                self.assertIs(family.mother.model.policy.optimizer, mothers[index].model.policy.optimizer)
                self.assertEqual(family.mother.updates, states[index]["updates"])
                self.assertEqual(family.world.mother.hp, 100)


if __name__ == "__main__":
    unittest.main()
