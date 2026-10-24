"""Meaningful invariants. Run manually: python -m unittest discover -s tests -v

These tests are provided but have not been executed as part of creation.
"""
from __future__ import annotations

import copy
import importlib.util
import random
import unittest

from maternal.config import ExperimentConfig, WorldConfig
from maternal.environment import Action, Body, FamilyWorld, OBS_SIZE


class WorldInvariants(unittest.TestCase):
    def test_joint_clock_increments_once_and_observations_keep_schema(self):
        world = FamilyWorld(seed=15)
        ages = world.mother.age_steps, world.child.age_steps
        world.step((Action.WAIT, Action.WAIT))
        self.assertEqual(world.t, 1)
        self.assertEqual(world.mother.age_steps, ages[0]+1)
        self.assertEqual(world.child.age_steps, ages[1]+1)
        self.assertEqual(len(world.observation(0)), OBS_SIZE)
        self.assertEqual(len(world.observation(1)), OBS_SIZE)

    def test_dead_child_never_recovers_and_mother_can_continue(self):
        world = FamilyWorld()
        world.child.hp = 0
        world.child.energy = 100
        for _ in range(10):
            rewards, done = world.step((Action.REST, Action.REST))
            self.assertEqual(world.child.hp, 0)
            self.assertEqual(rewards[1], 0)
            self.assertFalse(done)
        self.assertEqual(world.t, 10)

    def test_food_is_conserved_and_feeding_has_no_direct_maternal_bonus(self):
        feed = FamilyWorld(seed=1)
        feed.child.energy = 15
        feed.mother.food = 1
        waiting = copy.deepcopy(feed)
        reward_feed, _ = feed.step((Action.FEED, Action.WAIT))
        reward_wait, _ = waiting.step((Action.WAIT, Action.WAIT))
        self.assertEqual(feed.mother.food, 0)
        self.assertGreater(feed.child.energy, waiting.child.energy)
        self.assertEqual(reward_feed[0], reward_wait[0])
        feed.step((Action.FEED, Action.WAIT))
        self.assertEqual(feed.mother.food, 0)
        self.assertEqual(feed.stats["feeds"], 1)

    def test_skill_freeze_is_real_and_infant_cannot_gather(self):
        world = FamilyWorld(seed=2, learn_skills=False)
        world.child.age_steps = world.config.horizon
        world.child.pos = next(iter(world.food_sites))
        previous = world.child.forage
        world.step((Action.WAIT, Action.GATHER))
        self.assertEqual(world.child.forage, previous)
        infant = FamilyWorld(seed=2)
        infant.child.pos = next(iter(infant.food_sites))
        for _ in range(30):
            infant.step((Action.WAIT, Action.GATHER))
        self.assertEqual(infant.child.food, 0)

    def test_generated_maps_are_connected_and_do_not_spawn_in_danger(self):
        for seed in range(30):
            world = FamilyWorld(seed=seed)
            self.assertEqual(len(world.reachable(world.mother.pos)), 35-len(world.walls))
            self.assertNotIn(world.mother.pos, world.hazards)
            self.assertNotIn(world.child.pos, world.hazards)

    def test_worlds_do_not_share_mutable_bodies(self):
        body = Body("candidate", hp=47)
        first = FamilyWorld(mother=body)
        second = FamilyWorld(mother=body)
        first.mother.hp = 12
        self.assertEqual(second.mother.hp, 47)
        self.assertEqual(body.hp, 47)

    def test_final_step_does_not_allow_time_to_overrun(self):
        world = FamilyWorld()
        world.t = world.config.horizon-1
        _, done = world.step((Action.WAIT, Action.WAIT))
        self.assertTrue(done)
        with self.assertRaises(RuntimeError):
            world.step((Action.WAIT, Action.WAIT))

    def test_work_requires_desk_and_is_mother_only(self):
        world = FamilyWorld(seed=3)
        before = world.mother.fatigue
        world.mother.pos = next(iter(world.workstations))
        world.child.pos = world.mother.pos
        rewards, _ = world.step((Action.WORK, Action.WORK))
        self.assertEqual(world.stats["mother_work_steps"], 1)
        self.assertAlmostEqual(world.stats["mother_work_reward"], world.config.work_reward)
        self.assertGreater(world.mother.fatigue, before)
        self.assertEqual(world.work_bonus, [world.config.work_reward, 0])
        world.mother.pos = (0, 0)
        world.step((Action.WORK, Action.WAIT))
        self.assertEqual(world.stats["mother_work_steps"], 1)
        self.assertEqual(world.work_bonus, [0, 0])

    def test_home_food_is_always_stocked_and_has_no_random_gather_failure(self):
        world = FamilyWorld(seed=4)
        world.mother.pos = next(iter(world.food_sites))
        world.mother.food = 0
        for _ in range(2):
            world.step((Action.GATHER, Action.WAIT))
        self.assertEqual(world.mother.food, 2)
        world.step((Action.EAT, Action.WAIT))
        world.step((Action.GATHER, Action.WAIT))
        self.assertEqual(world.mother.food, 2)
        self.assertEqual(world.food_sites[world.mother.pos], 0)

    def test_unmet_needs_have_grace_and_log_actual_damage_cause(self):
        world = FamilyWorld(seed=5)
        world.child.pos = (0, 0)
        world.child.energy, world.child.hp = 0, .01
        world.child.hunger_steps = world.config.neglect_grace - 1
        world.step((Action.WAIT, Action.WAIT))
        self.assertTrue(world.child.alive)
        world.step((Action.WAIT, Action.WAIT))
        self.assertFalse(world.child.alive)
        self.assertEqual(world.child.death_cause, "hunger")
        self.assertAlmostEqual(world.stats["child_damage_hunger"], .01)
        self.assertEqual(world.stats["child_damage_hazard"], 0)

    def test_new_features_are_visible_and_social_block_keeps_station_channels(self):
        from maternal.environment import CUE_SLICE, LOCAL_CHANNELS
        world = FamilyWorld(seed=6)
        world.mother.pos = next(iter(world.workstations))
        obs = world.observation(0)
        center = CUE_SLICE.stop + 12*LOCAL_CHANNELS
        self.assertEqual(obs[center+5], 1)
        self.assertEqual(len(obs), OBS_SIZE)
        self.assertEqual(world.child.cold, 0)


HAS_ML = all(importlib.util.find_spec(name) is not None for name in ("torch", "numpy", "stable_baselines3", "gymnasium"))


@unittest.skipUnless(HAS_ML, "Install ML dependencies manually before learner invariant tests")
class LearningInvariants(unittest.TestCase):
    def config(self):
        config = ExperimentConfig.profile("smoke")
        config.ppo.rollout = 32
        config.ppo.batch_size = 16
        config.ppo.epochs = 1
        return config

    def test_ppo_changes_lifetime_weights_but_never_germline_or_phi(self):
        import torch
        from maternal.agent import Learner, actor_state, genome_hash
        learner = Learner(self.config(), 10)
        before_genome = genome_hash(learner.genome)
        before_actor = actor_state(learner.model.policy)
        before_phi = copy.deepcopy(learner.phi.state_dict())
        world = FamilyWorld()
        for _ in range(32):
            action, transition = learner.choose(world.observation(0))
            rewards, _ = world.step((action, Action.WAIT))
            learner.record(transition, rewards[0], world.observation(0), terminal=False)
        self.assertEqual(learner.updates, 1)
        self.assertEqual(genome_hash(learner.genome), before_genome)
        self.assertTrue(all(torch.equal(before_phi[k], v) for k, v in learner.phi.state_dict().items()))
        self.assertTrue(any(not torch.equal(before_actor[k], v) for k, v in actor_state(learner.model.policy).items()))

    def test_newborn_uses_genome_not_mothers_learned_policy(self):
        import torch
        from maternal.agent import Learner, actor_state, mutate
        mother = Learner(self.config(), 11)
        with torch.no_grad():
            mother.model.policy.action_net.weight.add_(1.0)
        child = Learner(self.config(), 12, mutate(mother.genome, 13, 0.0))
        state = actor_state(child.model.policy)
        self.assertTrue(all(torch.equal(state[k], v) for k, v in mother.genome["actor"].items()))
        self.assertFalse(torch.equal(child.model.policy.action_net.weight, mother.model.policy.action_net.weight))
        self.assertTrue((child.z == 0).all())
        self.assertEqual(child.model.rollout_buffer.pos, 0)
        self.assertEqual(child.model.policy.optimizer.state_dict()["state"], {})

    def test_single_death_transition_and_partial_boundary_are_not_discarded(self):
        from maternal.agent import Learner, genome_hash
        for terminal in (False, True):
            learner = Learner(self.config(), 91)
            obs = FamilyWorld(seed=7).observation(0)
            genome = genome_hash(learner.genome)
            _, transition = learner.choose(obs)
            learner.record(transition, -5 if terminal else .01, obs, terminal, flush=True)
            self.assertEqual(learner.updates, 1)
            self.assertEqual(learner.model.rollout_buffer.pos, 0)
            learner.end_context()
            self.assertEqual(learner.discarded, 0)
            self.assertEqual(genome_hash(learner.genome), genome)

    def test_evaluation_does_not_consume_training_rng_or_weights(self):
        import numpy as np
        import torch
        from maternal.agent import Learner
        learner = Learner(self.config(), 15)
        genome = copy.deepcopy(learner.genome)
        py, np_state, th = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
        own_rng = learner.generator.get_state().clone()
        actor = learner.evaluation_copy(seed=16)
        actor.choose(FamilyWorld().observation(0))
        self.assertEqual(random.getstate(), py)
        self.assertTrue(np.array_equal(np.random.get_state()[1], np_state[1]))
        self.assertTrue(torch.equal(torch.get_rng_state(), th))
        self.assertTrue(torch.equal(learner.generator.get_state(), own_rng))
        self.assertTrue(all(torch.equal(genome["actor"][k], v) for k, v in learner.genome["actor"].items()))

    def test_restored_branch_has_independent_policy_and_z(self):
        import torch
        from maternal.agent import Learner
        original = Learner(self.config(), 22)
        state = original.state()
        first = Learner.restore(self.config(), state)
        second = Learner.restore(self.config(), state)
        first.z[0] = 0.8
        with torch.no_grad():
            first.model.policy.action_net.bias.add_(2)
        self.assertEqual(second.z[0], 0)
        self.assertTrue(torch.equal(second.model.policy.action_net.bias, original.model.policy.action_net.bias))

    def test_checkpoint_restores_next_joint_action_and_world_transition(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from maternal.evolution import Experiment
        def one_step(engine):
            family = engine.families[0]
            world = family.world
            choices = [learner.choose(world.observation(i)) for i, learner in enumerate((family.mother, family.child))]
            rewards, _ = world.step((choices[0][0], choices[1][0]))
            for i, learner in enumerate((family.mother, family.child)):
                learner.record(choices[i][1], rewards[i], world.observation(i), terminal=False)
            engine.joint_steps += 1
            return [a[0] for a in choices], world.snapshot()
        with TemporaryDirectory() as directory:
            original = Experiment(self.config(), Path(directory)/"run")
            original._new_generation(lambda: None)
            one_step(original)
            original.save("interrupted.pt")
            restored = Experiment.resume(original.output/"models"/"interrupted.pt")
            self.assertEqual(restored.families[0].mother.model.rollout_buffer.pos, 1)
            self.assertEqual(one_step(original), one_step(restored))

    def test_population_branching_preserves_genome_but_reseeds_action_streams(self):
        import torch
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from maternal.agent import genome_hash
        from maternal.evolution import Experiment
        with TemporaryDirectory() as directory:
            engine = Experiment(self.config(), Path(directory)/"run")
            engine._new_generation(lambda: None)
            source = engine.families[0]
            adult = {"learner": source.child.state(False), "body": copy.deepcopy(source.world.child)}
            engine.next_adults = [copy.deepcopy(adult), copy.deepcopy(adult)]
            engine.generation = 1
            engine.phase = "initialize"
            engine._new_generation(lambda: None)
            first, second = engine.families[0].mother, engine.families[1].mother
            self.assertEqual(genome_hash(first.genome), genome_hash(second.genome))
            self.assertFalse(torch.equal(first.generator.get_state(), second.generator.get_state()))


if __name__ == "__main__":
    unittest.main()
