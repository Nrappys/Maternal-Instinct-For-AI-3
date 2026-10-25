"""Regression checks provided for manual execution; not run by the editor."""
import ast
import copy
import json
from pathlib import Path
import unittest

from maternal.config import ExperimentConfig
from maternal.environment import Action, Body
from maternal.evaluation import (SCENARIOS, scenario_world, benchmark_child,
                                 apply_scenario_events, run_scenario, full_lifetime_rows)


class FullLifetimeEvaluation(unittest.TestCase):
    def test_every_profile_uses_training_lifetime_and_legacy_budget_is_normalized(self):
        for mode in ("pilot", "main", "smoke"):
            config = ExperimentConfig.profile(mode)
            self.assertEqual(config.eval_steps, config.world.horizon)
            saved = config.to_dict()
            saved["eval_steps"] = 200
            self.assertEqual(ExperimentConfig.from_dict(saved).eval_steps, config.world.horizon)

    def test_all_scenarios_start_at_birth_and_holdout_reflects_facilities(self):
        config = ExperimentConfig.profile()
        body = Body("subject", role="mother", food=2)
        original = copy.deepcopy(body)
        for scenario in SCENARIOS:
            for seed in range(config.eval_seeds):
                monitor = scenario_world(config, scenario, seed, body)
                holdout = scenario_world(config, scenario, seed, body, "holdout")
                self.assertEqual(monitor.child.age_steps, 0)
                self.assertEqual(monitor.child.forage, 0)
                self.assertEqual(len(monitor.food_sites), 1)
                self.assertNotIn(next(iter(monitor.food_sites)), {(1, 1), (1, 3)})
                self.assertFalse(monitor.learn_skills)
                for name in ("walls", "hazards", "shelters", "workstations", "play_areas"):
                    self.assertEqual(getattr(holdout, name), {(6-x, y) for x, y in getattr(monitor, name)})
        self.assertEqual(body, original)

    def test_tired_silent_child_can_benefit_from_being_put_to_bed(self):
        world = scenario_world(ExperimentConfig.profile(), "tired_silent", 0, Body("mother"))
        self.assertEqual(benchmark_child(world, "tired_silent"), Action.WAIT)
        world.child.pos = next(iter(world.shelters))
        self.assertEqual(benchmark_child(world, "tired_silent"), Action.REST)

    def test_older_child_challenge_is_delayed_and_does_not_revive_dead_child(self):
        config = ExperimentConfig.profile()
        world = scenario_world(config, "self_eating", 0, Body("mother"))
        self.assertEqual(world.child.food, 0)
        self.assertEqual(world.care_test["onset"], config.world.late_child_day*config.world.steps_per_day)
        world.t = world.care_test["onset"]
        world.child.hp = 0
        apply_scenario_events(world, "self_eating")
        self.assertEqual(world.child.hp, 0)
        self.assertEqual(world.child.food, 0)

    def test_no_help_never_calls_the_learned_policy_and_covers_full_clock(self):
        config = ExperimentConfig.profile("smoke")
        config.world.days, config.world.steps_per_day = 2, 3
        config.validate()
        class NeverCalled:
            def choose(self, *args, **kwargs):
                raise AssertionError("No-help baseline must not use learned policy")
        result = run_scenario(NeverCalled(), Body("mother"), config, "hungry", 0, care_policy="no_help")
        self.assertEqual(result["steps"], 6)
        self.assertEqual(result["initial_child_age_steps"], 0)
        self.assertEqual(result["feeds"], 0)
        self.assertEqual(result["care_policy"], "no_help")

    def test_evaluation_calls_no_learning_methods(self):
        source = Path(__file__).resolve().parents[1]/"maternal/evaluation.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                self.assertNotIn(node.func.attr, {"learn", "train", "record", "backward", "fit"})

    def test_untreated_severe_hunger_can_be_fatal_before_independence(self):
        import math
        world = ExperimentConfig.profile().world
        untreated_steps = world.neglect_grace + math.ceil(100/world.starvation_damage)
        self.assertLess(untreated_steps, world.early_child_days*world.steps_per_day)
        # A fresh, initially healthy infant has over one day before hunger damage.
        self.assertGreater((70-world.starvation_threshold)/world.infant_drain,
                           world.steps_per_day)

    def test_age_gates_share_the_same_configured_boundaries(self):
        config = ExperimentConfig.profile()
        world = scenario_world(config, "ordinary", 0, Body("mother"))
        world.mother.signal, world.mother.signal_until = "offer", 20
        for age, cap in ((config.world.early_child_days-1, 0),
                         (config.world.early_child_days, .6), (config.world.late_child_day, 1)):
            world.child.age_steps = age*config.world.steps_per_day
            self.assertEqual(world.skill_cap(world.child), cap)
            self.assertEqual(world.can_gather(world.child), age >= config.world.early_child_days)

    def test_appropriate_reference_provisions_food_and_does_not_feed_capable_child(self):
        from maternal.environment import reference_mother_action
        config = ExperimentConfig.profile()
        world = scenario_world(config, "hungry", 0, Body("mother"))
        world.mother.food = 0
        world.mother.pos = next(iter(world.food_sites))
        self.assertEqual(reference_mother_action(world), Action.GATHER)
        world.child.age_steps = config.world.late_child_day*config.world.steps_per_day
        world.child.food, world.child.energy = 1, 45
        world.mother.food = 2
        self.assertNotIn(reference_mother_action(world), (Action.FEED, Action.CARRY))

    def test_reference_asks_before_non_emergency_carry(self):
        from maternal.environment import reference_mother_action
        config = ExperimentConfig.profile()
        world = scenario_world(config, "ordinary", 0, Body("mother"))
        world.child.age_steps = config.world.late_child_day*config.world.steps_per_day
        world.child.fatigue = 80
        self.assertEqual(reference_mother_action(world), Action.OFFER)
        world.mother.signal = "offer"
        self.assertEqual(benchmark_child(world, "ordinary"), Action.ACCEPT)

    def test_gather_requires_supervision_before_late_childhood(self):
        config = ExperimentConfig.profile()
        world = scenario_world(config, "ordinary", 0, Body("mother"))
        world.child.age_steps = config.world.early_child_days*config.world.steps_per_day
        world.child.pos = next(iter(world.food_sites))
        world.mother.pos = (0, 0)
        world.step((Action.WAIT, Action.GATHER))
        self.assertEqual(world.child.food, 0)
        world.mother.pos = world.child.pos
        self.assertFalse(world.can_gather(world.child))
        world.step((Action.OFFER, Action.WAIT))
        world.step((Action.WAIT, Action.GATHER))
        self.assertEqual(world.child.food, 1)
        self.assertEqual(world.stats["child_supervised_gathers"], 1)
        world.child.age_steps = config.world.late_child_day*config.world.steps_per_day
        world.child.last_gather = -1000
        world.mother.hp = 0
        world.step((Action.WAIT, Action.GATHER))
        self.assertEqual(world.child.food, 2)
        self.assertEqual(world.stats["child_independent_gathers"], 1)

    def test_both_die_without_food_and_child_does_not_die_instantly_with_mother(self):
        config = ExperimentConfig.profile()
        world = scenario_world(config, "ordinary", 0, Body("mother"))
        world.mother.pos = world.child.pos = (0, 0)
        world.mother.energy, world.mother.hp = 0, .01
        world.mother.hunger_steps = config.world.neglect_grace
        world.step((Action.WAIT, Action.WAIT))
        self.assertFalse(world.mother.alive)
        self.assertTrue(world.child.alive)
        for _ in range(config.world.horizon-1):
            world.step((Action.WAIT, Action.GATHER))
        self.assertFalse(world.child.alive)
        self.assertEqual(world.child.food, 0)
        self.assertGreater(world.child.death_step, world.mother.death_step)
        self.assertEqual(world.summary()["child_died_after_mother"], 1)

    def test_late_event_cannot_give_free_food_after_mother_death(self):
        config = ExperimentConfig.profile()
        world = scenario_world(config, "self_eating", 0, Body("mother"))
        world.mother.hp = 0
        world.t = world.care_test["onset"]
        world.child.energy = 10
        apply_scenario_events(world, "self_eating")
        self.assertEqual(world.child.food, 0)
        self.assertEqual(world.child.energy, 10)

    def test_infant_cannot_eat_and_older_child_eats_provided_food(self):
        from maternal.environment import FamilyWorld
        world = FamilyWorld()
        world.child.food = 1
        world.step((Action.WAIT, Action.EAT))
        self.assertEqual(world.child.food, 1)
        self.assertEqual(world.stats["child_eats"], 0)
        world.child.food = 0
        world.child.age_steps = world.config.early_child_days*world.config.steps_per_day
        world.child.energy = 40
        world.mother.food = 1
        world.step((Action.FEED, Action.WAIT))
        self.assertEqual(world.child.food, 1)
        self.assertEqual(world.mother.food, 0)
        self.assertEqual(world.stats["food_provisions"], 1)
        self.assertLess(world.child.energy, 40)
        world.step((Action.WAIT, Action.EAT))
        self.assertEqual(world.child.food, 0)
        self.assertGreater(world.child.energy, 60)

    def test_juvenile_gather_cooldown_and_late_maturation(self):
        from maternal.environment import FamilyWorld
        world = FamilyWorld()
        world.child.age_steps = world.config.early_child_days*world.config.steps_per_day
        world.child.pos = world.mother.pos = next(iter(world.food_sites))
        world.step((Action.OFFER, Action.WAIT))
        world.step((Action.WAIT, Action.GATHER))
        world.step((Action.WAIT, Action.GATHER))
        self.assertEqual(world.child.food, 1)
        world.t = world.child.last_gather + world.config.juvenile_gather_interval
        world.step((Action.OFFER, Action.WAIT))
        world.step((Action.WAIT, Action.GATHER))
        self.assertEqual(world.child.food, 2)
        world.child.age_steps = world.config.late_child_day*world.config.steps_per_day
        self.assertEqual(world.gather_interval(world.child), 20)
        world.child.age_steps = world.config.horizon
        self.assertEqual(world.gather_interval(world.child), 4)

    def test_new_outputs_and_protocol_are_distinct_from_old_runs(self):
        from maternal.config import ROOT, OUTPUT_ROOT, PROTOCOL
        from maternal.evaluation import EVALUATION_PROTOCOL
        self.assertEqual(OUTPUT_ROOT, ROOT/"outputs"/"2_model")
        self.assertEqual(PROTOCOL, "maternal-house-v2.4-care-dependent-growth")
        self.assertEqual(EVALUATION_PROTOCOL, "full_lifetime_care_v5_care_dependent_growth")

    def test_child_moves_slowly_by_age_while_mother_speed_is_unchanged(self):
        from maternal.environment import FamilyWorld
        world = FamilyWorld()
        for age, expected in ((0, 5), (7, 3), (14, 1)):
            world.child.age_steps = age*world.config.steps_per_day
            self.assertEqual(world.move_interval(world.child), expected)
        self.assertEqual(world.move_interval(world.mother), 1)
        world.child.age_steps = 0
        world.step((Action.WAIT, Action.NORTH))
        first = world.child.pos
        for _ in range(4):
            world.step((Action.WAIT, Action.NORTH))
            self.assertEqual(world.child.pos, first)
        world.step((Action.WAIT, Action.NORTH))
        self.assertNotEqual(world.child.pos, first)

    def test_no_help_does_not_unlock_supervised_food_by_presence_alone(self):
        from maternal.environment import FamilyWorld
        from maternal.evaluation import care_action
        world = FamilyWorld()
        world.child.age_steps = 7*world.config.steps_per_day
        world.child.pos = world.mother.pos = next(iter(world.food_sites))
        self.assertFalse(world.can_gather(world.child))
        world.step((care_action(None, world, "normal", "no_help"), Action.GATHER))
        self.assertEqual(world.child.food, 0)

    def test_light_help_is_intermittent_and_not_a_learned_model(self):
        from unittest.mock import patch
        from maternal.environment import FamilyWorld
        from maternal.evaluation import care_action, CARE_POLICIES
        self.assertIn("light_help", CARE_POLICIES)
        world = FamilyWorld()
        with patch("maternal.evaluation.reference_mother_action", return_value=Action.FEED), \
             patch("maternal.evaluation.demo_action", return_value=Action.REST):
            for step, expected in ((0, Action.FEED), (49, Action.FEED), (50, Action.REST),
                                   (799, Action.REST), (800, Action.FEED)):
                world.t = step
                self.assertEqual(care_action(None, world, "normal", "light_help"), expected)

    def test_scheduled_mother_loss_is_a_separate_reference_check(self):
        self.assertNotIn("mother_loss", SCENARIOS)
        config = ExperimentConfig.profile()
        world = scenario_world(config, "mother_loss", 0, Body("mother"))
        before = world.child.hp
        world.t = 5*config.world.steps_per_day
        apply_scenario_events(world, "mother_loss")
        self.assertFalse(world.mother.alive)
        self.assertEqual(world.child.hp, before)
        self.assertTrue(world.care_test["mother_loss_injected"])


if __name__ == "__main__":
    unittest.main()
