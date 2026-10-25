"""Frozen probes and plotting. No evaluation or plotting happens on import."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import copy
import csv
import html
import json
import random

from .environment import Action, Body, FamilyWorld, MOVES, demo_action, reference_mother_action, distance, path_action
from .config import PROTOCOL

SCENARIOS = ("hungry", "afraid_not_hungry", "tired_silent", "blocked_route",
             "dangerous_request", "safe_refusal", "competing_work", "self_eating", "false_cry")
INTERVENTIONS = ("normal", "no_sound", "no_cues", "z_zero", "z_flip", "direct_cue_block", "self_feature_control")


def sample_unique(records, count, seed):
    from .agent import genome_hash
    groups = {}
    for index, (learner, _) in enumerate(records):
        groups.setdefault(genome_hash(learner.genome), []).append(index)
    rng = random.Random(seed)
    representatives = [rng.choice(group) for group in groups.values()]
    return rng.sample(representatives, min(count, len(representatives)))


def standardize_body(body, skills=None):
    result = copy.deepcopy(body)
    result.hp, result.energy, result.fatigue, result.cold, result.fear = 100, 70, 0, 0, 0
    result.food, result.signal = 0, "quiet"
    result.signal_until = result.consent_until = result.refused_until = 0
    result.last_move, result.death_step = -100, None
    result.death_cause = ""
    result.hunger_steps = result.exhaustion_steps = result.stress_steps = 0
    if skills is not None:
        result.forage, result.avoidance = skills
    return result


def fitness_probe(learner, body, config, control=lambda: None, on_step=lambda: None, as_adult=True):
    # Isolated readiness probe AFTER childhood, with adult abilities unlocked.
    # Probe adulthood separately; live childhood still follows its age gates.
    scores = []
    for seed in range(config.fitness_seeds):
        control()
        world = FamilyWorld(config.world, 100_000 + seed, learn_skills=False)
        world.mother.hp = 0
        world.child = standardize_body(body)
        world.child.role = "mother" if as_adult else "child"
        world.child.age_steps = config.world.horizon
        world.child.pos = (2, 2)
        world.child.energy, world.child.fatigue = 35, 80
        actor = learner.evaluation_copy(seed=400_000+seed)
        initial_food = world.child.food
        for _ in range(min(config.fitness_steps, config.world.horizon)):
            control()
            if world.child.alive:
                action = actor.act(world.observation(1)) if hasattr(actor, "act") else actor.choose(world.observation(1))[0]
            else:
                action = Action.WAIT
            world.step((Action.WAIT, action))
            on_step()
        gathered_and_ate = world.stats["child_gathers"] > 0 and world.stats["child_eats"] > initial_food
        score = 0.5 * int(gathered_and_ate) + 0.5 * int(world.stats["child_rests"] > 0)
        scores.append(score if world.child.alive else 0.0)
    return sum(scores)/len(scores)


EVALUATION_PROTOCOL = "full_lifetime_care_v4_growing_child"
CARE_POLICIES = ("learned", "no_help", "appropriate_help", "always_help")


def evaluation_config(payload):
    """Balanced physics must match the checkpoint; never migrate old world rules."""
    from .config import ExperimentConfig, resolve_device
    from .evolution import source_hashes
    current = source_hashes()
    keys = ("maternal/agent.py", "maternal/environment.py", "maternal/evolution.py")
    saved = payload.get("source_hashes", {})
    if payload.get("protocol") != PROTOCOL or not all(saved.get(k) == current.get(k) for k in keys):
        raise ValueError("Frozen evaluation requires matching policy, physics and inheritance code")
    config = ExperimentConfig.from_dict(payload["config"])
    config.device = "cpu" if config.execution_backend == "processes" else resolve_device()
    config.eval_steps = config.world.horizon
    return config


def scenario_world(config, scenario, seed, mother_body, test_split="monitor"):
    if scenario not in (*SCENARIOS, "ordinary", "mother_loss") or test_split not in ("monitor", "holdout"):
        raise ValueError("Unknown scenario or test split")
    world = FamilyWorld(config.world, (800_000 if test_split == "monitor" else 1_800_000) + seed, learn_skills=False)
    world.mother = standardize_body(mother_body)
    world.mother.role, world.mother.pos = "mother", (1, 2)
    world.mother.age_steps, world.mother.food = config.world.horizon, 1
    world.child = Body("benchmark-child", root_id="benchmark", x=2, y=2,
                       age_steps=0, forage=0.0, avoidance=0.0)
    # One kitchen/bed/desk/play area, as in training. Positions are held out.
    # Monitor uses the right side; final holdout reflects it to a distinct layout.
    variant = seed % 2
    world.walls = {(3, 1 if variant == 0 else 3)}
    world.hazards = {(6, 0 if variant == 0 else 4)}
    world.shelters = {(4, 4 if variant == 0 else 0)}
    world.food_sites = {(5, 2): 0}
    world.workstations = {(0, 0 if variant == 0 else 4)}
    world.play_areas = {(2, 4 if variant == 0 else 0)}
    if scenario == "blocked_route":
        world.walls = {(3, 1), (3, 2), (3, 3)}
    if test_split == "holdout":
        mirror = lambda pos: (6-pos[0], pos[1])
        world.mother.pos, world.child.pos = mirror(world.mother.pos), mirror(world.child.pos)
        for name in ("walls", "hazards", "shelters", "workstations", "play_areas"):
            setattr(world, name, {mirror(pos) for pos in getattr(world, name)})
        world.food_sites = {mirror(pos): ready for pos, ready in world.food_sites.items()}
    delayed = scenario in ("dangerous_request", "safe_refusal", "self_eating")
    # Short debugging profiles place the older-child event near their end. Real
    # pilot/main episodes use the late-child age; food dependence never ends here.
    onset = min(config.world.late_child_day*config.world.steps_per_day,
                max(0, config.world.horizon-config.world.steps_per_day)) if delayed else 0
    if scenario == "mother_loss":
        # Dedicated reference check only, not pooled into learned-care performance.
        onset = min(5*config.world.steps_per_day, max(0, config.world.horizon-config.world.steps_per_day))
    world.care_test = {"onset": onset, "end": min(config.world.horizon, onset+config.world.steps_per_day),
                       "triggered": False, "layout": variant, "scenario": scenario}
    apply_scenario_events(world, scenario)
    return world


def apply_scenario_events(world, scenario):
    """Fixed challenge schedule shared by evaluation and the frozen viewer."""
    meta, c = world.care_test, world.child
    if world.t >= meta["end"]:
        world.noise_pos, world.noise_signal = None, "quiet"
    if world.t != meta["onset"] or meta["triggered"]:
        return
    meta["triggered"] = True
    if not c.alive or not world.mother.alive:
        return  # No resurrection or free late meals after the caregiver has died.
    if scenario in ("hungry", "blocked_route", "competing_work"):
        c.energy, c.signal = 20, "cry"
        if scenario == "competing_work":
            world.mother.pos = next(iter(world.workstations))
    elif scenario == "afraid_not_hungry":
        c.energy, c.fear, c.signal = 95, 85, "help"
    elif scenario == "tired_silent":
        c.fatigue, c.signal = 90, "quiet"
    elif scenario == "dangerous_request":
        hazard = next(iter(world.hazards))
        choices = [(hazard[0]+dx, hazard[1]+dy) for dx, dy in ((-1,0),(1,0),(0,-1),(0,1))]
        c.pos = next(pos for pos in choices if world.passable(pos) and pos not in world.hazards)
        c.signal = "help"
        world.carried = False
    elif scenario == "safe_refusal":
        # Safe position and satisfied hunger: refusing an unnecessary pickup.
        choices = [(world.mother.x+dx, world.mother.y+dy) for dx,dy in ((1,0),(-1,0),(0,1),(0,-1))]
        choices = [pos for pos in choices if world.passable(pos) and pos not in world.hazards]
        if choices:
            c.pos = choices[0]
        c.energy, c.signal, c.refused_until = 70, "refuse", meta["end"]
        world.carried = False
    elif scenario == "self_eating":
        # One provided meal tests eating without unnecessary intervention.
        # Independent gathering still follows the same age/cooldown rules.
        c.pos, c.energy, c.food = next(iter(world.food_sites)), 45, 1
        world.carried = False
    elif scenario == "mother_loss":
        meta["mother_loss_injected"] = True
        world.mother.hp = 0
        world.mother.death_step, world.mother.death_cause = world.t, "scheduled_mother_loss_test"
        world.carried = False
        world.event("Reference check: caregiver removed; child physiology continues normally")
    elif scenario == "false_cry":
        c.energy = 95
        world.noise_pos, world.noise_signal = next(iter(world.workstations)), "cry"
    c.signal_until = world.t + world.config.signal_duration


def benchmark_child(world, scenario):
    """Same fixed partner for every phase/baseline; no imitation targets in training."""
    c, meta = world.child, world.care_test
    if not c.alive:
        return Action.WAIT
    active = meta["onset"] <= world.t < meta["end"]
    if active and scenario == "dangerous_request":
        if world.mother.alive and world.mother.signal == "warn" and distance(c.pos, world.mother.pos) <= world.config.hearing:
            return Action.WAIT
        return Action.CALL if world.t % 4 == 0 else path_action(world, c.pos, next(iter(world.hazards)))
    if active and scenario == "safe_refusal" and world.t % 20 == 0:
        return Action.REFUSE
    if active and scenario == "tired_silent" and c.fatigue > 60:
        # Silent initially; once carried to bed, it can actually benefit from care.
        return Action.REST if c.pos in world.shelters else Action.WAIT
    if (world.mother.alive and world.mother.signal == "offer" and distance(c.pos, world.mother.pos) <= 1
            and c.refused_until <= world.t and (c.fatigue > 70 or c.fear > 60)):
        return Action.ACCEPT
    if c.food and c.energy < 65 and world.skill_cap(c) > 0:
        return Action.EAT
    if c.energy < 45:
        if world.can_gather(c):
            target = min(world.food_sites, key=lambda pos: distance(c.pos, pos))
            return Action.GATHER if c.pos == target else path_action(world, c.pos, target)
        return Action.CALL
    if world.skill_cap(c) == 0:
        if c.fatigue > 60:
            return Action.REST
        if c.energy < 45:
            return Action.CALL
        return Action.ACCEPT
    if world.mother.alive and c.fear > 60 and distance(c.pos, world.mother.pos) <= 1:
        return Action.ACCEPT
    return demo_action(world, 1, "self_only")


def care_action(learner, world, intervention, care_policy):
    if not world.mother.alive:
        return Action.WAIT
    if care_policy == "no_help":
        return demo_action(world, 0, "self_only")
    if care_policy == "appropriate_help":
        return reference_mother_action(world)
    if care_policy == "always_help":
        return demo_action(world, 0, "always_help")
    observation = world.observation(0, intervention)
    action = (learner.act(observation, intervention=intervention) if hasattr(learner, "act")
              else learner.choose(observation, intervention=intervention)[0])
    return Action(action)


def run_scenario(learner, body, config, scenario, seed, intervention="normal", control=lambda: None,
                 progress=None, on_step=lambda: None, test_split="monitor", care_policy="learned"):
    if care_policy not in CARE_POLICIES:
        raise ValueError("Unknown care policy")
    world = scenario_world(config, scenario, seed, body, test_split)
    world.care_test["care_policy"] = care_policy
    first_help, first_action = None, ""
    initial = world.child.energy, world.child.fear, world.child.fatigue
    challenge_before, challenge_after = None, None
    counts = {key: 0 for key in ("appropriate_help_attempts", "unnecessary_help_attempts",
                                "unserved_need_steps", "help_opportunity_steps", "challenge_exposed",
                                "challenge_survived", "challenge_need_resolved", "challenge_need_resolved_day",
                                "child_visible_at_challenge", "child_signal_audible_at_challenge")}
    stages = {stage: {key: 0.0 for key in ("steps", "health_sum", "need_steps", "alive_steps", "solo_gathers")}
              for stage in ("early_child", "middle_child", "late_child")}
    for step in range(config.world.horizon):
        control()
        apply_scenario_events(world, scenario)
        c = world.child
        if step == world.care_test["onset"]:
            counts["challenge_exposed"] = int(c.alive and world.mother.alive)
            separation = distance(c.pos, world.mother.pos)
            counts["child_visible_at_challenge"] = int(counts["challenge_exposed"] and separation <= world.config.vision)
            counts["child_signal_audible_at_challenge"] = int(counts["challenge_exposed"] and separation <= world.config.hearing
                                                             and c.signal != "quiet")
            challenge_before = c.energy, c.fear, c.fatigue
        action = care_action(learner, world, intervention, care_policy)
        if step == 0:
            first_action = Action(action).name
        hungry, afraid, tired = c.energy < 30, c.fear > 60, c.fatigue > 70
        danger = c.pos in world.hazards
        needy = c.alive and (hungry or afraid or tired or danger)
        near = distance(c.pos, world.mother.pos) <= 1
        opportunity = needy and world.mother.alive and near
        counts["help_opportunity_steps"] += int(opportunity)
        social = action in (Action.FEED, Action.SOOTHE, Action.CARRY)
        can_eat_alone = world.skill_cap(c) > 0 and c.food > 0 and c.energy >= 20
        receptive = c.refused_until <= world.t and (world.skill_cap(c) == 0 or c.consent_until > world.t)
        suitable = ((action == Action.FEED and c.energy < 50 and world.mother.food > 0 and not can_eat_alone)
                    or (action == Action.SOOTHE and c.fear > 35 and receptive)
                    or (action == Action.CARRY and (danger or tired) and (danger or c.refused_until <= world.t)))
        ongoing_help = (world.carried and tired and (action in MOVES or action == Action.RELEASE))
        counts["appropriate_help_attempts"] += int(near and c.alive and suitable)
        counts["unnecessary_help_attempts"] += int(near and c.alive and social and not suitable)
        counts["unserved_need_steps"] += int(opportunity and not suitable and not ongoing_help and not can_eat_alone)
        before = world.stats["feeds"] + world.stats["food_provisions"] + world.stats["soothes"] + world.stats["forced_carries"]
        carried, gathers = world.carried, world.stats["child_solo_gathers"]
        # Calendar age continues after death, so later-stage means include failures.
        age = step/world.config.steps_per_day
        stage = "early_child" if age < config.world.early_child_days else (
            "middle_child" if age < config.world.late_child_day else "late_child")
        world.step((action, benchmark_child(world, scenario)))
        on_step()
        after = world.stats["feeds"] + world.stats["food_provisions"] + world.stats["soothes"] + world.stats["forced_carries"]
        if (step >= world.care_test["onset"] and first_help is None
                and (after > before or (world.carried and not carried))):
            first_help = step-world.care_test["onset"]+1
        if step+1 == min(config.world.horizon, world.care_test["onset"]+20):
            challenge_after = c.energy, c.fear, c.fatigue
            counts["challenge_survived"] = int(counts["challenge_exposed"] and c.alive)
            resolved = {"hungry": c.energy >= 30, "blocked_route": c.energy >= 30,
                        "competing_work": c.energy >= 30, "afraid_not_hungry": c.fear <= 60,
                        "tired_silent": c.fatigue <= 70}
            counts["challenge_need_resolved"] = (int(counts["challenge_survived"] and resolved[scenario])
                                                   if scenario in resolved else -1)
        if step+1 == world.care_test["end"]:
            targets = {"hungry": c.energy >= 30, "blocked_route": c.energy >= 30,
                       "competing_work": c.energy >= 30, "afraid_not_hungry": c.fear <= 60,
                       "tired_silent": c.fatigue <= 70}
            counts["challenge_need_resolved_day"] = (int(counts["challenge_exposed"] and c.alive and targets[scenario])
                                                       if scenario in targets else -1)
        entry = stages[stage]
        entry["steps"] += 1
        entry["health_sum"] += c.hp
        entry["alive_steps"] += int(c.alive)
        entry["need_steps"] += int(c.alive and (c.energy < 30 or c.fear > 60 or c.fatigue > 70))
        entry["solo_gathers"] += world.stats["child_solo_gathers"]-gathers
        if progress and (step % 20 == 0 or step+1 == config.world.horizon):
            progress(world)
    stage_results = {}
    for name, entry in stages.items():
        stage_results.update({name+"_"+key: value for key, value in entry.items()})
        stage_results[name+"_health_mean"] = entry["health_sum"]/entry["steps"] if entry["steps"] else -1
    challenge_before = challenge_before or initial
    challenge_after = challenge_after or (world.child.energy, world.child.fear, world.child.fatigue)
    return {"evaluation_protocol": EVALUATION_PROTOCOL, "world_protocol": PROTOCOL, "care_policy": care_policy,
            "scenario": scenario, "scenario_seed": seed, "intervention": intervention,
            "episode_days": config.world.days, "steps_per_day": config.world.steps_per_day,
            "initial_child_age_steps": 0, "challenge_start_step": world.care_test["onset"],
            "scheduled_mother_loss": int(scenario == "mother_loss"),
            "mother_loss_injected": int(world.care_test.get("mother_loss_injected", False)),
            "first_action": first_action, "first_help_step": first_help if first_help is not None else -1,
            "help_within_20": int(first_help is not None and first_help <= 20),
            "challenge_energy_change_20": challenge_after[0]-challenge_before[0],
            "challenge_fear_change_20": challenge_after[1]-challenge_before[1],
            "challenge_fatigue_change_20": challenge_after[2]-challenge_before[2],
            "energy_change": world.child.energy-initial[0], "fear_change": world.child.fear-initial[1],
            "fatigue_change": world.child.fatigue-initial[2],
            "both_alive": int(world.mother.alive and world.child.alive),
            **counts, **stage_results, **world.summary()}


def _evaluation_subject(learner, body, config, generation, phase, index, count, skill_regime, test_split):
    from .agent import genome_hash

    test_body = copy.deepcopy(body)
    if skill_regime == "matched":
        test_body.forage, test_body.avoidance = 0.5, 0.3
    elif phase == "A0":
        test_body.forage = test_body.avoidance = 0.0
    metadata = {"generation": generation, "phase": phase, "candidate": index,
                "genome_hash": genome_hash(learner.genome), "arm": config.arm,
                "sampled_candidates": count,
                "run_seed": config.seed, "test_split": test_split, "skill_regime": skill_regime,
                "weight_source": {"A0": "inherited_initial", "A1": "before_parenthood", "A2": "after_parenthood"}[phase],
                "memory_reset": True, "learning_enabled": False,
                "subject_agent_id": body.agent_id, "source_child_id": body.clone_source_id,
                "subject_origin": "repeated_mother" if config.arm == "D" else ("founder" if generation == 0 else "grown_child")}
    return test_body, metadata


def _evaluate_phase_processes(records, config, generation, phase, indices, jobs, skill_regime,
                              test_split, control, progress, on_step):
    import multiprocessing as mp
    import pickle
    from concurrent.futures import ProcessPoolExecutor
    from .config import available_cpu_count
    from .parallel import _evaluation_initializer, ordered_evaluation_results

    if not indices:
        return
    subjects = []
    for index in indices:
        control()
        learner, body = records[index]
        test_body, metadata = _evaluation_subject(learner, body, config, generation, phase, index,
                                                  len(indices), skill_regime, test_split)
        subjects.append({"index": index, "state": learner.state(include_buffer=False),
                         "body": test_body, "metadata": metadata})
    by_index = {subject["index"]: subject for subject in subjects}
    total = max(1, len(indices)*len(SCENARIOS)*config.eval_seeds*len(jobs))
    workers = min(config.family_workers, available_cpu_count(), total)
    context = mp.get_context("spawn")
    cancel = context.Event()
    executor = ProcessPoolExecutor(max_workers=workers, mp_context=context,
                                   initializer=_evaluation_initializer,
                                   initargs=(config.to_dict(), pickle.dumps(subjects), cancel))
    try:
        scores = {}
        for result in ordered_evaluation_results(executor, [("fitness", index) for index in indices], workers, control):
            scores[result["index"]] = result["score"]
            for _ in range(result["steps"]):
                on_step()
        tasks = (("scenario", index, scenario, seed, intervention, policy)
                 for index in indices for scenario in SCENARIOS for seed in range(config.eval_seeds)
                 for intervention, policy in jobs)
        for completed, result in enumerate(ordered_evaluation_results(executor, tasks, workers, control), 1):
            control()
            index = result["index"]
            subject, score = by_index[index], scores[index]
            for _ in range(result["steps"]):
                on_step()
            if progress:
                progress(result["world"], completed/total)
            yield {**subject["metadata"], "basic_selfcare_score": score, "adult_readiness_score": score,
                   "mother_forage": subject["body"].forage,
                   "mother_avoidance": subject["body"].avoidance, **result["result"]}
    finally:
        # Workers check this event every step. Cancelled/partial cases produce no rows.
        cancel.set()
        executor.shutdown(wait=True, cancel_futures=True)


def evaluate_phase(records, config, generation, phase, control=lambda: None, progress=None,
                   interventions=("normal",), sample_indices=None, skill_regime="matched", on_step=lambda: None,
                   test_split="monitor", care_policies=CARE_POLICIES, parallel=True):
    """Frozen, paired tests. Scripted baselines run once in A0/normal only."""
    if phase not in ("A0", "A1", "A2"):
        raise ValueError("Unknown evaluation phase")
    indices = sample_indices if sample_indices is not None else sample_unique(
        records, config.eval_candidates, config.seed+generation*7919+900_000)
    jobs = [(intervention, policy) for intervention in interventions for policy in care_policies
            if policy == "learned" or (phase == "A0" and intervention == "normal")]
    if parallel and config.parallel_zero_shot and config.execution_backend == "processes":
        yield from _evaluate_phase_processes(records, config, generation, phase, list(indices), jobs,
                                              skill_regime, test_split, control, progress, on_step)
        return
    total = max(1, len(indices)*len(SCENARIOS)*config.eval_seeds*len(jobs))
    completed = 0
    for index in indices:
        learner, body = records[index]
        test_body, metadata = _evaluation_subject(learner, body, config, generation, phase, index,
                                                  len(indices), skill_regime, test_split)
        control_actor = learner.evaluation_copy(naive=phase == "A0", seed=600_000)
        basic_score = fitness_probe(control_actor, test_body, config, control, on_step, as_adult=True)
        for scenario in SCENARIOS:
            for seed in range(config.eval_seeds):
                for intervention, policy in jobs:
                    control()
                    actor = learner.evaluation_copy(naive=phase == "A0", seed=500_000+seed)
                    result = run_scenario(actor, test_body, config, scenario, seed, intervention, control,
                                          (lambda w: progress(w, (completed+w.t/config.world.horizon)/total)) if progress else None,
                                          on_step, test_split, policy)
                    yield {**metadata,
                           "basic_selfcare_score": basic_score, "adult_readiness_score": basic_score,
                           "mother_forage": test_body.forage, "mother_avoidance": test_body.avoidance, **result}
                    completed += 1


def read_rows(path):
    if not Path(path).exists():
        return []
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def full_lifetime_rows(directory, split="monitor", skills="matched"):
    """Exclude old short probes, interventions and unfinished manual jobs."""
    directory = Path(directory)
    sources = [directory/"evaluations.csv"]
    for path in sorted(directory.glob("evaluations_full_lifetime_*.csv")):
        metadata = path.with_suffix(".json")
        if metadata.is_file() and json.loads(metadata.read_text(encoding="utf-8")).get("status") == "completed":
            sources.append(path)
    selected = {}
    config = json.loads((directory/"config.json").read_text(encoding="utf-8"))
    candidates = min(config["eval_candidates"], config["population"])
    for path in sources:
        rows = [r for r in read_rows(path) if r.get("evaluation_protocol") == EVALUATION_PROTOCOL
                and r["intervention"] == "normal" and r["test_split"] == split and r["skill_regime"] == skills
                and int(r["steps"]) == int(r["episode_days"])*int(r["steps_per_day"])]
        groups = {}
        for row in rows:
            groups.setdefault(int(row["generation"]), []).append(row)
        for generation, group in groups.items():
            expected = {"A0": set(CARE_POLICIES), "A1": {"learned"}, "A2": {"learned"}}
            pairs = {(r["candidate"], r["scenario_seed"]) for r in group}
            keys = {(r["phase"], r["care_policy"], r["candidate"], r["scenario_seed"], r["scenario"]) for r in group}
            # Unique-genome sampling may legitimately produce fewer candidates.
            count = int(group[0].get("sampled_candidates", candidates))
            complete = len(pairs) == count*config["eval_seeds"] and all(
                (phase, policy, candidate, seed, scenario) in keys
                for phase, policies in expected.items() for policy in policies
                for candidate, seed in pairs for scenario in SCENARIOS)
            if complete:
                selected[generation] = group
    return [r for generation in sorted(selected) for r in selected[generation]]


def a0_report(run_directories, destination, split="monitor", skills="matched"):
    """Average families within each run before bootstrapping independent run seeds."""
    import numpy as np
    import matplotlib.pyplot as plt
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    runs, identities, signature = [], set(), None
    for directory in run_directories:
        directory = Path(directory)
        config = json.loads((directory/"config.json").read_text(encoding="utf-8"))
        rows = full_lifetime_rows(directory, split, skills)
        if not rows:
            if len(run_directories) > 1:
                raise ValueError("Every selected run needs completed full-lifetime tests; do not pool legacy short probes")
            return []
        identity = config["arm"], config["seed"]
        if identity in identities:
            raise ValueError("Repeated arm/run seed is not independent")
        identities.add(identity)
        settings = (config["world"], config["population"], config["generations"])
        if signature is not None and settings != signature:
            raise ValueError("Training world, population and schedule must match")
        signature = settings
        runs.append((identity, [r for r in rows if r["phase"] == "A0"]))
    pages = (("child_alive", "mother_alive", "both_alive", "child_health_mean"),
             ("unnecessary_help_attempts", "safe_refusal_overrides", "early_child_health_mean",
              "middle_child_health_mean", "late_child_health_mean"),
             ("early_child_need_steps", "middle_child_need_steps", "late_child_need_steps",
              "challenge_need_resolved_day", "appropriate_help_attempts"),
             ("food_provisions", "child_supervised_gathers", "child_independent_gathers", "child_eats"))
    summaries, names = [], []
    rng = np.random.default_rng(20261024)
    for page, metrics in enumerate(pages):
        fig, axes = plt.subplots(1, len(metrics), figsize=(4*len(metrics), 3.8), constrained_layout=True)
        for axis, metric in zip(axes, metrics):
            for arm in sorted({identity[0] for identity, _ in runs}):
                for policy, style in zip(CARE_POLICIES, ("-", "--", "-.", ":")):
                    generations = sorted({int(r["generation"]) for identity, rows in runs if identity[0] == arm for r in rows})
                    x, means, lows, highs = [], [], [], []
                    for generation in generations:
                        values = []
                        for identity, rows in runs:
                            if identity[0] != arm:
                                continue
                            trials = [float(r[metric]) for r in rows if int(r["generation"]) == generation
                                      and r["care_policy"] == policy and float(r[metric]) >= 0]
                            if trials:
                                values.append(sum(trials)/len(trials))
                        if not values:
                            continue
                        mean = sum(values)/len(values)
                        low = high = mean
                        if len(values) >= 2:
                            draws = rng.choice(values, size=(2000, len(values)), replace=True).mean(axis=1)
                            low, high = np.quantile(draws, (.025, .975))
                        x.append(generation); means.append(mean); lows.append(low); highs.append(high)
                        summaries.append({"arm": arm, "generation": generation, "care_policy": policy,
                            "metric": metric, "mean": mean, "independent_runs": len(values),
                            "low_95": float(low) if len(values) >= 2 else "",
                            "high_95": float(high) if len(values) >= 2 else "",
                            "test_split": split, "skills": skills, "evaluation_protocol": EVALUATION_PROTOCOL})
                    line, = axis.plot(x, means, style, label=f"{arm} / {policy}")
                    if len(runs) > 1:
                        axis.fill_between(x, lows, highs, color=line.get_color(), alpha=.08)
            axis.set(title=metric.replace("_", " "), xlabel="Generation / round (D)")
            axis.grid(axis="y", alpha=.15)
        axes[0].legend(fontsize=7, frameon=False)
        fig.suptitle(f"A0 inherited weights / {split} / {skills}; independent-run intervals")
        name = f"a0_{split}_{skills}_{page+1}.png"
        fig.savefig(destination/name, dpi=150)
        plt.close(fig)
        names.append(name)
    with (destination/f"a0_{split}_{skills}_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader(); writer.writerows(summaries)
    # Scenario-level results keep hunger, fear and silence from hiding in one mean.
    scenario_groups = {}
    scenario_metrics = ("child_alive", "mother_alive", "both_alive", "child_health_mean",
                        "challenge_need_resolved_day", "challenge_energy_change_20",
                        "challenge_fear_change_20", "challenge_fatigue_change_20",
                        "unnecessary_help_attempts", "safe_refusal_overrides", "child_damage_hazard")
    for identity, rows in runs:
        local = {}
        for row in rows:
            key = int(row["generation"]), row["scenario"], row["care_policy"]
            local.setdefault(key, []).append(row)
        for (generation, scenario, policy), trials in local.items():
            for metric in scenario_metrics:
                values = [float(r[metric]) for r in trials
                          if not (metric == "challenge_need_resolved_day" and float(r[metric]) < 0)]
                if values:
                    scenario_groups.setdefault((identity[0], generation, scenario, policy, metric), []).append(sum(values)/len(values))
    scenario_summary = [{"arm": key[0], "generation": key[1], "scenario": key[2],
                         "care_policy": key[3], "metric": key[4], "mean": sum(values)/len(values),
                         "independent_runs": len(values), "test_split": split, "skills": skills}
                        for key, values in sorted(scenario_groups.items())]
    with (destination/f"a0_{split}_{skills}_by_scenario.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scenario_summary[0]))
        writer.writeheader(); writer.writerows(scenario_summary)
    return names


def plot_run(run_directory):
    """Explicit post-processing command; never called after training automatically."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    run = Path(run_directory).resolve()
    rows = read_rows(run/"metrics.csv")
    evaluations = read_rows(run/"evaluations.csv")
    if not rows:
        raise ValueError("No generation metrics found in this run")
    destination = run/"plots"
    destination.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
                         "axes.spines.right": False, "axes.facecolor": "#fafbf9",
                         "figure.facecolor": "#fafbf9", "axes.labelcolor": "#33433e"})
    by_generation = {}
    for row in rows:
        by_generation.setdefault(int(row["generation"]), []).append(row)
    gens = sorted(by_generation)
    def avg(g, key):
        return sum(float(r[key]) for r in by_generation[g])/len(by_generation[g])
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), constrained_layout=True)
    for axis, key, title, color in zip(axes,
            ("child_alive", "child_health_mean", "solo_score"),
            ("Child survival", "Child health · full horizon", "Adult readiness (frozen)"),
            ("#3f8071", "#648d9b", "#b28c66")):
        axis.plot(gens, [avg(g, key) for g in gens], color=color, linewidth=2)
        axis.set(title=title, xlabel="Generation / training round")
        axis.grid(axis="y", alpha=0.15)
    fig.savefig(destination/"development.png", dpi=170)
    plt.close(fig)
    image_names = ["development.png"]
    development = read_rows(run/"development.csv")
    if development:
        days = sorted({int(row["day"]) for row in development})
        grouped = {d: [r for r in development if int(r["day"]) == d] for d in days}
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)
        for key, label, color in (("child_forage", "Food handling skill", "#477e6b"), ("child_avoidance", "Avoidance skill", "#7796a7")):
            axes[0].plot(days, [sum(float(r[key]) for r in grouped[d])/len(grouped[d]) for d in days], label=label, color=color)
        feeds = [sum(float(r["feeds"]) for r in grouped[d])/len(grouped[d]) for d in days]
        carries = [sum(float(r["forced_carries"]) for r in grouped[d])/len(grouped[d]) for d in days]
        axes[1].plot(days, feeds, label="Feeds (cumulative)", color="#477e6b")
        axes[1].plot(days, carries, label="Forced carries (cumulative)", color="#c18f6c")
        for axis, title in zip(axes, ("Development by age", "Care actions · descriptive counts")):
            axis.set(title=title, xlabel="Child day")
            axis.grid(axis="y", alpha=0.15)
            axis.legend(frameon=False)
        fig.savefig(destination/"care_by_age.png", dpi=170)
        plt.close(fig)
        image_names.append("care_by_age.png")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)
    scatter = axes[0].scatter([float(r["help_energy"]) for r in rows], [float(r["child_health_mean"]) for r in rows],
                            c=[int(r["generation"]) for r in rows], cmap="summer_r", s=18, alpha=0.65)
    axes[0].set(title="Care allocation and child health", xlabel="Food energy transferred", ylabel="Mean child HP")
    fig.colorbar(scatter, ax=axes[0], label="Generation / round")
    axes[1].plot(gens, [len({r["child_genome"] for r in by_generation[g]}) for g in gens], color="#7796a7", label="Unique child genomes")
    selection_file = run/"events.jsonl"
    if selection_file.exists():
        events = [json.loads(line) for line in selection_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        selected = [event for event in events if event.get("type") == "selection"]
        axes[1].plot([e["generation"] for e in selected], [e["unique_sources"] for e in selected], color="#c18f6c", label="Unique selected sources")
    axes[1].set(title="Population diversity", xlabel="Generation / round")
    axes[1].legend(frameon=False)
    fig.savefig(destination/"allocation_diversity.png", dpi=170)
    plt.close(fig)
    image_names.append("allocation_diversity.png")
    if "mother_work_steps" in rows[0]:
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
        axes[0].plot(gens, [avg(g, "mother_work_steps") for g in gens], color="#c18f6c", label="Work steps")
        axes[0].plot(gens, [avg(g, "work_while_child_needy") for g in gens], color="#7796a7", label="Work during child need")
        axes[0].set(title="Work allocation", xlabel="Generation", ylabel="Steps per family")
        axes[0].legend(frameon=False)
        axes[1].scatter([float(r["mother_work_steps"]) for r in rows],
                        [float(r["child_health_mean"]) for r in rows], s=18, alpha=.6, color="#c18f6c")
        axes[1].set(title="Work and child wellbeing", xlabel="Mother work steps", ylabel="Mean child HP")
        for cause, color in (("hunger", "#c18f6c"), ("exhaustion", "#b19b6e"), ("stress", "#7796a7"), ("hazard", "#477e6b")):
            axes[2].plot(gens, [avg(g, "child_damage_"+cause) for g in gens], label=cause, color=color)
        axes[2].set(title="Child damage by cause", xlabel="Generation", ylabel="HP lost per family")
        axes[2].legend(frameon=False)
        fig.savefig(destination/"work_and_needs.png", dpi=170)
        plt.close(fig)
        image_names.append("work_and_needs.png")
    if development and "child_stress" in development[0]:
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
        for axis, key, title in zip(axes, ("child_energy", "child_fatigue", "child_stress"),
                                    ("Child satiety", "Child fatigue", "Child stress")):
            axis.plot(days, [sum(float(r[key]) for r in grouped[d])/len(grouped[d]) for d in days], color="#c18f6c")
            axis.set(title=title, xlabel="Child age / day", ylim=(0,100))
        fig.savefig(destination/"needs_by_age.png", dpi=170)
        plt.close(fig)
        image_names.append("needs_by_age.png")
    if evaluations:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
        for phase, color in (("A0", "#3f8071"), ("A1", "#698eaa"), ("A2", "#c28f73")):
            for axis, key, title in zip(axes, ("child_health_mean", "help_within_20"),
                                       ("Frozen test · child health", "Response within 20 steps (descriptive)")):
                data = {}
                for row in evaluations:
                    if row["phase"] == phase and row["intervention"] == "normal" and row.get("care_policy", "learned") == "learned":
                        data.setdefault(int(row["generation"]), []).append(float(row[key]))
                x = sorted(data)
                axis.plot(x, [sum(data[g])/len(data[g]) for g in x], label=phase, color=color)
                axis.set(title=title, xlabel="Generation / training round")
                axis.grid(axis="y", alpha=0.15)
        axes[0].legend(frameon=False)
        fig.savefig(destination/"zero_shot.png", dpi=170)
        plt.close(fig)
        image_names.append("zero_shot.png")
        import numpy as np
        scenario_names = list(dict.fromkeys(row["scenario"] for row in evaluations))
        matrix = np.full((len(scenario_names), len(gens)), np.nan)
        for i, scenario in enumerate(scenario_names):
            for j, gen in enumerate(gens):
                values = [float(r["child_health_mean"]) for r in evaluations if r["phase"] == "A0" and r["scenario"] == scenario
                          and int(r["generation"]) == gen and r["intervention"] == "normal" and r.get("care_policy", "learned") == "learned"]
                if values:
                    matrix[i, j] = sum(values)/len(values)
        fig, axis = plt.subplots(figsize=(max(7, len(gens)*0.12), 4.5), constrained_layout=True)
        picture = axis.imshow(matrix, aspect="auto", cmap="YlGn", vmin=0, vmax=100)
        axis.set_yticks(range(len(scenario_names)), labels=[s.replace("_", " ") for s in scenario_names])
        ticks = sorted(set((0, len(gens)//2, len(gens)-1)))
        axis.set_xticks(ticks, labels=[gens[i] for i in ticks])
        axis.set(title="Genome-only · child health by scenario", xlabel="Generation / round")
        fig.colorbar(picture, ax=axis, label="Mean child HP")
        fig.savefig(destination/"scenario_heatmap.png", dpi=170)
        plt.close(fig)
        image_names.append("scenario_heatmap.png")
    if evaluations and "fatigue_change" in evaluations[0]:
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
        cases = (("hungry", "energy_change", 1, "Hungry infant / satiety change"),
                 ("afraid_not_hungry", "fear_change", -1, "Afraid child / stress reduction"),
                 ("tired_silent", "fatigue_change", -1, "Tired child / fatigue reduction"))
        for axis, (scenario, metric, sign, title) in zip(axes, cases):
            for phase, color in (("A0", "#477e6b"), ("A1", "#7796a7"), ("A2", "#c18f6c")):
                data = {}
                for row in evaluations:
                    if row["scenario"] == scenario and row["phase"] == phase and row["intervention"] == "normal" and row.get("care_policy", "learned") == "learned":
                        data.setdefault(int(row["generation"]), []).append(sign*float(row[metric]))
                x = sorted(data)
                axis.plot(x, [sum(data[g])/len(data[g]) for g in x], label=phase, color=color)
            axis.set(title=title, xlabel="Generation")
            axis.axhline(0, color="#cbb9a5", linewidth=1)
        axes[0].legend(frameon=False)
        fig.savefig(destination/"need_responses.png", dpi=170)
        plt.close(fig)
        image_names.append("need_responses.png")
    extra_rows = [r for path in sorted(run.glob("evaluations_extra_*.csv")) for r in read_rows(path)]
    if extra_rows:
        def pair_key(r):
            return tuple(r[k] for k in ("generation", "phase", "candidate", "scenario", "scenario_seed", "skill_regime", "test_split")) + (r.get("evaluation_protocol", "legacy_200"), r.get("steps", ""), r.get("care_policy", "learned"))
        normals = {pair_key(r): float(r["child_health_mean"]) for r in evaluations+extra_rows if r["intervention"] == "normal"}
        deltas = {}
        for row in extra_rows:
            key = pair_key(row)
            if row["intervention"] != "normal" and key in normals:
                deltas.setdefault(row["intervention"], []).append(float(row["child_health_mean"])-normals[key])
        if deltas:
            names = list(deltas)
            fig, axis = plt.subplots(figsize=(9, 3.5), constrained_layout=True)
            axis.barh([s.replace("_", " ") for s in names], [sum(deltas[s])/len(deltas[s]) for s in names], color="#7796a7")
            axis.axvline(0, color="#b4bfb8", linewidth=1)
            axis.set(title="Matched interventions · descriptive differences", xlabel="Child mean HP minus normal")
            fig.savefig(destination/"interventions.png", dpi=170)
            plt.close(fig)
            image_names.append("interventions.png")
    config = json.loads((run/"config.json").read_text(encoding="utf-8"))
    primary_names = []
    for split in ("monitor", "holdout"):
        for skills in ("matched", "native"):
            primary_names.extend(a0_report([run], destination, split, skills))
    primary = "".join(f'<img src="plots/{name}" alt="{html.escape(name)}">' for name in primary_names)
    body = ("<h2>Primary evidence: full-lifetime A0</h2>" +
            (primary if primary else "<p>No completed full-lifetime A0 tests yet. Legacy short probes cannot answer this question.</p>") +
            "<p>Learned policy vs scripted no-help / appropriate-help / help-priority baselines. Attempts are descriptive proxies, not proof of appropriate care. "
            "Intervals resample independent run seeds; a single run has no interval. Missing/extinct generations are not imputed; see CSV run counts. "
            "Stage health includes deaths as zero; short profiles may not reach later stages. D is repeated training, not biological generations.</p>" +
            "<h2>Supplementary training and recorded frozen diagnostics</h2>" +
            "".join(f'<img src="plots/{name}" alt="{html.escape(name)}">' for name in image_names))
    report = ("<!doctype html><meta charset='utf-8'><title>Maternal · run report</title>"
              "<style>body{max-width:1100px;margin:56px auto;padding:0 28px;background:#fafbf9;"
              "color:#283f36;font:16px/1.7 system-ui}img{width:100%;margin:20px 0}small{color:#68786f}</style>"
              f"<h1>Maternal · run report</h1><p>Arm {html.escape(config['arm'])} · seed {config['seed']}</p>"
              "<small>Single-run descriptive plots. No confidence intervals or claim of instinct. "
              "Frequent helping is not necessarily appropriate helping.</small>"+body)
    (run/"report.html").write_text(report, encoding="utf-8")
    return run/"report.html"


def compare_runs(run_directories, output):
    """Bootstrap independent runs, never individual family rows. Explicit command only."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    groups, identities = {}, set()
    for directory in run_directories:
        directory = Path(directory).resolve()
        config = json.loads((directory/"config.json").read_text(encoding="utf-8"))
        manifest = json.loads((directory/"manifest.json").read_text(encoding="utf-8"))
        if manifest["status"] not in ("completed", "extinct"):
            raise ValueError(f"Comparison requires completed/extinct runs, not interrupted runs: {directory}")
        key = config["arm"], config["seed"]
        if key in identities:
            raise ValueError(f"Repeated arm/seed {key}; do not count repeated identical runs as independent")
        identities.add(key)
        rows = read_rows(directory/"metrics.csv")
        if not rows:
            raise ValueError(f"No metrics: {directory}")
        groups.setdefault(config["arm"], []).append((config, manifest, rows, str(directory)))
    if not groups:
        raise ValueError("Provide at least one run directory")
    first_config = next(iter(groups.values()))[0][0]
    available = [bool(full_lifetime_rows(path)) for path in run_directories]
    if any(available) and not all(available):
        raise ValueError("Some selected runs lack full-lifetime monitor/matched tests. Evaluate them before comparing A0.")
    for runs in groups.values():
        for config, manifest, rows, _ in runs:
            if manifest.get("protocol") != PROTOCOL:
                raise ValueError("Compare only runs using the current balanced-world protocol; old results remain separate")
            if config["world"] != first_config["world"] or config["generations"] != first_config["generations"] or config["population"] != first_config["population"]:
                raise ValueError("Run schedules, world settings and populations must match for this comparison")
    destination = Path(output).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    generations = first_config["generations"]
    rng = np.random.default_rng(20261022)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    metrics = ("child_alive", "child_health_mean", "solo_score")
    titles = ("Child survival", "Child health · population outcome", "Adult readiness (frozen)")
    colors = {"A": "#477e6b", "B": "#b19b6e", "C": "#7796a7", "D": "#c18f6c"}
    summary = []
    for arm, runs in sorted(groups.items()):
        for axis, metric, title in zip(axes, metrics, titles):
            matrix = np.zeros((len(runs), generations), dtype=float)
            for index, (config, manifest, rows, _) in enumerate(runs):
                for generation in range(generations):
                    values = [float(r[metric]) for r in rows if int(r["generation"]) == generation]
                    if values:
                        matrix[index, generation] = sum(values)/len(values)
                    elif manifest["status"] != "extinct":
                        raise ValueError("Completed run is missing generation metrics")
                    # Extinct future populations have zero outcome; never discard failed runs.
            mean = matrix.mean(axis=0)
            indices = rng.integers(0, len(runs), size=(2000, len(runs)))
            draws = matrix[indices].mean(axis=1)
            low, high = np.quantile(draws, (0.025, 0.975), axis=0)
            x = np.arange(generations)
            axis.plot(x, mean, color=colors[arm], label=f"{arm} · n={len(runs)}")
            if len(runs) >= 2:
                axis.fill_between(x, low, high, color=colors[arm], alpha=0.13)
            axis.set(title=title, xlabel="Generation / training round")
            axis.grid(axis="y", alpha=0.15)
            for generation in range(generations):
                summary.append({"arm": arm, "metric": metric, "generation": generation,
                                "independent_runs": len(runs), "mean": float(mean[generation]),
                                "low_95": float(low[generation]) if len(runs) >= 2 else "",
                                "high_95": float(high[generation]) if len(runs) >= 2 else ""})
    axes[0].legend(frameon=False)
    fig.savefig(destination/"comparison.png", dpi=170)
    plt.close(fig)
    with (destination/"comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    provenance = {"method": "2000 bootstrap resamples at independent-run level; percentile 95% intervals",
                  "extinction": "future population outcomes = 0; failed runs retained",
                  "interpretation": "Descriptive intervals; not a significance test. D is not an evolutionary generation.",
                  "runs": [item[3] for runs in groups.values() for item in runs]}
    (destination/"comparison_manifest.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    if all(available):
        images = a0_report(run_directories, destination)
        (destination/"a0_comparison_notes.txt").write_text(
            "Primary: A0 full-lifetime monitor/matched, normal condition. Families/scenarios are averaged within run first; "
            "95% bootstrap intervals use independent run seeds. CSV reports n at each generation. "
            "Missing/extinct generations are not filled in for genome evaluation. "
            "comparison.png is supplementary training evidence, not proof of instinct. "
            "No-help, appropriate-help and always-help are scripted references, not trained arms.", encoding="utf-8")
        return destination/images[0]
    return destination/"comparison.png"
