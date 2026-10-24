"""Population training, explicit branching, checkpoints, and one joint collector."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import copy
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import random
import sys

import numpy as np
import torch

from .agent import Learner, genome_hash, mutate
from .config import ExperimentConfig, PROTOCOL, ROOT, OUTPUT_ROOT, resolve_device
from .environment import Action, Body, FamilyWorld
from .evaluation import evaluate_phase, fitness_probe, sample_unique


class StopRequested(Exception):
    pass


@dataclass
class Family:
    world: FamilyWorld
    mother: Learner
    child: Learner


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def append_json(path, value):
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, default=json_default)+"\n")


def append_csv(path, row):
    path = Path(path)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def source_hashes():
    paths = list(ROOT.glob("*.py")) + list((ROOT/"maternal").glob("*.py")) + list((ROOT/"tests").glob("*.py"))
    # Git may change CRLF/LF across Windows and Linux; code identity stays the same.
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            for p in sorted(paths)}


def load_checkpoint(path):
    """Only load local checkpoints created by this project, never untrusted .pt files."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("schema") != 1 or payload.get("protocol") != PROTOCOL:
        raise ValueError("Checkpoint uses different world rules. Old unbalanced models cannot resume or be pooled with balanced runs; keep their recorded results separate.")
    return payload


class Experiment:
    def __init__(self, config: ExperimentConfig, output=None):
        config.validate()
        config.device = resolve_device(config.device)
        self.config = config
        self.rng = random.Random(config.seed)
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.set_num_threads(1)
        stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
        self.output = Path(output) if output else OUTPUT_ROOT/f"run_{stamp}_{config.arm}_{config.seed}"
        self.output.mkdir(parents=True, exist_ok=False)
        (self.output/"models").mkdir()
        (self.output/"config.json").write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
        versions = {}
        for name in ("torch", "numpy", "stable-baselines3", "gymnasium", "matplotlib", "ipywidgets", "ipykernel", "jupyterlab"):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = "not installed"
        self.manifest = {"protocol": PROTOCOL, "python": sys.version, "packages": versions,
                         "source_hashes": source_hashes(), "created_utc": datetime.now(timezone.utc).isoformat(),
                         "status": "initialized", "validated": False, "training_started": False,
                         "runtime_device": config.device,
                         "entrypoint": "Jupyter Notebook", "project": "Maternal Instinct For AI #3"}
        self._manifest()
        self.generation, self.family_index = 0, 0
        self.phase = "initialize"
        self.families: list[Family] = []
        self.starts: list[dict] = []
        self.start_bodies: list[Body] = []
        self.next_adults = None
        self.joint_steps = 0
        self.agent_transitions = 0
        self.fitness_steps = 0
        self.evaluation_steps = 0
        self.seen_eval = set()
        self.seen_metrics = set()
        self.watched_family = 0
        self.unique_candidates = config.population
        self.last_checkpoint = ""
        self.completed_generations = 0
        self.build_index = 0
        self.active_eval_phase = 0

    def _manifest(self, **updates):
        self.manifest.update(updates)
        (self.output/"manifest.json").write_text(json.dumps(self.manifest, indent=2), encoding="utf-8")

    def _seed(self):
        return self.rng.randrange(2**31-1)

    def _new_generation(self, control):
        if self.build_index == 0:
            self.families, self.starts = [], []
            self.start_bodies = []
        for slot in range(self.build_index, self.config.population):
            control()
            if self.next_adults is None:
                m = Learner(self.config, self._seed())
                body = Body(f"g0-m{slot}", root_id=f"root-{slot}", role="mother",
                            age_steps=self.config.world.horizon, forage=0.5, avoidance=0.3, food=1)
            else:
                adult = self.next_adults[slot]
                m = Learner.restore(self.config, adult["learner"])
                if self.config.arm != "D":
                    # Branches retain experience, but must not share identical action RNG streams.
                    m.generator.manual_seed(self._seed() ^ 0x4711)
                body = copy.deepcopy(adult["body"])
                source_id = body.agent_id
                body.agent_id = f"g{self.generation}-m{slot}"
                body.clone_source_id = source_id
                body.role = "mother"
                body.generation = self.generation
                # Consent/carry are relations to a former partner, not inherited physiology.
                body.consent_until = body.refused_until = body.signal_until = 0
                body.signal = "quiet"
                body.last_move = -100
                m.end_context()
                if self.config.arm == "D":
                    body.hp, body.energy, body.fatigue, body.cold, body.fear = 100, 70, 0, 0, 0
                    body.death_step, body.death_cause = None, ""
                    body.hunger_steps = body.exhaustion_steps = body.stress_steps = 0
                    body.age_steps = self.config.world.horizon
                    body.food = 1
                    m.z[:] = 0
            genome = m.genome
            if self.config.arm != "D":
                genome = mutate(genome, self._seed(), self.config.mutation_scale)
            c = Learner(self.config, self._seed(), genome)
            child_body = Body(f"g{self.generation+1}-c{slot}", parent_id=body.agent_id,
                              root_id=body.root_id, generation=self.generation+1)
            world = FamilyWorld(self.config.world, self._seed(), body, child_body)
            self.families.append(Family(world, m, c))
            self.starts.append(m.state(include_buffer=False))
            self.start_bodies.append(copy.deepcopy(world.mother))
            append_json(self.output/"lineage.jsonl", {
                "generation": self.generation, "family": slot, "mother": asdict(body),
                "child": asdict(child_body), "mother_genome": genome_hash(m.genome),
                "child_genome": genome_hash(c.genome)})
            self.build_index = slot+1
        self.family_index = 0
        self.build_index = 0
        self.phase = "care"

    def _progress(self, callback, world=None, label="Learning", fraction=0):
        selected = min(self.watched_family, len(self.families)-1)
        if world is None and self.families:
            world = self.families[max(0, selected)].world
        completed_care = self.joint_steps/(self.config.population*self.config.world.horizon)
        within = min(1, max(0, completed_care-self.generation))
        if self.phase == "care":
            within *= 0.80
        elif self.phase == "assessment":
            within = 0.80+0.05*fraction
        elif self.phase == "evaluation":
            within = 0.85+0.14*(self.active_eval_phase+fraction)/3
        elif self.phase == "selection":
            within = 0.99
        else:
            within = 0
        overall = min(1, (self.generation+within)/self.config.generations)
        if self.phase == "finished":
            overall = self.completed_generations/self.config.generations
        payload = {"kind": "frame", "phase": self.phase, "label": label,
                   "generation": self.generation, "generations": self.config.generations,
                   "family": self.family_index, "population": self.config.population,
                   "overall": overall, "phase_fraction": fraction,
                   "joint_steps": self.joint_steps, "agent_transitions": self.agent_transitions,
                   "output": str(self.output), "arm": self.config.arm,
                   "checkpoint": self.last_checkpoint, "unique": self.unique_candidates}
        if world is not None:
            payload["world"] = world.snapshot()
            if self.families and world is self.families[max(0, selected)].world:
                family = self.families[max(0, selected)]
                payload["z_mother"], payload["z_child"] = family.mother.z.tolist(), family.child.z.tolist()
        callback(payload)

    def run(self, callback=lambda frame: None, control=lambda: None):
        if self.phase == "finished":
            status = "extinct" if not self.candidates and self.config.arm != "D" else "completed"
            self._manifest(status=status)
            callback({"kind": "done", "status": status, "output": str(self.output)})
            return
        self._manifest(status="running", training_started=True)
        try:
            while self.generation < self.config.generations and self.phase != "finished":
                control()
                if self.phase == "initialize":
                    self._progress(callback, label="Building independent agents")
                    self._new_generation(control)
                if self.phase == "care":
                    for index in range(self.family_index, len(self.families)):
                        self.family_index = index
                        self.watched_family = index
                        family = self.families[index]
                        world = family.world
                        while world.t < self.config.world.horizon:
                            control()
                            observations = [world.observation(0), world.observation(1)]
                            bodies = (world.mother, world.child)
                            living = [body.alive for body in bodies]
                            choices = [learner.choose(obs) if alive else (int(Action.WAIT), None)
                                       for learner, obs, alive in zip((family.mother, family.child), observations, living)]
                            rewards, boundary = world.step((choices[0][0], choices[1][0]))
                            for who, learner in enumerate((family.mother, family.child)):
                                if living[who]:
                                    terminal = not bodies[who].alive or (boundary and who == 0)
                                    learner.record(choices[who][1], rewards[who], world.observation(who),
                                                   terminal, learn=self.config.arm != "C", flush=boundary)
                                    self.agent_transitions += 1
                            self.joint_steps += 1
                            if world.t % 16 == 0 or boundary:
                                within = (index + world.t/self.config.world.horizon)/self.config.population
                                self._progress(callback, label="Lifetime learning" if self.config.arm != "C" else "Frozen policy · evolution only", fraction=within)
                        family.mother.end_context()
                        family.child.end_context()
                        self.family_index = index+1
                    self.phase = "assessment"
                    self.family_index = len(self.families)
                if self.phase == "assessment":
                    results = []
                    candidates = []
                    for index, family in enumerate(self.families):
                        control()
                        p = fitness_probe(family.child, family.world.child, self.config, control,
                                          lambda: self._count_probe("fitness"), as_adult=True) if family.world.child.alive else 0.0
                        health = family.world.summary()["child_health_mean"]/100
                        score = int(family.world.child.alive)*(0.5+0.3*health+0.2*p)
                        results.append(score)
                        key = self.generation, index
                        row = {"generation": self.generation, "family": index, "arm": self.config.arm,
                               "run_seed": self.config.seed, "solo_score": p, "adult_readiness_score": p, "fitness": score,
                               "fitness_health_score": health, "fitness_rule": "survival*(0.5+0.3*health+0.2*solo)",
                               "mother_genome": genome_hash(family.mother.genome),
                               "child_genome": genome_hash(family.child.genome),
                               "mother_updates": family.mother.updates, "child_updates": family.child.updates,
                               "joint_steps_total": self.joint_steps, **family.world.summary()}
                        if key not in self.seen_metrics:
                            for daily in family.world.daily:
                                append_csv(self.output/"development.csv", {"generation": self.generation,
                                           "family": index, "arm": self.config.arm, "run_seed": self.config.seed, **daily})
                            append_csv(self.output/"metrics.csv", row)
                            self.seen_metrics.add(key)
                        if family.world.child.alive:
                            candidates.append(index)
                        self._progress(callback, family.world, "Train fitness probes", (index+1)/self.config.population)
                    self.scores, self.candidates = results, candidates
                    self.phase = "evaluation"
                if self.phase == "evaluation":
                    # Same sampled mother indices in all phases; evaluation initialization is RNG-isolated.
                    sample = sample_unique([(f.mother, f.world.mother) for f in self.families],
                                           self.config.eval_candidates, self.config.seed+self.generation*7919+900_000)
                    for phase_index, phase in enumerate(("A0", "A1", "A2")):
                        self.active_eval_phase = phase_index
                        records = []
                        for index, family in enumerate(self.families):
                            learner = family.mother if phase == "A2" else Learner.restore(self.config, self.starts[index])
                            body = family.world.mother if phase == "A2" else self.start_bodies[index]
                            records.append((learner, body))
                        for row in evaluate_phase(records, self.config, self.generation, phase, control,
                                lambda w, f: self._progress(callback, w, f"Zero-shot · {phase} · {w.care_test.get('care_policy', 'learned')}", f), sample_indices=sample,
                                on_step=lambda: self._count_probe("evaluation")):
                            key = (row["generation"], row["phase"], row["candidate"], row["scenario"],
                                   row["scenario_seed"], row["intervention"], row["care_policy"])
                            if key not in self.seen_eval:
                                append_csv(self.output/"evaluations.csv", row)
                                self.seen_eval.add(key)
                    self.completed_generations = self.generation+1
                    if not self.candidates and self.config.arm != "D":
                        self.phase = "finished"
                        self._manifest(status="extinct", extinction_generation=self.generation)
                        self.save(f"gen_{self.generation:03d}.pt")
                        self._progress(callback, label="Population extinct · results retained", fraction=1)
                        callback({"kind": "done", "status": "extinct", "output": str(self.output)})
                        return
                    self.phase = "selection"
                if self.phase == "selection":
                    control()
                    if self.config.arm == "D":
                        selected = list(range(self.config.population))
                        self.next_adults = [{"learner": f.mother.state(False), "body": copy.deepcopy(f.world.mother)} for f in self.families]
                    else:
                        values = [self.scores[i] for i in self.candidates]
                        highest = max(values)
                        weights = None if self.config.arm == "B" else [math.exp((v-highest)/self.config.selection_temperature) for v in values]
                        selected = self.rng.choices(self.candidates, weights=weights, k=self.config.population)
                        self.next_adults = [{"learner": self.families[i].child.state(False),
                                             "body": copy.deepcopy(self.families[i].world.child)} for i in selected]
                    self.unique_candidates = len(set(selected))
                    append_json(self.output/"events.jsonl", {"type": "selection", "generation": self.generation,
                                "selected_family_indices": selected, "unique_sources": self.unique_candidates})
                    previous_generation = self.generation
                    self.generation += 1
                    self.phase = "finished" if self.generation >= self.config.generations else "initialize"
                    self.save(f"gen_{previous_generation:03d}.pt")
                    self._progress(callback, label="Generation saved", fraction=1)
            self._manifest(status="completed")
            callback({"kind": "done", "status": "completed", "output": str(self.output)})
        except StopRequested:
            self.save("interrupted.pt")
            self._manifest(status="stopped")
            callback({"kind": "done", "status": "stopped", "output": str(self.output)})
        except Exception as exc:
            self._manifest(status="error", error=f"{type(exc).__name__}: {exc}")
            raise

    def _count_probe(self, kind):
        if kind == "fitness":
            self.fitness_steps += 1
        else:
            self.evaluation_steps += 1

    def save(self, filename="latest.pt"):
        data = {"schema": 1, "protocol": PROTOCOL, "config": self.config.to_dict(),
                "source_hashes": source_hashes(), "output": str(self.output),
                "generation": self.generation, "family_index": self.family_index, "phase": self.phase,
                "build_index": self.build_index,
                "completed_generations": self.completed_generations,
                "joint_steps": self.joint_steps, "agent_transitions": self.agent_transitions,
                "fitness_steps": self.fitness_steps, "evaluation_steps": self.evaluation_steps,
                "families": [{"world": copy.deepcopy(f.world), "mother": f.mother.state(), "child": f.child.state()} for f in self.families],
                "starts": self.starts, "start_bodies": self.start_bodies, "next_adults": self.next_adults,
                "scores": getattr(self, "scores", []), "candidates": getattr(self, "candidates", []),
                "unique_candidates": self.unique_candidates, "rng": self.rng.getstate(),
                "python_rng": random.getstate(), "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if self.config.device == "cuda" else None}
        path = self.output/"models"/filename
        temporary = path.with_suffix(".tmp")
        torch.save(data, temporary)
        os.replace(temporary, path)
        if filename != "latest.pt":
            # Replace latest atomically, without recomputing states or changing any RNG.
            import shutil
            latest_temp = self.output/"models"/"latest.tmp"
            shutil.copyfile(path, latest_temp)
            os.replace(latest_temp, self.output/"models"/"latest.pt")
        self.last_checkpoint = str(path)

    @classmethod
    def resume(cls, path):
        path = Path(path).resolve()
        payload = load_checkpoint(path)
        if payload.get("source_hashes") != source_hashes():
            raise ValueError("Source files changed since checkpoint; exact resume is refused. Keep a matching source copy.")
        obj = cls.__new__(cls)
        obj.config = ExperimentConfig.from_dict(payload["config"])
        obj.config.device = resolve_device()
        # The checkpoint's current location is authoritative after moving a run.
        obj.output = path.parent.parent
        if path.parent.name != "models" or not all((obj.output/name).is_file() for name in ("manifest.json", "config.json")):
            raise ValueError("Move the complete run folder (including models, config.json, manifest.json and result CSVs), not just the .pt file")
        obj.manifest = json.loads((obj.output/"manifest.json").read_text(encoding="utf-8"))
        disk_config = json.loads((obj.output/"config.json").read_text(encoding="utf-8"))
        if (obj.manifest.get("source_hashes") != payload["source_hashes"]
                or any(disk_config.get(key) != payload["config"].get(key)
                       for key in ("arm", "seed", "population", "generations", "world", "ppo"))):
            raise ValueError("Checkpoint does not match the surrounding run manifest/config")
        obj.manifest["resumed_from_output"] = payload.get("output")
        obj.manifest["runtime_device"] = obj.config.device
        obj.rng = random.Random()
        obj.rng.setstate(payload["rng"])
        obj.build_index = payload.get("build_index", 0)
        obj.active_eval_phase = 0
        for key in ("generation", "family_index", "phase", "joint_steps", "agent_transitions", "fitness_steps",
                    "evaluation_steps", "starts", "start_bodies", "next_adults", "scores", "candidates", "unique_candidates", "completed_generations"):
            setattr(obj, key, payload[key])
        obj.families = [Family(item["world"], Learner.restore(obj.config, item["mother"]),
                               Learner.restore(obj.config, item["child"])) for item in payload["families"]]
        obj.watched_family, obj.last_checkpoint = 0, str(path)
        obj.seen_eval, obj.seen_metrics = set(), set()
        from .evaluation import read_rows
        for row in read_rows(obj.output/"metrics.csv"):
            obj.seen_metrics.add((int(row["generation"]), int(row["family"])))
        for row in read_rows(obj.output/"evaluations.csv"):
            obj.seen_eval.add((int(row["generation"]), row["phase"], int(row["candidate"]), row["scenario"], int(row["scenario_seed"]), row["intervention"], row.get("care_policy", "learned")))
        random.setstate(payload["python_rng"])
        np.random.set_state(payload["numpy_rng"])
        torch.set_rng_state(payload["torch_rng"])
        if obj.config.device == "cuda" and payload.get("cuda_rng"):
            saved_cuda_rng = payload["cuda_rng"]
            for index, state in enumerate(saved_cuda_rng[:torch.cuda.device_count()]):
                torch.cuda.set_rng_state(state, device=index)
        torch.set_num_threads(1)
        return obj
