"""Configuration only: importing this module never starts a run."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
import json
import os

ROOT = Path(__file__).resolve().parent.parent
PROJECT_ID = "maternal-instinct-3"
OUTPUT_ROOT = ROOT / "outputs" / "2_model"
PROTOCOL = "maternal-house-v2.3-growing-child"


def available_cpu_count():
    """Respect Linux CPU affinity when present; no hardware probing or execution."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def recommended_process_workers(population=16):
    # Conservative starting point for this 155H / 16 GB laptop. Each worker
    # imports its own Torch/SB3 runtime; more workers also cost more memory.
    return max(1, min(4, population, available_cpu_count() - 2))


def resolve_device(requested="auto"):
    """Called when a job starts, never while importing config or displaying panels."""
    if requested not in ("auto", "cuda", "cpu"):
        raise ValueError("device must be auto, cuda or cpu")
    if requested == "cpu":
        return "cpu"
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class WorldConfig:
    width: int = 7
    height: int = 5
    steps_per_day: int = 200
    days: int = 20
    early_child_days: int = 7
    late_child_day: int = 14
    juvenile_gather_interval: int = 40
    late_gather_interval: int = 20
    mature_gather_interval: int = 4
    vision: int = 2
    hearing: int = 4
    food_capacity: int = 2
    meal_energy: float = 30.0
    food_sites: int = 1
    adult_drain: float = 0.03
    infant_drain: float = 0.12
    juvenile_drain: float = 0.07
    late_child_drain: float = 0.05
    move_energy: float = 0.025
    carry_energy: float = 0.05
    starvation_threshold: float = 20.0
    starvation_damage: float = 0.08
    recovery: float = 0.008
    fatigue_cost: float = 0.05
    rest_recovery: float = 0.5
    fatigue_gain_adult: float = 0.008
    fatigue_gain_child: float = 0.015
    exhaustion_threshold: float = 90.0
    exhaustion_damage: float = 0.01
    stress_threshold: float = 90.0
    stress_damage: float = 0.01
    neglect_grace: int = 40
    unmet_need_stress: float = 0.08
    work_reward: float = 0.015
    work_fatigue: float = 0.12
    work_energy: float = 0.01
    work_fatigue_limit: float = 85.0
    play_relief: float = 0.7
    fear_gain: float = 2.0
    fear_recovery: float = 0.35
    soothe_effect: float = 12.0
    skill_gain: float = 0.003
    hazard_damage: float = 0.5
    hazard_period: int = 50
    signal_duration: int = 20

    @property
    def horizon(self) -> int:
        return self.days * self.steps_per_day


@dataclass
class PPOConfig:
    learning_rate: float = 1e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    rollout: int = 128
    batch_size: int = 64
    epochs: int = 4
    entropy: float = 0.01
    latent: int = 4
    hidden: int = 32


@dataclass
class ExperimentConfig:
    mode: str = "pilot"
    arm: str = "A"
    seed: int = 0
    population: int = 16
    generations: int = 20
    mutation_scale: float = 0.02
    selection_temperature: float = 0.15
    eval_candidates: int = 4
    eval_seeds: int = 3
    eval_steps: int = 4000
    fitness_seeds: int = 5
    fitness_steps: int = 200
    device: str = "auto"
    family_workers: int = 1
    ppo_shuffle_rng: str = "legacy"
    execution_backend: str = "threads"
    process_chunk_steps: int = 128
    parallel_zero_shot: bool = True
    world: WorldConfig = field(default_factory=WorldConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)

    @classmethod
    def profile(cls, mode: str = "pilot", arm: str = "A", seed: int = 0):
        config = cls(mode=mode, arm=arm, seed=seed)
        if mode == "main":
            config.population, config.generations = 64, 100
            config.eval_candidates, config.eval_seeds = 8, 5
        elif mode == "smoke":
            config.population, config.generations = 2, 2
            config.eval_candidates, config.eval_seeds = 1, 1
            config.fitness_seeds, config.fitness_steps = 1, 40
            config.world.days = 2
            config.ppo.rollout, config.ppo.batch_size = 64, 32
        config.execution_backend = "processes"
        config.family_workers = recommended_process_workers(config.population)
        config.validate()
        return config

    def validate(self):
        if type(self.parallel_zero_shot) is not bool:
            raise ValueError("parallel_zero_shot must be True or False")
        if self.execution_backend not in ("serial", "threads", "processes"):
            raise ValueError("execution_backend must be serial, threads or processes")
        if type(self.process_chunk_steps) is not int or not 1 <= self.process_chunk_steps <= 1024:
            raise ValueError("process_chunk_steps must be an integer from 1 to 1024")
        if type(self.family_workers) is not int or not 1 <= self.family_workers <= 64:
            raise ValueError("family_workers must be an integer from 1 to 64")
        if self.ppo_shuffle_rng not in ("legacy", "per_agent"):
            raise ValueError("ppo_shuffle_rng must be legacy or per_agent")
        if self.family_workers > 1 or self.execution_backend == "processes":
            # Global NumPy shuffling would depend on thread scheduling.
            self.ppo_shuffle_rng = "per_agent"
        if self.device not in ("auto", "cuda", "cpu"):
            raise ValueError("device must be auto, cuda or cpu")
        if self.arm not in "ABCD" or len(self.arm) != 1:
            raise ValueError("arm must be A, B, C or D")
        if self.population < 1 or self.generations < 1:
            raise ValueError("population and generations must be positive")
        if (self.world.width, self.world.height) != (7, 5):
            raise ValueError("v1 uses a 7 × 5 grid")
        if self.world.days < 1 or self.world.steps_per_day < 1:
            raise ValueError("days and steps_per_day must be positive")
        if not 0 < self.world.early_child_days < self.world.late_child_day:
            raise ValueError("Require 0 < early_child_days < late_child_day")
        if not 1 <= self.world.mature_gather_interval <= self.world.late_gather_interval:
            raise ValueError("Require 1 <= mature_gather_interval <= late_gather_interval")
        if self.world.juvenile_gather_interval < self.world.late_gather_interval:
            raise ValueError("juvenile_gather_interval must be >= late_gather_interval")
        # Caregiving evaluation always covers the same lifetime as training.
        # The evaluation budget follows this configuration's episode clock.
        self.eval_steps = self.world.horizon
        if self.world.vision != 2:
            raise ValueError("The fixed observation schema requires vision=2")
        if not 1 <= self.world.food_sites <= 10 or self.world.food_capacity < 1:
            raise ValueError("invalid food-site count or inventory capacity")
        if self.world.hazard_period < 1 or self.world.signal_duration < 1:
            raise ValueError("hazard and signal periods must be positive")
        for item in fields(self.world):
            value = getattr(self.world, item.name)
            if isinstance(value, (int, float)) and value < 0:
                raise ValueError(f"world.{item.name} cannot be negative")
        if not 0 <= self.seed < 2**31:
            raise ValueError("seed must be in [0, 2**31)")
        if self.ppo.rollout < 2 or self.ppo.batch_size < 2:
            raise ValueError("PPO batches must contain at least two transitions")
        if self.ppo.rollout % self.ppo.batch_size:
            raise ValueError("rollout must be divisible by batch_size")
        if self.selection_temperature <= 0 or self.mutation_scale < 0:
            raise ValueError("invalid evolutionary parameters")
        for name in ("eval_candidates", "eval_seeds", "eval_steps", "fitness_seeds", "fitness_steps"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict):
        value = dict(value)
        value["world"] = WorldConfig(**value.get("world", {}))
        value["ppo"] = PPOConfig(**value.get("ppo", {}))
        allowed = {f.name for f in fields(cls)}
        result = cls(**{k: v for k, v in value.items() if k in allowed})
        result.validate()
        return result

    @classmethod
    def load(cls, path: str | Path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
