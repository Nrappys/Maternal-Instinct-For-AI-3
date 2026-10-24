"""Independent SB3 PPO learners driven by a single external joint-step collector.

This module is imported only after an explicit training/evaluation request.
PPO.learn() is deliberately not used: it would advance two separate world clocks.
"""
from __future__ import annotations

import copy
import hashlib
import random
from contextlib import contextmanager

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from torch import nn
from stable_baselines3 import PPO
from stable_baselines3.common.logger import Logger
from stable_baselines3.common.buffers import RolloutBuffer

from .config import ExperimentConfig, resolve_device
from .environment import Action, CUE_SLICE, OBS_SIZE, LOCAL_CHANNELS


@contextmanager
def isolated_initialization(seed):
    py_state, np_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


class SpaceOnlyEnv(gym.Env):
    """Space declaration for SB3. Never used to collect real transitions."""
    def __init__(self, latent):
        self.observation_space = spaces.Box(-2, 2, shape=(OBS_SIZE+latent,), dtype=np.float32)
        self.action_space = spaces.Discrete(len(Action))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(self.observation_space.shape, np.float32), {}

    def step(self, action):
        raise RuntimeError("Use FamilyWorld.step once for both agents; never call PPO.learn()")


class AgentRolloutBuffer(RolloutBuffer):
    """SB3 minibatches with an explicitly owned RNG, never global NumPy state."""

    def get(self, batch_size=None):
        if not self.full:
            raise RuntimeError("Cannot sample an incomplete rollout")
        indices = self.shuffle_rng.permutation(self.buffer_size * self.n_envs)
        if not self.generator_ready:
            for name in ("observations", "actions", "values", "log_probs", "advantages", "returns"):
                setattr(self, name, self.swap_and_flatten(getattr(self, name)))
            self.generator_ready = True
        batch_size = batch_size or self.buffer_size * self.n_envs
        for start in range(0, len(indices), batch_size):
            yield self._get_samples(indices[start:start + batch_size])


def actor_state(policy):
    return {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()
            if k.startswith("mlp_extractor.policy_net.") or k.startswith("action_net.")}


def genome_hash(genome):
    digest = hashlib.sha256()
    for section in ("actor", "phi"):
        for key, value in sorted(genome[section].items()):
            digest.update((section+key).encode())
            digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()[:20]


def mutate(genome, seed, scale):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    result = copy.deepcopy(genome)
    for section in ("actor", "phi"):
        for key, value in result[section].items():
            sigma = scale * max(float(value.float().square().mean().sqrt()), 0.1)
            result[section][key] = value + torch.randn(value.shape, generator=generator, dtype=value.dtype) * sigma
    return result


class Learner:
    def __init__(self, config: ExperimentConfig, seed: int, genome=None):
        self.config, self.seed = config, seed
        device = resolve_device(config.device)
        p = config.ppo
        with isolated_initialization(seed):
            self.model = PPO("MlpPolicy", SpaceOnlyEnv(p.latent), learning_rate=p.learning_rate,
                             n_steps=p.rollout, batch_size=p.batch_size, n_epochs=p.epochs,
                             gamma=p.gamma, gae_lambda=p.gae_lambda, clip_range=p.clip_range,
                             ent_coef=p.entropy, vf_coef=0.5, max_grad_norm=0.5,
                             policy_kwargs={"net_arch": {"pi": [p.hidden, p.hidden], "vf": [p.hidden, p.hidden]},
                                            "activation_fn": nn.Tanh}, device=device, seed=seed, verbose=0)
            self.phi = nn.Sequential(nn.Linear(OBS_SIZE+p.latent, 16), nn.Tanh(), nn.Linear(16, p.latent), nn.Tanh())
            # Frozen latent mapping stays on CPU; PPO policy/update tensors use device.
        self.model.set_logger(Logger(folder=None, output_formats=[]))
        if genome is not None:
            self.model.policy.load_state_dict(genome["actor"], strict=False)
            self.phi.load_state_dict(genome["phi"])
        self.phi.requires_grad_(False)
        self.phi.eval()
        self.genome = {"actor": actor_state(self.model.policy), "phi": copy.deepcopy(self.phi.state_dict())}
        self.z = np.zeros(p.latent, dtype=np.float32)
        self.generator = torch.Generator(device="cpu").manual_seed(seed ^ 0x5A17)
        self.shuffle_rng = np.random.default_rng(seed ^ 0x6B28)
        self.episode_start = True
        self.transitions = 0
        self.updates = 0
        self.discarded = 0
        self.last_loss: float | None = None

    def _input(self, observation, intervention="normal", commit=False):
        raw = np.asarray(observation, dtype=np.float32)
        with torch.no_grad():
            updated = self.phi(torch.from_numpy(np.concatenate((raw, self.z)))).numpy()
        if intervention == "z_zero":
            updated = np.zeros_like(updated)
        elif intervention == "z_flip":
            updated = -updated
        direct = raw.copy()
        if intervention == "direct_cue_block":
            direct[CUE_SLICE] = 0
            # The other-agent occupancy channel is a second direct social pathway.
            direct[CUE_SLICE.stop + 4::LOCAL_CHANNELS] = 0
        elif intervention == "self_feature_control":
            direct[4] = 0  # zero own fear as a limited feature-removal control
        if commit:
            self.z = updated.copy()
        return np.concatenate((direct, updated)).astype(np.float32)

    def choose(self, observation, intervention="normal", greedy=False):
        augmented = self._input(observation, intervention, commit=True)
        tensor = torch.from_numpy(augmented[None, :]).to(self.model.device)
        self.model.policy.set_training_mode(False)
        with torch.no_grad():
            distribution = self.model.policy.get_distribution(tensor)
            # Sample on CPU with the existing per-agent RNG on either backend.
            probabilities = distribution.distribution.probs[0].cpu()
            action = torch.argmax(probabilities) if greedy else torch.multinomial(probabilities, 1, generator=self.generator)[0]
            log_prob = distribution.log_prob(action.reshape(1).to(self.model.device))
            value = self.model.policy.predict_values(tensor)
        return int(action), (augmented, int(action), value, log_prob, self.episode_start)

    def record(self, transition, reward, next_observation, terminal, learn=True, flush=False):
        self.transitions += 1
        if not learn:
            self.episode_start = bool(terminal)
            return
        obs, action, value, log_prob, start = transition
        buffer = self.model.rollout_buffer
        buffer.add(obs[None, :], np.array([action]), np.array([reward]), np.array([start]), value, log_prob)
        self.episode_start = bool(terminal)
        self.model.num_timesteps += 1
        if buffer.full or terminal or flush:
            self._flush(next_observation, terminal)

    def _flush(self, next_observation, terminal):
        """Train exactly the collected samples, including a one-sample death tail.

        The temporary SB3 buffer has its real length: no padding, duplication,
        or silently dropped rewards. Child horizon boundaries bootstrap because
        the child may continue life as a mother; death uses a zero bootstrap.
        """
        original = self.model.rollout_buffer
        count = original.buffer_size if original.full else original.pos
        if not count:
            return
        p = self.config.ppo
        buffer_type = AgentRolloutBuffer if self.config.ppo_shuffle_rng == "per_agent" else RolloutBuffer
        tail = buffer_type(count, self.model.observation_space, self.model.action_space,
                             device=self.model.device, gamma=p.gamma, gae_lambda=p.gae_lambda, n_envs=1)
        if isinstance(tail, AgentRolloutBuffer):
            tail.shuffle_rng = self.shuffle_rng
        for name in ("observations", "actions", "rewards", "episode_starts", "values", "log_probs"):
            setattr(tail, name, getattr(original, name)[:count].copy())
        tail.pos, tail.full = count, True
        with torch.no_grad():
            last = torch.zeros(1, device=self.model.device) if terminal else self.model.policy.predict_values(
                torch.from_numpy(self._input(next_observation)[None, :]).to(self.model.device))
        tail.compute_returns_and_advantage(last_values=last, dones=np.array([terminal]))
        old_batch_size = self.model.batch_size
        self.model.rollout_buffer, self.model.batch_size = tail, min(old_batch_size, count)
        try:
            self.model.train()
            self.updates += 1
            self.last_loss = float(self.model.logger.name_to_value.get("train/loss", float("nan")))
        finally:
            self.model.rollout_buffer, self.model.batch_size = original, old_batch_size
        original.reset()

    def end_context(self):
        buffer = self.model.rollout_buffer
        if buffer.pos:
            raise RuntimeError("Unflushed PPO samples: record the horizon with flush=True before ending context")
        buffer.reset()
        self.episode_start = True

    def state(self, include_buffer=True):
        result = {"seed": self.seed, "genome": copy.deepcopy(self.genome),
                  "policy": copy.deepcopy(self.model.policy.state_dict()),
                  "optimizer": copy.deepcopy(self.model.policy.optimizer.state_dict()),
                  "z": self.z.copy(), "generator": self.generator.get_state().clone(),
                  "shuffle_rng": copy.deepcopy(self.shuffle_rng.bit_generator.state),
                  "episode_start": self.episode_start, "transitions": self.transitions,
                  "updates": self.updates, "discarded": self.discarded,
                  "n_updates": self.model._n_updates, "last_loss": self.last_loss}
        if include_buffer:
            result["buffer"] = copy.deepcopy(self.model.rollout_buffer)
        return result

    @classmethod
    def restore(cls, config, state):
        obj = cls(config, state["seed"], state["genome"])
        obj.model.policy.load_state_dict(state["policy"])
        obj.model.policy.optimizer.load_state_dict(state["optimizer"])
        obj.z = state["z"].copy()
        obj.generator.set_state(state["generator"])
        if "shuffle_rng" in state:
            obj.shuffle_rng.bit_generator.state = copy.deepcopy(state["shuffle_rng"])
        for name in ("episode_start", "transitions", "updates", "discarded", "last_loss"):
            setattr(obj, name, state[name])
        obj.model._n_updates = state.get("n_updates", obj.updates * config.ppo.epochs)
        obj.model.num_timesteps = obj.transitions
        if "buffer" in state:
            obj.model.rollout_buffer = copy.deepcopy(state["buffer"])
            obj.model.rollout_buffer.device = obj.model.device
        return obj

    def evaluation_copy(self, naive=False, seed=0):
        # Initialization preserves global RNG; evaluation never consumes training RNG.
        result = Learner(self.config, seed, self.genome)
        if not naive:
            result.model.policy.load_state_dict(copy.deepcopy(self.model.policy.state_dict()))
        result.z[:] = 0
        return result
