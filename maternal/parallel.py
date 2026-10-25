"""Explicit, persistent CPU processes for independent family learning.

Importing this module starts nothing. Targets live in this importable module so
the spawn backend works from Jupyter on Windows and Linux. Pipe payloads are
trusted local pickle bytes, not shared Torch tensor handles or external files.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import pickle
import time
import traceback
from types import SimpleNamespace

from .config import available_cpu_count


def advance_family(family, config, limit):
    """Same per-family clock and PPO flush boundaries as the serial collector."""
    from .environment import Action

    world = family.world
    steps = transitions = 0
    for _ in range(min(limit, config.world.horizon - world.t)):
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
                               terminal, learn=config.arm != "C", flush=boundary)
                transitions += 1
        steps += 1
    if steps and world.t == config.world.horizon:
        family.mother.end_context()
        family.child.end_context()
    return steps, transitions


def family_state(index, family):
    return {"index": index, "world": family.world,
            "mother": family.mother.state(), "child": family.child.state()}


def _send(connection, value):
    connection.send_bytes(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))


def _worker_main(connection, config_dict):
    # Set limits before importing NumPy/Torch in this spawned target.
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[name] = "1"
    try:
        import torch
        from .agent import Learner
        from .config import ExperimentConfig

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        config = ExperimentConfig.from_dict(config_dict)
        config.device = "cpu"
        families = {}
        while True:
            command = pickle.loads(connection.recv_bytes())
            if command["kind"] == "close":
                return
            if command["kind"] == "load":
                families = {row["index"]: SimpleNamespace(
                    world=row["world"], mother=Learner.restore(config, row["mother"]),
                    child=Learner.restore(config, row["child"])) for row in command["families"]}
                _send(connection, {"ok": True})
            elif command["kind"] == "advance":
                steps = transitions = 0
                frames = []
                for index, family in families.items():
                    if family.world.t < config.world.horizon:
                        count, learned = advance_family(family, config, config.process_chunk_steps)
                        steps += count
                        transitions += learned
                    frames.append({"index": index, "world": family.world.snapshot(),
                                   "z_mother": family.mother.z.tolist(),
                                   "done": family.world.t == config.world.horizon})
                _send(connection, {"ok": True, "steps": steps,
                                   "transitions": transitions, "frames": frames})
            elif command["kind"] == "snapshot":
                _send(connection, {"ok": True, "families": [family_state(i, f) for i, f in families.items()]})
            else:
                raise ValueError("Unknown family-process command")
    except (EOFError, BrokenPipeError):
        pass
    except BaseException:
        try:
            _send(connection, {"ok": False, "error": traceback.format_exc()})
        except (EOFError, OSError):
            pass
    finally:
        connection.close()


class ProcessFamilies:
    """Coordinator-owned channels; workers never write run files or widgets."""

    def __init__(self, config, families):
        if config.device != "cpu" or config.ppo_shuffle_rng != "per_agent":
            raise ValueError("Process families require CPU and per-agent PPO shuffle RNG")
        self.config, self.families = config, families
        self.count = min(config.family_workers, len(families), available_cpu_count())
        self.workers = []
        self.consistent = False

    def __enter__(self):
        context = mp.get_context("spawn")
        try:
            for _ in range(self.count):
                parent, child = context.Pipe()
                process = context.Process(target=_worker_main, args=(child, self.config.to_dict()),
                                          name="maternal-family", daemon=True)
                try:
                    process.start()
                except BaseException:
                    parent.close()
                    raise
                finally:
                    child.close()
                self.workers.append((process, parent))
            # Model/optimizer/buffer state is transferred only here and at exit.
            for slot, (_, connection) in enumerate(self.workers):
                _send(connection, {"kind": "load", "families": [family_state(i, self.families[i])
                      for i in range(slot, len(self.families), self.count)]})
            for worker in self.workers:
                self._receive(worker)
            self.consistent = True
            return self
        except BaseException:
            self.close()
            raise

    def _receive(self, worker):
        process, connection = worker
        deadline = time.monotonic() + 300
        while not connection.poll(0.2):
            if not process.is_alive():
                self.consistent = False
                raise RuntimeError(f"Family worker exited (code {process.exitcode}); no partial state saved")
            if time.monotonic() > deadline:
                self.consistent = False
                raise TimeoutError("Family worker did not respond within 300 seconds")
        try:
            result = pickle.loads(connection.recv_bytes())
        except (EOFError, OSError):
            self.consistent = False
            raise RuntimeError("Family worker connection closed; no partial state saved") from None
        if not result.get("ok"):
            self.consistent = False
            raise RuntimeError("Family worker failed:\n" + result.get("error", "unknown error"))
        return result

    def advance(self):
        self.consistent = False
        for _, connection in self.workers:
            _send(connection, {"kind": "advance"})
        results = [self._receive(worker) for worker in self.workers]
        self.consistent = True
        return results

    def snapshot(self):
        if not self.consistent:
            raise RuntimeError("Cannot snapshot workers while a round is incomplete")
        for _, connection in self.workers:
            _send(connection, {"kind": "snapshot"})
        return [row for worker in self.workers for row in self._receive(worker)["families"]]

    def close(self):
        for _, connection in self.workers:
            try:
                _send(connection, {"kind": "close"})
            except OSError:
                pass
            connection.close()
        for process, _ in self.workers:
            process.join(timeout=2)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
        self.workers.clear()

    def __exit__(self, *_):
        self.close()


def _evaluation_initializer(config_dict, subjects_bytes, cancel):
    """One frozen subject cache per CPU process; no training state is shared."""
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    import torch
    from .agent import Learner
    from .config import ExperimentConfig

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    global _eval_config, _eval_subjects, _eval_cancel, _eval_actors
    _eval_config = ExperimentConfig.from_dict(config_dict)
    _eval_config.device = "cpu"
    _eval_cancel = cancel
    _eval_subjects = {}
    _eval_actors = {}
    for item in pickle.loads(subjects_bytes):
        _eval_subjects[item["index"]] = (Learner.restore(_eval_config, item["state"]),
                                        item["body"], item["metadata"])


def _evaluation_job(job):
    from .evaluation import fitness_probe, run_scenario

    learner, body, metadata = _eval_subjects[job[1]]
    steps = 0

    def control():
        if _eval_cancel.is_set():
            raise RuntimeError("Frozen evaluation cancelled by coordinator")

    def on_step():
        nonlocal steps
        steps += 1

    control()
    naive = metadata["phase"] == "A0"
    if job[0] == "fitness":
        actor = learner.evaluation_copy(naive=naive, seed=600_000)
        score = fitness_probe(actor, body, _eval_config, control, on_step, as_adult=True)
        return pickle.dumps({"index": job[1], "score": score, "steps": steps})
    _, index, scenario, seed, intervention, policy = job
    # Private actor weights are immutable throughout this worker's phase.
    # Reuse the model allocation, but reset all case memory and action RNG.
    if index not in _eval_actors:
        _eval_actors[index] = learner.evaluation_copy(naive=naive, seed=500_000+seed)
    actor = _eval_actors[index]
    actor.reset_evaluation_context(500_000+seed)
    final_world = None

    def progress(world):
        nonlocal final_world
        final_world = world

    result = run_scenario(actor, body, _eval_config, scenario, seed, intervention,
                          control, progress, on_step, metadata["test_split"], policy)
    return pickle.dumps({"index": index, "result": result, "world": final_world, "steps": steps})


def ordered_evaluation_results(executor, jobs, workers, control):
    """Bound the queue and yield in serial job order, never completion order."""
    from collections import deque
    from concurrent.futures import TimeoutError as FutureTimeout

    iterator = iter(jobs)
    pending = deque()
    for _ in range(2 * workers):
        job = next(iterator, None)
        if job is None:
            break
        pending.append(executor.submit(_evaluation_job, job))
    while pending:
        future = pending.popleft()
        while True:
            control()
            try:
                value = future.result(timeout=0.1)
                break
            except FutureTimeout:
                if future.done():
                    # Propagate a timeout raised inside the worker itself.
                    value = future.result()
                    break
        yield pickle.loads(value)
        job = next(iterator, None)
        if job is not None:
            pending.append(executor.submit(_evaluation_job, job))
