"""Explicit notebook controls. Creating/displaying a panel never starts a run.

No torch, numpy, PPO or Matplotlib imports occur until the corresponding button
is pressed. All panels use the kernel event loop to update their widgets.
"""
from __future__ import annotations

import asyncio
import copy
import html
import importlib.util
import json
from pathlib import Path
import queue
import threading
import time
import traceback

import ipywidgets as W
from IPython.display import HTML, Image, display

from .config import ExperimentConfig, ROOT, PROTOCOL, OUTPUT_ROOT, available_cpu_count
from .environment import Action, FamilyWorld, demo_action

from .rendering import COLORS, world_png

_RUN_LOCK = threading.Lock()


def device_description(config):
    """Only called by a started worker, after device selection."""
    import torch
    if config.execution_backend == "processes":
        return "Device: CPU · multi-core process mode; 1 PyTorch thread per process"
    if config.device == "cuda":
        return f"Device: CUDA · {torch.cuda.get_device_name(0)}"
    reason = "CPU selected / CPU-only PyTorch build" if torch.version.cuda is None else "CPU selected or CUDA unavailable to PyTorch"
    return f"Device: CPU · {reason}"


def execution_description(config):
    workers = 1 if config.execution_backend == "serial" else min(config.family_workers, config.population)
    if config.execution_backend == "processes":
        workers = min(workers, available_cpu_count())
    evaluation = "CPU processes" if config.execution_backend == "processes" and config.parallel_zero_shot else "sequential"
    return (f"Family backend: {config.execution_backend} · workers: {workers} · PPO shuffle: {config.ppo_shuffle_rng} · "
            f"train fitness: sequential · zero-shot: {evaluation}")


class NotebookStop(Exception):
    pass


def check_ml_dependencies():
    names = ("torch", "numpy", "stable_baselines3", "gymnasium")
    missing = [name for name in names if importlib.util.find_spec(name) is None]
    if missing:
        raise ImportError("Missing: " + ", ".join(missing) + ". Install requirements.txt manually in the selected kernel. Nothing was installed automatically.")


def _escape(value):
    return html.escape(str(value), quote=True)


class _Panel:
    def __init__(self, title, config, preview=False):
        self.title, self.config = title, copy.deepcopy(config)
        self.is_preview = preview
        self.messages = queue.Queue(maxsize=6)
        self.gate, self.cancel = threading.Event(), threading.Event()
        self.thread = None
        self.task = None
        self.running = False
        self.closed = False
        self.delay = 0.03
        self.show_simulation = True
        self.started_at = None
        self.log = W.Output(layout=W.Layout(max_height="160px", overflow="auto"))
        self.view = W.Image(format="png", layout=W.Layout(width="100%", max_width="1120px", height="auto"))
        self.status = W.HTML("<span style='color:#A48A73'>Ready · paused. Nothing has started.</span>")
        self.episode = W.FloatProgress(description="Episode", min=0, max=1, value=0, style={"bar_color": COLORS["mother"]}, layout=W.Layout(width="100%", height="22px"))
        self.overall = W.FloatProgress(description="Overall", min=0, max=1, value=0, style={"bar_color": COLORS["shelter"]}, layout=W.Layout(width="100%", height="22px"))
        self.start = W.Button(description="Start preview" if preview else "Start training", layout=W.Layout(width="150px"))
        self.start.style.button_color = COLORS["mother"]
        self.pause = W.Button(description="Pause", disabled=True, layout=W.Layout(width="100px"))
        self.stop = W.Button(description="Stop" if preview else "Stop & save", disabled=True, layout=W.Layout(width="130px"))
        self.pace = W.Dropdown(options=("Calm", "Flow", "Fast", "Off"), value="Flow", description="View", layout=W.Layout(width="180px"))
        self.pace.observe(self._set_pace, names="value")
        self.start.on_click(self._start)
        self.pause.on_click(self._pause)
        self.stop.on_click(self._stop)
        self.extra_controls = W.VBox([])
        initial = FamilyWorld(self.config.world, self.config.seed).snapshot()
        self.view.value = world_png(initial, title, "Initial preview layout · no model loaded · paused")
        self.widget = W.VBox([self.extra_controls, self.view, self.status, self.episode,
                              W.HBox([self.start, self.pause, self.stop, self.pace], layout=W.Layout(flex_flow="row wrap", gap="6px")),
                              self.overall, self.log], layout=W.Layout(width="100%", max_width="1120px", gap="10px"))

    @property
    def active(self):
        return self.thread is not None and self.thread.is_alive()

    def _set_pace(self, change):
        mode = change["new"]
        self.show_simulation = mode != "Off"
        self.delay = {"Calm": 0.1, "Flow": 0.03, "Fast": 0.0, "Off": 0.0}[mode]
        self.view.layout.display = "" if self.show_simulation else "none"

    def _start(self, _):
        if self.active:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self.status.value = "This panel requires an active Jupyter kernel event loop. Open the notebook in JupyterLab or VS Code."
            return
        if not self.is_preview and not _RUN_LOCK.acquire(blocking=False):
            self.status.value = "Another training/evaluation job is active. Stop it before starting this panel."
            return
        self.cancel.clear()
        self.gate.set()
        self.running = True
        self.closed = False
        self.started_at = time.monotonic()
        self.start.disabled = True
        self.pause.disabled = self.stop.disabled = False
        self.pause.description = "Pause"
        self.status.value = "Starting after your button click…"
        self._lock_controls(True)
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
        self.task = loop.create_task(self._poll())

    def _lock_controls(self, disabled):
        for widget in self.extra_controls.children:
            if hasattr(widget, "disabled"):
                widget.disabled = disabled

    def _pause(self, _):
        if self.gate.is_set():
            self.gate.clear()
            self.pause.description = "Resume"
            self.status.value = "Paused after the current atomic operation."
        else:
            self.gate.set()
            self.pause.description = "Pause"
            self.status.value = "Running…"

    def _stop(self, _):
        self.cancel.set()
        self.gate.set()
        self.stop.disabled = self.pause.disabled = True
        self.status.value = "Stopping after the current operation…"

    def close(self):
        self.closed = True
        if self.active:
            self._stop(None)

    def _check(self):
        while not self.gate.wait(0.1):
            if self.cancel.is_set():
                raise NotebookStop()
        if self.cancel.is_set():
            raise NotebookStop()

    def _emit(self, event):
        if self.closed and event.get("kind") == "frame":
            return
        while True:
            try:
                self.messages.put_nowait(event)
                break
            except queue.Full:
                try:
                    self.messages.get_nowait()
                except queue.Empty:
                    pass
        if event.get("kind") == "frame" and self.delay:
            self.cancel.wait(self.delay)

    def _worker(self):
        try:
            self._job()
        except Exception as exc:
            if isinstance(exc, NotebookStop) or type(exc).__name__ == "StopRequested":
                self._emit({"kind": "done", "status": "Stopped"})
            else:
                self._emit({"kind": "error", "message": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
        finally:
            if not self.is_preview:
                _RUN_LOCK.release()

    async def _poll(self):
        while self.active or not self.messages.empty():
            latest = None
            while True:
                try:
                    event = self.messages.get_nowait()
                except queue.Empty:
                    break
                if event["kind"] == "frame":
                    latest = event
                elif event["kind"] == "log":
                    self.log.append_stdout(event["text"]+"\n")
                elif event["kind"] == "error":
                    latest = None
                    self.status.value = "Stopped with an error. Nothing was installed or retried automatically."
                    self.log.append_stderr(event["message"]+"\n"+event["traceback"]+"\n")
                elif event["kind"] == "done":
                    if latest:
                        self._frame(latest)
                    latest = None
                    self.status.value = _escape(event.get("status", "Finished"))
                    if event.get("output"):
                        self.log.append_stdout("Saved: "+event["output"]+"\n")
            if latest:
                self._frame(latest)
            await asyncio.sleep(0.05)
        self.running = False
        self.start.disabled = False
        self.pause.disabled = self.stop.disabled = True
        self._lock_controls(False)

    def _frame(self, frame):
        world = frame.get("world")
        if world:
            note = frame.get("label", "")
            if "generation" in frame:
                generation = min(frame["generation"]+1, frame.get("generations", self.config.generations))
                note += f" · round {generation}/{frame.get('generations', self.config.generations)}"
            if "family" in frame:
                population = frame.get("population", self.config.population)
                note += f" · family {min(frame['family']+1, population)}/{population}"
            if self.show_simulation:
                self.view.value = world_png(world, self.title, note, frame.get("z_mother"))
            fraction = frame.get("episode_fraction", world["step"]/world["horizon"])
            self.episode.value = max(0, min(1, fraction))
            status = f"Day {world['day']}/{world['days']} · step {world['step']:,}/{world['horizon']:,} · {note}"
            if self.started_at is not None:
                elapsed = int(time.monotonic() - self.started_at)
                status += f" · elapsed {elapsed//60:02d}:{elapsed%60:02d}"
            if self.gate.is_set():
                self.status.value = _escape(status)
        self.overall.value = max(0, min(1, frame.get("overall", 0)))

    def _job(self):
        raise NotImplementedError


class PreviewPanel(_Panel):
    def __init__(self, config):
        super().__init__("A world of care", config, preview=True)
        self.start.description = "Start preview"

    def _job(self):
        world = FamilyWorld(self.config.world, self.config.seed)
        while world.t < world.config.horizon:
            self._check()
            world.step((demo_action(world, 0), demo_action(world, 1)))
            if world.t % 4 == 0 or world.t == world.config.horizon:
                self._emit({"kind": "frame", "world": world.snapshot(), "overall": world.t/world.config.horizon,
                            "label": "SCRIPTED PREVIEW · NO LEARNING"})
        self._emit({"kind": "done", "status": "Preview finished · no model was trained or saved"})


class BalancePanel(_Panel):
    """Explicit scripted reference check. Does not construct or train a model."""
    def __init__(self, config):
        super().__init__("Balance check · scripted references · no training", config)
        self.start.description = "Check balance"
        self.stop.description = "Stop check"

    def _job(self):
        import csv
        import hashlib
        from .environment import Body
        from .evaluation import SCENARIOS, run_scenario, EVALUATION_PROTOCOL
        config = copy.deepcopy(self.config)
        config.validate()
        policies = ("no_help", "appropriate_help", "always_help")
        scenarios = ("ordinary", "mother_loss", *SCENARIOS)
        jobs = [(scenario, seed, policy) for scenario in scenarios
                for seed in range(config.eval_seeds) for policy in policies]
        destination = OUTPUT_ROOT/f"balance_check_{time.time_ns()}"
        destination.mkdir(parents=True, exist_ok=False)
        metadata = {"status": "running", "world_protocol": PROTOCOL,
                    "evaluation_protocol": EVALUATION_PROTOCOL, "config": config.to_dict(),
                    "models_loaded": False, "learning_enabled": False,
                    "source_hashes": {p.name: hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
                                      for p in (ROOT/"maternal").glob("*.py")}}
        metadata_path = destination/"metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        results = []
        self._emit({"kind": "log", "text": f"Reference check: {len(jobs)} episodes; no model or training. Saved to {destination}"})
        for index, (scenario, seed, policy) in enumerate(jobs):
            self._check()
            def progress(world):
                self._emit({"kind": "frame", "world": world.snapshot(),
                            "overall": (index+world.t/config.world.horizon)/len(jobs),
                            "episode_fraction": world.t/config.world.horizon,
                            "label": f"CHECK · {policy} · {scenario} · seed {seed} · {index+1}/{len(jobs)}"})
            result = run_scenario(None, Body("reference-mother", role="mother", forage=.5, avoidance=.3),
                                  config, scenario, seed, control=self._check,
                                  progress=progress, care_policy=policy)
            results.append(result)
            with (destination/"episodes.csv").open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(result))
                if index == 0:
                    writer.writeheader()
                writer.writerow(result)
        summary = []
        for scenario in scenarios:
            for policy in policies:
                selected = [row for row in results if row["scenario"] == scenario and row["care_policy"] == policy]
                row = {"scenario": scenario, "care_policy": policy, "test_seeds": len(selected)}
                for metric in ("child_alive", "mother_alive", "both_alive", "child_health_mean", "child_need_steps",
                               "child_orphan_steps", "child_died_after_mother", "orphaned_child_alive_at_end",
                               "mother_work_steps", "unnecessary_help_attempts", "safe_refusal_overrides"):
                    row[metric] = sum(item[metric] for item in selected)/len(selected)
                summary.append(row)
        with (destination/"summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
            writer.writeheader(); writer.writerows(summary)
        metadata["status"] = "completed"
        metadata["interpretation"] = "Inspect appropriate_help vs no_help and always_help. These scripted references are not optimal policies or proof of global feasibility."
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        self._emit({"kind": "frame", "overall": 1})
        self._emit({"kind": "done", "status": "Balance check completed · inspect summary.csv before training", "output": str(destination)})


class TrainingPanel(_Panel):
    def __init__(self, config, arms=None, seeds=None):
        super().__init__("Across generations", config)
        self.batch = arms is not None
        self.arm = W.SelectMultiple(options=tuple("ABCD"), value=tuple(arms), description="Arms") if self.batch else W.Dropdown(options=tuple("ABCD"), value=config.arm, description="Arm")
        self.seed_input = W.Text(value=", ".join(map(str, seeds or (config.seed,))), description="Seeds") if self.batch else W.BoundedIntText(value=config.seed, min=0, max=2**31-1, description="Seed")
        self.start.description = "Start batch" if self.batch else "Start training"
        self.backend = W.Dropdown(options=(("Serial", "serial"), ("Threads", "threads"),
                                           ("Multi-core CPU", "processes")),
                                  value=config.execution_backend, description="Backend")
        self.workers = W.BoundedIntText(value=config.family_workers, min=1, max=64, description="Workers")
        self.extra_controls.children = (self.arm, self.seed_input, self.backend, self.workers)

    def _job(self):
        self._check()
        check_ml_dependencies()
        from .evolution import Experiment, StopRequested
        def control():
            try:
                self._check()
            except NotebookStop:
                raise StopRequested() from None
        arms = tuple(self.arm.value) if self.batch else (self.arm.value,)
        seeds = tuple(int(s.strip()) for s in self.seed_input.value.split(",") if s.strip()) if self.batch else (self.seed_input.value,)
        if not arms or not seeds or len(set(seeds)) != len(seeds):
            raise ValueError("Choose at least one arm and distinct integer seeds")
        jobs = [(arm, seed) for arm in arms for seed in seeds]
        configs = []
        results = []
        for arm, seed in jobs:
            config = copy.deepcopy(self.config)
            config.arm, config.seed = arm, seed
            config.execution_backend = self.backend.value
            config.family_workers = 1 if config.execution_backend == "serial" else self.workers.value
            config.validate()
            configs.append(config)
        for index, config in enumerate(configs):
            control()
            arm, seed = config.arm, config.seed
            self._emit({"kind": "log", "text": f"Job {index+1}/{len(jobs)} · arm {arm} · seed {seed}"})
            engine = Experiment(config)
            self._emit({"kind": "log", "text": device_description(engine.config)})
            self._emit({"kind": "log", "text": execution_description(engine.config)})
            if arm == "D":
                self._emit({"kind": "log", "text": "D retains the same mothers across all requested rounds; child extinction does not end this control arm. Compare time per round, not total time against extinct runs."})
            if engine.config.ppo_shuffle_rng == "per_agent":
                self._emit({"kind": "log", "text": "Per-agent PPO shuffle RNG differs from legacy runs. Same seed need not reproduce old results; parallel speedup is not guaranteed."})
            terminal = {}
            def callback(frame):
                if frame["kind"] == "done":
                    terminal.update(frame)
                    self._emit({"kind": "log", "text": f"{arm}/{seed}: {frame['status']} · {frame['output']}"})
                else:
                    payload = dict(frame)
                    payload["overall"] = (index+frame.get("overall", 0))/len(jobs)
                    payload["label"] = f"{arm}/{seed} · "+frame.get("label", "")
                    self._emit(payload)
            engine.run(callback, control)
            results.append(f"{arm}/{seed}: {terminal.get('status', 'unknown')} · {engine.completed_generations}/{config.generations} generations")
            if self.cancel.is_set() or terminal.get("status") == "stopped":
                self._emit({"kind": "done", "status": "Stopped · remaining jobs were not started", "output": str(engine.output)})
                return
        self._emit({"kind": "done", "status": " | ".join(results), "output": str(OUTPUT_ROOT)})


class ResumePanel(_Panel):
    def __init__(self, config):
        super().__init__("Resume a saved run", config)
        self.path = W.Text(description="Checkpoint", placeholder="outputs/full_lifetime/run_.../models/latest.pt", layout=W.Layout(width="100%"))
        self.extra_controls.children = (self.path,)
        self.start.description = "Resume training"

    def _job(self):
        self._check()
        check_ml_dependencies()
        from .evolution import Experiment, StopRequested
        path = _checkpoint_path(self.path.value)
        engine = Experiment.resume(path)
        self._emit({"kind": "log", "text": device_description(engine.config)})
        self._emit({"kind": "log", "text": execution_description(engine.config)})
        def control():
            try:
                self._check()
            except NotebookStop:
                raise StopRequested() from None
        engine.run(self._emit, control)


def _checkpoint_path(value):
    if not value.strip():
        raise ValueError("Enter a trusted local checkpoint path first")
    path = Path(value).expanduser()
    path = path if path.is_absolute() else ROOT/path
    path = path.resolve()
    if not path.is_file() or path.suffix != ".pt":
        raise FileNotFoundError(path)
    return path


class FrozenModelPanel(_Panel):
    def __init__(self, config):
        super().__init__("Frozen model · new situation", config)
        self.start.description = "Show model"
        self.stop.description = "Stop viewing"
        self.path = W.Text(description="Checkpoint", placeholder="outputs/full_lifetime/run_.../models/gen_000.pt", layout=W.Layout(width="100%"))
        self.phase = W.Dropdown(options=("A0", "A1", "A2"), value="A0", description="Phase")
        from .evaluation import SCENARIOS
        self.scenario = W.Dropdown(options=SCENARIOS, value="hungry", description="Scenario", layout=W.Layout(width="350px"))
        self.family = W.BoundedIntText(value=0, min=0, max=10000, description="Family")
        self.extra_controls.children = (self.path, self.phase, self.scenario, self.family)

    def _job(self):
        self._check()
        check_ml_dependencies()
        from .agent import Learner
        from .evolution import load_checkpoint, source_hashes
        from .evaluation import scenario_world, benchmark_child, apply_scenario_events, evaluation_config
        payload = load_checkpoint(_checkpoint_path(self.path.value))
        config = evaluation_config(payload)
        self._emit({"kind": "log", "text": device_description(config)})
        index = self.family.value
        if not 0 <= index < len(payload["families"]):
            raise ValueError("Family index is outside checkpoint population")
        item = payload["families"][index]
        if item["world"].t != config.world.horizon:
            raise ValueError("Choose a completed-generation checkpoint for matched A0/A1/A2 viewing")
        phase = self.phase.value
        state = item["mother"] if phase == "A2" else payload["starts"][index]
        source = Learner.restore(config, state)
        actor = source.evaluation_copy(naive=phase == "A0", seed=500_000)
        body = copy.deepcopy(item["world"].mother if phase == "A2" else payload["start_bodies"][index])
        body.forage, body.avoidance = 0.5, 0.3
        world = scenario_world(config, self.scenario.value, 0, body)
        limit = world.config.horizon
        for _ in range(limit):
            self._check()
            apply_scenario_events(world, self.scenario.value)
            action, _ = actor.choose(world.observation(0)) if world.mother.alive else (Action.WAIT, None)
            world.step((action, benchmark_child(world, self.scenario.value)))
            self._emit({"kind": "frame", "world": world.snapshot(), "overall": world.t/limit,
                        "episode_fraction": world.t/limit, "z_mother": actor.z.tolist(),
                        "label": f"FROZEN {phase} · {self.scenario.value} · NO LEARNING"})
        self._emit({"kind": "done", "status": "Frozen viewing finished · fresh test world, not a replay of the original trajectory"})


class EvaluationPanel(_Panel):
    def __init__(self, config):
        super().__init__("Frozen evaluation", config)
        self.start.description = "Start evaluation"
        self.stop.description = "Stop evaluation"
        self.path = W.Text(description="Checkpoint", placeholder="outputs/full_lifetime/run_.../models/gen_000.pt", layout=W.Layout(width="100%"))
        from .evaluation import INTERVENTIONS
        self.interventions = W.SelectMultiple(options=INTERVENTIONS, value=("normal",), description="Conditions", layout=W.Layout(width="100%", height="120px"))
        self.split = W.Dropdown(options=("monitor", "holdout"), value="monitor", description="Test split")
        self.skills = W.Dropdown(options=("matched", "native"), value="matched", description="Skills")
        self.parallel_eval = W.Checkbox(value=config.parallel_zero_shot and config.execution_backend == "processes",
                                       description="Multi-core CPU")
        self.eval_workers = W.BoundedIntText(value=config.family_workers, min=1, max=64, description="Workers")
        self.extra_controls.children = (self.path, self.interventions, self.split, self.skills,
                                        self.parallel_eval, self.eval_workers)

    def _job(self):
        self._check()
        check_ml_dependencies()
        from .agent import Learner
        from .evolution import load_checkpoint, source_hashes, append_csv
        from .evaluation import evaluate_phase, evaluation_config, EVALUATION_PROTOCOL
        path = _checkpoint_path(self.path.value)
        payload = load_checkpoint(path)
        config = evaluation_config(payload)
        config.parallel_zero_shot = self.parallel_eval.value
        if config.parallel_zero_shot:
            config.execution_backend, config.device = "processes", "cpu"
            config.family_workers = self.eval_workers.value
        config.validate()
        self._emit({"kind": "log", "text": device_description(config)})
        self._emit({"kind": "log", "text": execution_description(config)})
        if not payload["families"] or any(item["world"].t != config.world.horizon for item in payload["families"]):
            raise ValueError("Choose a completed care episode with A0/A1/A2 snapshots")
        if not self.interventions.value:
            raise ValueError("Select at least one condition")
        generation = payload["families"][0]["world"].mother.generation
        destination = path.parent.parent/f"evaluations_full_lifetime_{time.time_ns()}.csv"
        destination.with_suffix(".json").write_text(json.dumps({
            "evaluation_protocol": EVALUATION_PROTOCOL, "checkpoint": str(path), "status": "running",
            "saved_eval_steps": payload["config"].get("eval_steps"), "eval_steps": config.world.horizon,
            "source_hashes": source_hashes(), "checkpoint_source_hashes": payload["source_hashes"],
            "weights_frozen": True, "memory_reset": True, "runtime_device": config.device,
            "evaluation_backend": "processes" if config.parallel_zero_shot and config.execution_backend == "processes" else "serial",
            "evaluation_workers": min(config.family_workers, available_cpu_count()) if config.parallel_zero_shot else 1,
            "test_split": self.split.value, "skill_regime": self.skills.value,
            "note": "New evaluation only. Legacy CSVs and training checkpoint are unchanged."
        }, indent=2), encoding="utf-8")
        for phase_index, phase in enumerate(("A0", "A1", "A2")):
            records = []
            for index, item in enumerate(payload["families"]):
                state = item["mother"] if phase == "A2" else payload["starts"][index]
                body = item["world"].mother if phase == "A2" else payload["start_bodies"][index]
                records.append((Learner.restore(config, state), body))
            def progress(world, fraction):
                self._emit({"kind": "frame", "world": world.snapshot(),
                            "overall": (phase_index+fraction)/3,
                            "episode_fraction": world.t/config.world.horizon,
                            "label": f"FROZEN {phase} · {world.care_test.get('care_policy', 'learned')} · {self.split.value} · NO LEARNING"})
            for row in evaluate_phase(records, config, generation, phase, control=self._check, progress=progress,
                                      interventions=tuple(self.interventions.value), skill_regime=self.skills.value,
                                      test_split=self.split.value):
                append_csv(destination, row)
        metadata = json.loads(destination.with_suffix(".json").read_text(encoding="utf-8"))
        metadata["status"] = "completed"
        destination.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        self._emit({"kind": "frame", "overall": 1})
        self._emit({"kind": "done", "status": "Frozen evaluation completed · no weights changed", "output": str(destination)})


class ReportsPanel:
    def __init__(self):
        self.thread = None
        self.run = W.Dropdown(options=[("Choose a run", "")], description="Run", layout=W.Layout(width="100%"))
        self.multi = W.SelectMultiple(options=[], description="Compare", layout=W.Layout(width="100%", height="110px"))
        self.refresh = W.Button(description="Refresh runs")
        self.plot = W.Button(description="Build run report")
        self.compare = W.Button(description="Compare selected")
        self.output = W.Output()
        self.refresh.on_click(self._refresh)
        self.plot.on_click(lambda _: self._launch(False))
        self.compare.on_click(lambda _: self._launch(True))
        self.widget = W.VBox([self.run, self.multi, W.HBox([self.refresh, self.plot, self.compare]), self.output])

    def _refresh(self, _):
        options = []
        for path in sorted((ROOT/"outputs").rglob("manifest.json")):
            if not path.parent.name.startswith("run_"):
                continue
            try:
                manifest = json.loads(path.read_text(encoding="utf-8"))
                if manifest["status"] in ("completed", "extinct", "stopped") and (path.parent/"metrics.csv").is_file():
                    environment = "home" if manifest.get("protocol") == PROTOCOL else "previous environment"
                    saved_config = json.loads((path.parent/"config.json").read_text(encoding="utf-8"))
                    environment += f" · recorded test {saved_config.get('eval_steps', '?')} steps"
                    label = path.parent.relative_to(ROOT/"outputs").as_posix()
                    options.append((label+" · "+manifest["status"]+" · "+environment, str(path.parent)))
            except (OSError, ValueError, KeyError):
                continue
        self.run.options = [("Choose a run", "")]+options
        self.multi.options = options
        self.output.append_stdout(f"Found {len(options)} saved runs with metrics.\n")

    def _launch(self, compare):
        if self.thread and self.thread.is_alive():
            return
        selected = tuple(self.multi.value) if compare else (self.run.value,)
        if not selected or not all(selected):
            self.output.append_stderr("Select saved run(s) first.\n")
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self.output.append_stderr("Open these controls inside an active Jupyter kernel.\n")
            return
        if not _RUN_LOCK.acquire(blocking=False):
            self.output.append_stderr("Stop the active job before building reports.\n")
            return
        future = loop.create_future()
        def work():
            try:
                from .evaluation import compare_runs, plot_run
                result = compare_runs(selected, OUTPUT_ROOT/f"comparison_{time.time_ns()}") if compare else plot_run(selected[0])
                loop.call_soon_threadsafe(future.set_result, Path(result))
            except Exception as exc:
                loop.call_soon_threadsafe(future.set_exception, exc)
            finally:
                _RUN_LOCK.release()
        self.thread = threading.Thread(target=work, daemon=True)
        self.thread.start()
        async def finish():
            try:
                result = await future
                self.output.append_stdout(str(result)+"\n")
                if result.suffix == ".png":
                    self.output.append_display_data(Image(filename=str(result)))
                    if result.name.startswith("a0_"):
                        for picture in sorted(result.parent.glob("a0_*.png")):
                            if picture != result:
                                self.output.append_display_data(Image(filename=str(picture)))
                        self.output.append_stdout("Primary: A0 frozen tests. comparison.png contains supplementary training results.\n")
                else:
                    relative = result.relative_to(ROOT).as_posix()
                    self.output.append_display_data(HTML(f'<a href="{_escape(relative)}" target="_blank">Open HTML report</a>'))
                    for picture in sorted((result.parent/"plots").glob("*.png")):
                        self.output.append_display_data(Image(filename=str(picture)))
            except Exception as exc:
                self.output.append_stderr(f"{type(exc).__name__}: {exc}\n")
        loop.create_task(finish())
