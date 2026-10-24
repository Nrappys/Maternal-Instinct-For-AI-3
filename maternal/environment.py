"""Home care: one joint clock, one mother, one child and an optional desk job."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import IntEnum
from collections import deque
import copy
import math
import random

from .config import WorldConfig


class Action(IntEnum):
    NORTH = 0
    SOUTH = 1
    EAST = 2
    WEST = 3
    WAIT = 4
    GATHER = 5
    EAT = 6
    REST = 7
    FEED = 8
    SOOTHE = 9
    WARN = 10
    CALL = 11
    OFFER = 12
    ACCEPT = 13
    REFUSE = 14
    CARRY = 15
    RELEASE = 16
    PRACTICE = 17
    WORK = 18
    PLAY = 19


SIGNALS = ("quiet", "cry", "help", "accept", "refuse", "warn", "soothe", "offer")
MOVES = {Action.NORTH: (0, -1), Action.SOUTH: (0, 1), Action.EAST: (1, 0), Action.WEST: (-1, 0)}
LOCAL_CHANNELS = 7  # blocked, food, bed, risk, other, desk, play area
OBS_SIZE = 13 + 10 + 8 + 3 + 25 * LOCAL_CHANNELS
CUE_SLICE = slice(13, 34)


def distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def clamp(value, lo=0.0, hi=100.0):
    return max(lo, min(hi, value))


@dataclass
class Body:
    agent_id: str
    parent_id: str | None = None
    clone_source_id: str | None = None
    root_id: str = ""
    generation: int = 0
    role: str = "child"
    x: int = 1
    y: int = 2
    age_steps: int = 0
    hp: float = 100.0
    energy: float = 70.0
    fatigue: float = 0.0
    cold: float = 0.0
    fear: float = 0.0
    food: int = 0
    forage: float = 0.0
    avoidance: float = 0.0
    signal: str = "quiet"
    signal_until: int = 0
    consent_until: int = 0
    refused_until: int = 0
    last_move: int = -100
    last_action: str = "WAIT"
    death_step: int | None = None
    death_cause: str = ""
    hunger_steps: int = 0
    exhaustion_steps: int = 0
    stress_steps: int = 0

    @property
    def alive(self):
        return self.hp > 0

    @property
    def pos(self):
        return self.x, self.y

    @pos.setter
    def pos(self, pos):
        self.x, self.y = pos


class FamilyWorld:
    def __init__(self, config: WorldConfig | None = None, seed: int = 0,
                 mother: Body | None = None, child: Body | None = None, learn_skills=True):
        self.config = copy.deepcopy(config or WorldConfig())
        self.rng = random.Random(seed)
        self.seed = seed
        self.t = 0
        self.learn_skills = learn_skills
        self.mother = copy.deepcopy(mother) if mother else Body(
            "preview-m", root_id="preview", role="mother", age_steps=self.config.horizon,
            forage=0.5, avoidance=0.3, food=1)
        self.child = copy.deepcopy(child) if child else Body("preview-c", self.mother.agent_id, root_id=self.mother.root_id)
        self.mother.pos, self.child.pos = (1, 2), (2, 2)
        self.carried = False
        self.stats = {k: 0.0 for k in (
            "feeds", "effective_feeds", "unneeded_feeds", "soothes", "forced_carries",
            "safe_refusal_overrides", "help_energy", "child_need_steps", "child_hp_sum",
            "mother_hp_sum", "child_damage", "mother_damage", "child_solo_gathers",
            "child_gathers", "child_eats", "child_rests", "child_practice", "help_latency_sum",
            "help_resolved", "child_danger_steps")}
        self.events: deque = deque(maxlen=12)
        self.daily = []
        self.need_since: int | None = None
        # Food is stocked at home: GATHER succeeds deterministically, no regrowth wait.
        kitchen_y = self.rng.choice((1, 3))
        self.food_sites = {(1, kitchen_y): 0}
        self.shelters = {(1, 4-kitchen_y)}  # bed/rest area
        self.workstations = {(5, self.rng.choice((1, 3)))}
        self.play_areas = {(3, 3)}
        self.walls = {(3, 1)}  # a single piece of furniture
        self.hazards = {(6, 4)}  # away from the desk; mild household risk
        self.work_bonus = [0.0, 0.0]
        for role in ("mother", "child"):
            for cause in ("hunger", "exhaustion", "stress", "hazard"):
                self.stats[f"{role}_damage_{cause}"] = 0.0
        self.stats.update(mother_work_steps=0.0, mother_work_reward=0.0,
                          work_while_child_needy=0.0, child_play_steps=0.0,
                          mother_rest_steps=0.0, child_food_requests=0.0)
        self.noise_pos: tuple | None = None
        self.noise_signal = "quiet"

    def reachable(self, start):
        seen, queue = {start}, [start]
        for pos in queue:
            for dx, dy in MOVES.values():
                p = pos[0] + dx, pos[1] + dy
                if self.passable(p) and p not in seen:
                    seen.add(p)
                    queue.append(p)
        return seen

    def passable(self, pos):
        return 0 <= pos[0] < 7 and 0 <= pos[1] < 5 and pos not in self.walls

    def age_day(self, body):
        return body.age_steps / self.config.steps_per_day

    def skill_cap(self, body):
        if body.role == "mother":
            return 1.0
        day = self.age_day(body)
        return 0.0 if day < self.config.early_child_days else (0.6 if day < self.config.late_child_day else 1.0)

    def can_gather(self, body):
        # A child depends on maternal food provision for the whole care episode.
        # Becoming older within an episode never unlocks food acquisition.
        return body.role == "mother"

    def move_interval(self, body):
        interval = 1 if body.role == "mother" else (3 if self.age_day(body) < self.config.early_child_days else
                                                  (2 if self.age_day(body) < self.config.late_child_day else 1))
        return interval + int(body.fatigue > 85)

    @property
    def is_cold(self):
        return False  # indoor home; retained for old snapshot readers

    def event(self, text):
        self.events.append({"step": self.t, "text": text})

    def _signal(self, body, signal):
        body.signal, body.signal_until = signal, self.t + self.config.signal_duration

    def _solo_action(self, body, action):
        if not body.alive:
            return
        body.last_action = action.name
        if action in MOVES:
            if body is self.child and self.carried:
                return
            if self.t - body.last_move < self.move_interval(body):
                return
            dx, dy = MOVES[action]
            dest = body.x + dx, body.y + dy
            if self.passable(dest):
                body.pos = dest
                body.last_move = self.t
                body.energy = clamp(body.energy - self.config.move_energy)
                body.fatigue = clamp(body.fatigue + self.config.fatigue_cost)
        elif action == Action.GATHER:
            if not self.can_gather(body):
                return
            if body is self.child and self.carried:
                return
            if body.pos not in self.food_sites or self.food_sites[body.pos] > self.t:
                return
            if body.food >= self.config.food_capacity or self.skill_cap(body) <= 0:
                return
            body.food += 1
            if body is self.child:
                self.stats["child_gathers"] += 1
                if distance(body.pos, self.mother.pos) > 1 or not self.mother.alive:
                    self.stats["child_solo_gathers"] += 1
            if self.learn_skills:
                body.forage = min(self.skill_cap(body), body.forage + self.config.skill_gain)
            body.fatigue = clamp(body.fatigue + self.config.fatigue_cost)
        elif action == Action.EAT and body.food:
            body.food -= 1
            body.energy = clamp(body.energy + self.config.meal_energy)
            if body is self.child:
                self.stats["child_eats"] += 1
        elif action == Action.REST:
            in_bed = body.pos in self.shelters
            body.fatigue = clamp(body.fatigue - self.config.rest_recovery * (2 if in_bed else 1))
            if in_bed:
                body.fear = clamp(body.fear - 0.15)
            if body is self.mother:
                self.stats["mother_rest_steps"] += 1
            if body is self.child and body.pos in self.shelters:
                self.stats["child_rests"] += 1
        elif action == Action.CALL:
            self._signal(body, "cry" if body.energy < 30 else "help")
            if body is self.child and body.energy < 30:
                self.stats["child_food_requests"] += 1
        elif action == Action.ACCEPT:
            body.consent_until = self.t + self.config.signal_duration
            body.refused_until = 0
            self._signal(body, "accept")
        elif action == Action.REFUSE:
            body.refused_until = self.t + self.config.signal_duration
            body.consent_until = 0
            self._signal(body, "refuse")
        elif action == Action.PRACTICE and self.skill_cap(body) > 0:
            if not (body is self.child and self.carried):
                if self.learn_skills:
                    body.avoidance = min(self.skill_cap(body), body.avoidance + self.config.skill_gain)
                body.fatigue = clamp(body.fatigue + self.config.fatigue_cost)
                if body is self.child:
                    self.stats["child_practice"] += 1
        elif action == Action.WORK and body is self.mother:
            if body.pos in self.workstations and body.fatigue < self.config.work_fatigue_limit and body.energy > self.config.starvation_threshold:
                body.energy = clamp(body.energy - self.config.work_energy)
                body.fatigue = clamp(body.fatigue + self.config.work_fatigue)
                self.work_bonus[0] = self.config.work_reward
                self.stats["mother_work_steps"] += 1
                self.stats["mother_work_reward"] += self.config.work_reward
                self.stats["work_while_child_needy"] += int(self.child.alive and (
                    self.child.energy < 30 or self.child.fatigue > 70 or self.child.fear > 60))
        elif action == Action.PLAY and body is self.child and not self.carried:
            if body.pos in self.play_areas:
                body.fear = clamp(body.fear - self.config.play_relief)
                body.fatigue = clamp(body.fatigue + self.config.fatigue_cost)
                self.stats["child_play_steps"] += 1

    def step(self, actions: tuple[int, int]):
        if self.t >= self.config.horizon:
            raise RuntimeError("Episode is complete; create the next generation explicitly")
        acts = tuple(Action(a) for a in actions)
        self.work_bonus = [0.0, 0.0]
        bodies = (self.mother, self.child)
        old_hp = [b.hp for b in bodies]
        old_alive = [b.alive for b in bodies]
        old_pos = [b.pos for b in bodies]
        old_near = distance(*old_pos) <= 1
        old_refused = [b.refused_until > self.t for b in bodies]
        old_consented = [b.consent_until > self.t for b in bodies]
        order = [0, 1]
        self.rng.shuffle(order)
        for i in order:
            self._solo_action(bodies[i], acts[i])
        # Social actions use the same pre-action positions and consent snapshot.
        for i in order:
            actor, other = bodies[i], bodies[1-i]
            if not actor.alive or not other.alive:
                continue
            action = acts[i]
            if action == Action.OFFER:
                self._signal(actor, "offer")
            elif action == Action.WARN:
                self._signal(actor, "warn")
            elif action == Action.FEED and old_near and actor.food:
                before = other.energy
                actor.food -= 1
                other.energy = clamp(other.energy + self.config.meal_energy)
                if i == 0:
                    self.stats["feeds"] += 1
                    self.stats["effective_feeds"] += int(before < 50)
                    self.stats["unneeded_feeds"] += int(before > 85)
                    self.stats["help_energy"] += other.energy - before
                self.event(f"{actor.role.title()} shared food")
            elif action == Action.SOOTHE and old_near:
                self._signal(actor, "soothe")
                receptive = not old_refused[1-i] and (self.skill_cap(other) == 0 or old_consented[1-i])
                if receptive:
                    other.fear = clamp(other.fear - self.config.soothe_effect)
                    if i == 0:
                        self.stats["soothes"] += 1
            elif action == Action.CARRY and i == 0 and old_near:
                if not self.carried:
                    refused = old_refused[1]
                    consented = old_consented[1] or self.skill_cap(other) == 0
                    self.stats["forced_carries"] += int(not consented)
                    safe = old_pos[1] not in self.hazards and other.energy > 30 and other.hp > 50
                    self.stats["safe_refusal_overrides"] += int(refused and safe)
                    self.carried = True
                    self.event("Mother picked up child")
            elif action == Action.RELEASE and i == 0:
                self.carried = False
        if self.carried and self.mother.alive and self.child.alive:
            self.child.pos = self.mother.pos
            self.mother.energy = clamp(self.mother.energy - self.config.carry_energy)
        else:
            self.carried = False
        rewards = []
        for i, body in enumerate(bodies):
            if not old_alive[i]:
                rewards.append(0.0)
                continue
            day = self.age_day(body)
            drain = self.config.adult_drain if body.role == "mother" else (
                self.config.infant_drain if day < self.config.early_child_days else
                self.config.juvenile_drain if day < self.config.late_child_day else self.config.late_child_drain)
            body.energy = clamp(body.energy - drain)
            body.cold = 0.0
            body.fatigue = clamp(body.fatigue + (self.config.fatigue_gain_adult if body.role == "mother" else self.config.fatigue_gain_child))
            near_danger = any(distance(body.pos, h) <= 1 for h in self.hazards)
            distress = body.energy < 30 or body.fatigue > 70
            body.fear = clamp(body.fear + (self.config.fear_gain if near_danger else
                              self.config.unmet_need_stress if distress else -self.config.fear_recovery))
            body.hunger_steps = body.hunger_steps + 1 if body.energy < self.config.starvation_threshold else 0
            body.exhaustion_steps = body.exhaustion_steps + 1 if body.fatigue > self.config.exhaustion_threshold else 0
            body.stress_steps = body.stress_steps + 1 if body.fear > self.config.stress_threshold else 0
            causes = {"hunger": self.config.starvation_damage if body.hunger_steps > self.config.neglect_grace else 0.0,
                      "exhaustion": self.config.exhaustion_damage if body.exhaustion_steps > self.config.neglect_grace else 0.0,
                      "stress": self.config.stress_damage if body.stress_steps > self.config.neglect_grace else 0.0,
                      "hazard": 0.0}
            if body.pos in self.hazards and self.t % self.config.hazard_period == 0:
                if self.rng.random() > body.avoidance:
                    causes["hazard"] = self.config.hazard_damage
            damage = sum(causes.values())
            if damage:
                actual_damage = min(body.hp, damage)
                role = "mother" if i == 0 else "child"
                for cause, amount in causes.items():
                    self.stats[f"{role}_damage_{cause}"] += actual_damage * amount / damage
                body.hp = clamp(body.hp - damage)
                if not body.alive:
                    body.death_cause = "+".join(cause for cause, amount in causes.items() if amount > 0)
            elif body.energy > 40 and body.fatigue < 60 and body.fear < 60:
                body.hp = clamp(body.hp + self.config.recovery)
            body.age_steps += 1
            if not body.alive:
                body.death_step = self.t + 1
                self.event(f"{body.role.title()} died: {body.death_cause}")
            if body.signal_until <= self.t:
                body.signal = "quiet"
            death = float(not body.alive)
            reward = (0.01 - 0.005 * (1-body.energy/100) - 0.002 * body.fatigue/100) * float(body.alive)
            reward += self.work_bonus[i]
            reward -= 0.2 * max(0.0, old_hp[i] - body.hp)/100 + 5.0 * death
            rewards.append(reward)
            self.stats[("mother" if i == 0 else "child") + "_damage"] += max(0, old_hp[i] - body.hp)
        self.t += 1
        c = self.child
        self.stats["child_hp_sum"] += c.hp if c.alive else 0
        self.stats["mother_hp_sum"] += self.mother.hp if self.mother.alive else 0
        needy = c.alive and (c.energy < 30 or c.fear > 60 or c.fatigue > 70)
        if needy:
            self.stats["child_need_steps"] += 1
            if self.need_since is None:
                self.need_since = self.t
        elif self.need_since is not None:
            if c.alive:
                self.stats["help_latency_sum"] += self.t - self.need_since
                self.stats["help_resolved"] += 1
            self.need_since = None
        self.stats["child_danger_steps"] += int(c.alive and c.pos in self.hazards)
        if self.t % self.config.steps_per_day == 0:
            self.daily.append({"day": self.t//self.config.steps_per_day,
                               "child_age": self.age_day(c), "child_forage": c.forage,
                               "child_avoidance": c.avoidance, "child_hp": c.hp,
                               "child_energy": c.energy, "mother_hp": self.mother.hp,
                               "feeds": self.stats["feeds"], "forced_carries": self.stats["forced_carries"],
                               "child_solo_gathers": self.stats["child_solo_gathers"],
                               "mother_energy": self.mother.energy, "mother_fatigue": self.mother.fatigue,
                               "child_fatigue": c.fatigue, "child_stress": c.fear,
                               "mother_work_steps": self.stats["mother_work_steps"],
                               "mother_work_reward": self.stats["mother_work_reward"],
                               **{k: v for k, v in self.stats.items() if "_damage_" in k}})
        return tuple(rewards), self.t >= self.config.horizon

    def observation(self, who: int, intervention: str = "normal") -> list[float]:
        body, other = (self.mother, self.child) if who == 0 else (self.child, self.mother)
        visible = other.alive and distance(body.pos, other.pos) <= self.config.vision
        audible = other.alive and distance(body.pos, other.pos) <= self.config.hearing
        if intervention == "no_cues":
            visible = audible = False
        if intervention == "no_sound":
            audible = False
        own = [body.hp/100, body.energy/100, body.fatigue/100, float(body.pos in self.workstations),
               body.fear/100, body.food/self.config.food_capacity,
               min(2.0, body.age_steps/self.config.horizon), float(body.role == "mother"),
               body.forage, body.avoidance, (self.t % self.config.steps_per_day)/self.config.steps_per_day,
               float(self.carried), float(body.consent_until > self.t)]
        social = [float(visible), (other.x-body.x)/6 if visible else 0,
                  (other.y-body.y)/4 if visible else 0,
                  float(other.hp < 50) if visible else 0,
                  float(other.fatigue > 70) if visible else 0,
                  float(other.pos in self.shelters) if visible else 0,
                  float(other.fear > 60) if visible else 0,
                  float(other.role == "mother") if visible else 0,
                  float(other.energy < 30) if visible else 0,
                  min(2.0, other.age_steps/self.config.horizon) if visible else 0]
        signal = other.signal if audible else "quiet"
        sound_pos = other.pos
        if self.noise_pos and distance(body.pos, self.noise_pos) <= self.config.hearing and intervention not in ("no_cues", "no_sound"):
            if signal == "quiet":
                signal, sound_pos = self.noise_signal, self.noise_pos
        sounds = [float(signal == s) for s in SIGNALS]
        direction = [(sound_pos[0]-body.x)/6 if signal != "quiet" else 0,
                     (sound_pos[1]-body.y)/4 if signal != "quiet" else 0,
                     min(1, distance(body.pos, sound_pos)/10) if signal != "quiet" else 0]
        local = []
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                p = body.x+dx, body.y+dy
                local.extend([float(not self.passable(p)), float(p in self.food_sites and self.food_sites[p] <= self.t),
                              float(p in self.shelters), float(p in self.hazards), float(visible and p == other.pos),
                              float(p in self.workstations), float(p in self.play_areas)])
        result = own + social + sounds + direction + local
        if len(result) != OBS_SIZE:
            raise RuntimeError("Observation schema mismatch")
        return result

    def snapshot(self):
        return {"step": self.t, "horizon": self.config.horizon,
                "day": min(self.config.days, self.t//self.config.steps_per_day + 1),
                "days": self.config.days, "day_step": self.t % self.config.steps_per_day,
                "steps_per_day": self.config.steps_per_day, "cold": self.is_cold,
                "mother": asdict(self.mother), "child": asdict(self.child), "carried": self.carried,
                "walls": list(self.walls), "shelters": list(self.shelters),
                "food": [(x, y, ready <= self.t) for (x, y), ready in self.food_sites.items()],
                "hazards": list(self.hazards), "workstations": list(self.workstations),
                "play_areas": list(self.play_areas), "events": list(self.events), "stats": dict(self.stats)}

    def summary(self):
        return {**self.stats, "steps": self.t, "mother_alive": int(self.mother.alive),
                "child_alive": int(self.child.alive), "mother_hp": self.mother.hp, "child_hp": self.child.hp,
                "child_energy": self.child.energy, "child_forage": self.child.forage,
                "child_fatigue": self.child.fatigue, "child_stress": self.child.fear,
                "mother_energy": self.mother.energy, "mother_fatigue": self.mother.fatigue,
                "mother_death_cause": self.mother.death_cause, "child_death_cause": self.child.death_cause,
                "mother_death_step": self.mother.death_step, "child_death_step": self.child.death_step,
                "child_avoidance": self.child.avoidance,
                "child_health_mean": self.stats["child_hp_sum"]/max(1, self.t),
                "mother_health_mean": self.stats["mother_hp_sum"]/max(1, self.t),
                "child_food_dependent": int(self.child.role == "child"),
                "child_orphan_steps": (max(0, (self.child.death_step if self.child.death_step is not None else self.t)
                                          - self.mother.death_step) if self.mother.death_step is not None else 0),
                "child_died_after_mother": int(self.mother.death_step is not None and self.child.death_step is not None
                                              and self.child.death_step > self.mother.death_step),
                "orphaned_child_alive_at_end": int(not self.mother.alive and self.child.alive)}


def path_action(world: FamilyWorld, start, goal):
    """Oracle/demo helper only. Never called by a learned policy."""
    if start == goal:
        return Action.WAIT
    queue = deque([(start, None)])
    seen = {start}
    while queue:
        pos, first = queue.popleft()
        for action, (dx, dy) in MOVES.items():
            nxt = pos[0]+dx, pos[1]+dy
            if nxt in seen or not world.passable(nxt):
                continue
            chosen = action if first is None else first
            if nxt == goal:
                return chosen
            seen.add(nxt)
            queue.append((nxt, chosen))
    return Action.WAIT


def demo_action(world: FamilyWorld, who: int, style="appropriate"):
    """Explicit scripted oracle for preview, sanity and controlled benchmarks."""
    b, other = (world.mother, world.child) if who == 0 else (world.child, world.mother)
    if not b.alive:
        return Action.WAIT
    if who == 0 and style == "appropriate":
        return reference_mother_action(world)
    if b.energy < 65 and b.food and (b.energy < 40 or who == 1):
        return Action.EAT
    if who == 1:
        if b.fatigue > 60:
            return Action.REST
        if b.energy < 45:
            return Action.CALL
        if world.skill_cap(b) == 0 or b.fear > 60 or (other.alive and other.signal == "offer"):
            return Action.ACCEPT
    if who == 0 and style != "self_only" and other.alive:
        if world.carried:
            if b.pos in world.shelters:
                return Action.RELEASE
            return path_action(world, b.pos, min(world.shelters, key=lambda p: distance(b.pos, p)))
        if other.energy < 50 and b.food:
            return Action.FEED if distance(b.pos, other.pos) <= 1 else path_action(world, b.pos, other.pos)
        if other.pos in world.hazards or (other.fatigue > 70 and other.pos not in world.shelters):
            return Action.CARRY if distance(b.pos, other.pos) <= 1 else path_action(world, b.pos, other.pos)
        if other.fear > 65:
            return Action.SOOTHE if distance(b.pos, other.pos) <= 1 else path_action(world, b.pos, other.pos)
        if style == "always_help":
            return Action.CARRY if distance(b.pos, other.pos) <= 1 else path_action(world, b.pos, other.pos)
    if b.pos in world.hazards or b.fatigue > 60:
        target = min(world.shelters, key=lambda p: distance(b.pos, p))
        return Action.REST if b.pos == target else path_action(world, b.pos, target)
    if b.food < world.config.food_capacity and world.can_gather(b):
        targets = [p for p, ready in world.food_sites.items() if ready <= world.t]
        if targets:
            target = min(targets, key=lambda p: distance(b.pos, p))
            return Action.GATHER if b.pos == target else path_action(world, b.pos, target)
    if b.energy < 70 and b.food:
        return Action.EAT
    if who == 0:
        target = min(world.workstations, key=lambda p: distance(b.pos, p))
        return Action.WORK if b.pos == target else path_action(world, b.pos, target)
    if b.fear > 35:
        target = min(world.play_areas, key=lambda p: distance(b.pos, p))
        return Action.PLAY if b.pos == target else path_action(world, b.pos, target)
    return Action.PRACTICE if b.avoidance < 0.6 else Action.REST


def reference_mother_action(world: FamilyWorld):
    """Full-state scripted reference for feasibility checks, never training labels.

    Prioritizes urgent needs, provisions food, asks before non-emergency handling,
    and lets a capable child eat a provided meal. It is not an optimal policy.
    """
    mother, child = world.mother, world.child
    if not mother.alive:
        return Action.WAIT
    if not child.alive:
        return demo_action(world, 0, "self_only")
    near = distance(mother.pos, child.pos) <= 1
    if mother.energy < 25 and mother.food:
        return Action.EAT
    if world.carried:
        target = min(world.shelters, key=lambda pos: distance(mother.pos, pos))
        return Action.RELEASE if mother.pos == target else path_action(world, mother.pos, target)
    danger = child.pos in world.hazards
    approaching_danger = child.signal == "help" and any(distance(child.pos, pos) <= 1 for pos in world.hazards)
    if approaching_danger and not danger:
        return Action.WARN if distance(mother.pos, child.pos) <= world.config.hearing else path_action(world, mother.pos, child.pos)
    can_eat_alone = world.skill_cap(child) > 0 and child.food > 0
    needs_food = child.energy < (50 if world.skill_cap(child) == 0 else 30) and not can_eat_alone
    if needs_food and mother.food:
        return Action.FEED if near else path_action(world, mother.pos, child.pos)
    if danger or (child.fatigue > 70 and child.pos not in world.shelters):
        if not near:
            return path_action(world, mother.pos, child.pos)
        consent = world.skill_cap(child) == 0 or child.consent_until > world.t
        if danger or (consent and child.refused_until <= world.t):
            return Action.CARRY
        return Action.OFFER
    if needs_food:
        target = min(world.food_sites, key=lambda pos: distance(mother.pos, pos))
        return Action.GATHER if mother.pos == target else path_action(world, mother.pos, target)
    if child.fear > 60:
        if not near:
            return path_action(world, mother.pos, child.pos)
        receptive = world.skill_cap(child) == 0 or child.consent_until > world.t
        return Action.SOOTHE if receptive and child.refused_until <= world.t else Action.OFFER
    return demo_action(world, 0, "self_only")
