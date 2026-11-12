"""Reproducible training datasets and multi-turn teacher trajectories."""
from __future__ import annotations

from dataclasses import dataclass, replace
from random import Random

from .config import Config
from .game import Action, State, World, initial_state, legal_actions, outcome, transition
from .generate import deadlock, derived_seed, generate
from .graph import Graph, Query, TransitionSpec, observe, transition_spec
from .solver import Solution, solve
from .tasks import (Observation, RelationCase, ValueTarget, balanced, controlled, game_queries,
                    observation, pool_targets, value_target)


@dataclass
class Step:
    obs: Observation
    state: State
    action: Action | None
    after: Graph | None
    spec: TransitionSpec | None = None
    queries: tuple[Query, ...] = ()
    changed_queries: frozenset[tuple] = frozenset()
    pool: tuple[float | None, ...] | None = None
    value: ValueTarget | None = None
    best: tuple[int, ...] = ()
    optimal: bool = False
    solvability_weight: float = 1.


@dataclass
class Episode:
    world: World
    steps: list[Step]


def relation_data(cfg: Config, split: str) -> list[RelationCase]:
    total = cfg.data.relation_train if split == "train" else cfg.data.relation_val
    dataset = []
    for index in range(total):
        seed = derived_seed(cfg.runtime.seed, "relation", split, index)
        recipes = cfg.generation.recipes
        recipe = cfg.generation.required_recipe or recipes[index % len(recipes)]
        world = generate(seed, replace(cfg.generation, recipes=(recipe,)),
                         cfg.search, split, verify=False).world
        state = initial_state(world)
        queries = balanced(game_queries(world, state), Random(seed), cfg.data.queries_per_class)
        if queries:
            dataset.append(RelationCase(observe(world, state), queries))
        if split == "train":
            dataset.extend(controlled(seed, split, "curriculum", level) for level in range(1, 6))
        else:
            dataset.extend(controlled(seed, split, axis, level)
                           for axis, level in (("hops", index % 5 + 1), ("ands", 2), ("ors", 2)))
    return dataset


def walkthrough(world: World, solution: Solution, cap: int | None = None) -> list[tuple[State, Action]]:
    current = initial_state(world)
    steps = []
    for action in solution.actions[:cap]:
        steps.append((current, action))
        current, _ = transition(world, current, action)
    return steps


def exploratory(world: World, path: list[tuple[State, Action]], rng: Random, extra: int) -> list[tuple[State, Action]]:
    probes = []
    for index, (current, _) in enumerate(path):
        for action in legal_actions(world, current):
            if action.kind == "use":
                after, _ = transition(world, current, action, checked=True)
                if any(a > 0 == b for new, old in zip(after.statuses, current.statuses) for a, b in zip(new, old)):
                    probes.append((index, action))
    if probes:
        offset, action = rng.choice(probes)
    else:
        offset = rng.randrange(len(path)) if path else 0
        current = path[offset][0] if path else initial_state(world)
        action = rng.choice(legal_actions(world, current))
    steps = path[:offset]
    current = path[offset][0] if path else initial_state(world)
    for _ in range(extra):
        if outcome(world, current) != "running":
            break
        steps.append((current, action))
        current, _ = transition(world, current, action, checked=True)
        if outcome(world, current) == "running":
            legal = legal_actions(world, current)
            active = tuple(a for a in legal if a.kind in ("use", "attack", "chest", "key", "breach", "combine"))
            action = rng.choice(active if active and rng.random() < .65 else legal)
    return steps


def attach(world: World, trace: list[tuple[State, Action]], stage: int, solution: Solution,
           cfg: Config, *, expert: bool) -> Episode:
    steps = []
    prev = None
    labels_before = {}
    for index, (state, action) in enumerate(trace):
        obs = observation(world, state, prev)
        next_state, _ = transition(world, state, action, checked=True)
        after = observe(world, next_state) if stage == 2 else None
        step = Step(obs, state, action, after)
        if stage == 2:
            step.spec = transition_spec(obs.graph)
            raw = game_queries(world, state)
            answers = {(q.task, q.nodes): q.label for q in raw}
            step.changed_queries = frozenset(k for k, label in answers.items() if k in labels_before
                                             and label != labels_before[k])
            selected = balanced(raw, Random(derived_seed(world.seed, index)), cfg.data.queries_per_class)
            step.queries = tuple(dict.fromkeys((*selected, *(q for q in raw if (q.task, q.nodes) in step.changed_queries))))
            labels_before = answers
        elif index % cfg.data.label_stride == 0 or action.kind in ("key", "breach", "combine", "chest", "use", "attack"):
            result = (Solution(solution.status, solution.actions[index:]) if expert and stage != 5
                      else solve(world, state, cfg.search, ties=stage == 5))
            if stage == 3:
                step.pool = pool_targets(world, state, result, cfg.search)
            elif stage == 4:
                step.value = value_target(world, state, result, cfg.reward, cfg.optimizer.gamma,
                                          cfg.optimizer.approximate_weight)
            elif result.found:
                step.best = tuple(obs.actions.index(a) for a in result.best_actions)
                step.optimal = result.status == "optimal"
                step.value = value_target(world, state, result, cfg.reward, cfg.optimizer.gamma,
                                          cfg.optimizer.approximate_weight)
        steps.append(step)
        prev = obs.graph
    if stage in (3, 4) and trace and outcome(world, next_state) != "running":
        terminal = Step(observation(world, next_state, prev), next_state, None, None)
        terminal_result = solve(world, next_state, cfg.search)
        if stage == 3:
            terminal.pool = pool_targets(world, next_state, terminal_result, cfg.search)
        else:
            terminal.value = ValueTarget(0., 1., True)
        steps.append(terminal)
    return Episode(world, steps)


def sequences(cfg: Config, stage: int, split: str) -> list[Episode]:
    number = cfg.data.train_worlds if split == "train" else cfg.data.val_worlds
    output = []
    for index in range(number):
        seed = derived_seed(cfg.runtime.seed, "episode", split, index)
        generated = generate(seed, cfg.generation, cfg.search, split)
        world, solution = generated.world, generated.solution
        expert = walkthrough(world, solution)
        output.append(attach(world, expert, stage, solution, cfg, expert=True))
        if stage == 2:
            rng = Random(derived_seed(seed, "explore"))
            for _ in range(cfg.data.explorations):
                probe = exploratory(world, expert, rng, cfg.data.explore_turns)
                output.append(attach(world, probe, stage, solution, cfg, expert=False))
        if stage in (3, 4) and cfg.data.negative_every and index % cfg.data.negative_every == 0:
            if world.critical_doors and world.items:
                trapped = deadlock(world)
                negative = exploratory(trapped, [], Random(seed), min(4, cfg.data.explore_turns))
                output.append(attach(trapped, negative, stage, Solution("impossible"), cfg, expert=True))
    if stage == 3 and split == "train":
        known = [step for episode in output for step in episode.steps
                 if step.pool is not None and step.pool[-1] is not None]
        positives = sum(step.pool[-1] for step in known)
        negatives = len(known) - positives
        if positives and negatives:
            weights = {0.: len(known) / (2 * negatives), 1.: len(known) / (2 * positives)}
            for step in known:
                step.solvability_weight = weights[step.pool[-1]]
    return [episode for episode in output if episode.steps]
