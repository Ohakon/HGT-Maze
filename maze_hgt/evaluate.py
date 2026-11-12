"""Held-out policy comparisons, controlled relation tests and paired interventions."""
from __future__ import annotations

from dataclasses import asdict, replace
from math import sqrt
from pathlib import Path
from random import Random
from statistics import fmean
from time import perf_counter

import torch

from .config import Config
from .game import (DOOR_ACTIONS, Action, Affinity, Condition, Effect, State, Support, World, all_of, any_of,
                   door_matches, initial_state, legal_actions, outcome, transition)
from .generate import Generated, derived_seed, generate, rename
from .graph import FORWARD, Query, observe, uid
from .io import write_json
from .metrics import Metrics, weight_count
from .model import Network
from .rl import collect, curriculum
from .tasks import AXES, RelationCase, balanced, controlled, game_queries, observation
from .train import aggregate, relation_loss


def held_out(cfg: Config, number: int, *, ood: bool = False) -> list[Generated]:
    settings = (replace(curriculum(cfg.generation, 2),
                        recipes=("compound", *tuple(r for r in cfg.generation.recipes if r != "compound")),
                        required_recipe="compound", max_combo=max(2, cfg.generation.max_combo))
                if ood else cfg.generation)
    search = replace(cfg.search, mode="guided", budget=max(cfg.search.budget, 6000)) if (
        ood and cfg.search.mode == "auto") else cfg.search
    return [generate(derived_seed(cfg.runtime.seed, "ood" if ood else "id", i), settings, search, "test")
            for i in range(number)]


def play_random(world: World, cfg: Config) -> tuple[State, float, float]:
    rng = Random(derived_seed(world.seed, "random-policy"))
    state = initial_state(world)
    total, discounted = 0., 0.
    while outcome(world, state) == "running":
        action = rng.choice(legal_actions(world, state))
        state, reward = transition(world, state, action, cfg.reward, checked=True)
        total += reward
        discounted += cfg.optimizer.gamma ** (state.turn - 1) * reward
    return state, total, discounted


def _result(generated: Generated, status: str, turns: int, total: float, discounted: float) -> dict:
    world, answer = generated.world, generated.solution
    return {"seed": world.seed, "recipe": world.recipe, "attempts": generated.attempt + 1,
            "tiles": len(world.floor), "depth": len(world.critical_doors), "enemies": len(world.enemies),
            "outcome": status, "turns": turns, "return": total, "discounted_return": discounted,
            "reference_status": answer.status, "reference_turns": len(answer.actions),
            "reference_expanded": answer.expanded,
            "excess_turns": turns - len(answer.actions) if status == "success" and answer.status == "optimal" else None}


def _summary(rows: list[dict], seconds: float) -> dict:
    completed = [row for row in rows if row["outcome"] == "success"]
    regrets = [row["excess_turns"] for row in completed if row["excess_turns"] is not None]
    n, rate = len(rows), len(completed) / len(rows)
    z = 1.95996398454
    denom = 1 + z * z / n
    center = (rate + z * z / (2 * n)) / denom
    radius = z * sqrt(rate * (1 - rate) / n + z * z / (4 * n * n)) / denom
    return {"episodes": n, "clears": len(completed), "clear_rate": rate,
            "clear_rate_ci95": [max(0., center - radius), min(1., center + radius)],
            "death_rate": sum(row["outcome"] == "death" for row in rows) / len(rows),
            "mean_turns": fmean(row["turns"] for row in rows), "mean_return": fmean(row["return"] for row in rows),
            "discounted_return": fmean(row["discounted_return"] for row in rows),
            "mean_success_turns": fmean(row["turns"] for row in completed) if completed else None,
            "optimal_regret_turns": fmean(regrets) if regrets else None, "optimal_regret_n": len(regrets),
            "wall_seconds": seconds, "wall_ms_per_turn": 1000 * seconds / sum(row["turns"] for row in rows),
            "runs": rows}


def game_benchmark(cfg: Config, generated: list[Generated], models: dict[str, Network]) -> dict:
    out = {}
    worlds = [row.world for row in generated]
    for name in ("reference", "random", *models):
        first = perf_counter()
        rows = []
        if name in models:
            model = models[name]
            rollouts = collect(model, worlds, cfg, list(range(len(worlds))), greedy=True)
            for obj, episode in zip(generated, rollouts):
                rewards = [s.reward for s in episode.steps]
                rows.append(_result(obj, episode.result, len(episode.steps), sum(rewards),
                                    sum(cfg.optimizer.gamma ** i * r for i, r in enumerate(rewards))))
            model_info = {"parameters": weight_count(model),
                          "policy_parameters": sum(weight_count(module) for key, module in model.named_children()
                                                   if key in ("hgt", "gru", "events", "memory_norm", "pool", "value",
                                                              "actions", "action_context", "actor")),
                          "architecture": asdict(model.cfg),
                          "mean_nodes": fmean(len(s.obs.graph.nodes) for ep in rollouts for s in ep.steps),
                          "mean_edges": fmean(len(s.obs.graph.edges) for ep in rollouts for s in ep.steps)}
        else:
            model_info = {}
            for obj in generated:
                world = obj.world
                if name == "reference":
                    state, rewards = initial_state(world), []
                    for action in obj.solution.actions:
                        state, score = transition(world, state, action, cfg.reward, checked=True)
                        rewards.append(score)
                    total = sum(rewards)
                    discounted = sum(cfg.optimizer.gamma ** i * r for i, r in enumerate(rewards))
                else:
                    state, total, discounted = play_random(world, cfg)
                rows.append(_result(obj, outcome(world, state), state.turn, total, discounted))
        out[name] = _summary(rows, perf_counter() - first) | model_info
    return out


@torch.no_grad()
def score_cases(model: Network, cases: list[RelationCase]) -> dict:
    model.eval()
    book = Metrics()
    for offset in range(0, len(cases), 8):
        subset = cases[offset:offset + 8]
        frames = model.encode([case.graph for case in subset], hgt_only=True)
        for case, frame in zip(subset, frames):
            relation_loss(model, frame, case.queries, book)
    return aggregate(book, 1)


def relation_benchmark(cfg: Config, models: dict[str, Network], id_worlds: list[Generated],
                       ood_worlds: list[Generated], repeats: int) -> dict:
    def game_cases(examples: list[Generated]) -> list[RelationCase]:
        cases = []
        for obj in examples:
            state = initial_state(obj.world)
            queries = balanced(game_queries(obj.world, state), Random(obj.world.seed), cfg.data.queries_per_class)
            if queries:
                cases.append(RelationCase(observe(obj.world, state), queries))
        return cases

    id_cases, ood_cases = game_cases(id_worlds), game_cases(ood_worlds)
    axes = {(axis, level): [controlled(derived_seed(cfg.runtime.seed, "controlled-test", i), "test", axis, level)
                            for i in range(repeats)] for axis, levels in AXES.items() for level in levels}
    results = {}
    for key, model in models.items():
        axis_stats = {}
        for (axis, level), cases in axes.items():
            stats = score_cases(model, cases)
            stats.update({"level": level, "nodes": fmean(len(c.graph.nodes) for c in cases),
                          "edges": fmean(sum(e.rel in FORWARD for e in c.graph.edges) for c in cases),
                          "node_types": fmean(len({n.kind for n in c.graph.nodes}) for c in cases),
                          "edge_types": fmean(len({e.rel for e in c.graph.edges if e.rel in FORWARD}) for c in cases)})
            rows = axis_stats.setdefault(axis, [])
            stats["accuracy_drop"] = rows[0]["relations.accuracy"] - stats["relations.accuracy"] if rows else 0.
            rows.append(stats)
        results[key] = {"id": score_cases(model, id_cases), "ood": score_cases(model, ood_cases), "controlled": axis_stats}
    return results


def _fixture(cfg: Config, seed: int) -> World:
    base = replace(cfg.generation, min_depth=1, max_depth=1, min_items=4, max_items=max(4, cfg.generation.max_items),
                   min_enemies=2, max_enemies=max(2, cfg.generation.max_enemies), recipes=("key",), required_recipe="")
    source = generate(seed, base, cfg.search, "test").world
    items = tuple(replace(item, key=i < 2, charges=-1) for i, item in enumerate(source.items[:4]))
    enemies = tuple(replace(e, hp=6, attack=1, loot=()) for e in source.enemies[:2])
    door = replace(source.doors[0], requires=Condition("item", 0))
    return replace(source, items=items, enemies=enemies, doors=(door,), starting_items=(0, 1, 2, 3),
                   chests=(), ground=(), effects=(Effect(2, "damage", -1, 2), Effect(3, "damage", -1, 2)),
                   affinities=(), supports=(), max_combo=1)


def _near(world: World, target: tuple[int, int]) -> tuple[int, int]:
    blocked = {*world.door_at, *world.enemy_at, world.goal}
    return next((p for p in world.floor_neighbors[target] if p not in blocked), world.start)


def interventions(cfg: Config, seed: int) -> list[tuple[str, World, World]]:
    base = _fixture(cfg, seed)
    key = replace(base, start=_near(base, base.doors[0].pos))
    door = key.doors[0]
    fight = replace(base, start=_near(base, base.enemies[1].pos))
    and_door = replace(key, max_combo=2, doors=(replace(door, requires=all_of(Condition("item", 0), Condition("item", 1))),))
    or_door = replace(and_door, doors=(replace(door, requires=any_of(Condition("item", 0), Condition("item", 1))),))
    center, sides = next((p, tuple(q for q in qs if q not in (base.goal, door.pos)))
                         for p, qs in base.floor_neighbors.items()
                         if p not in (base.goal, door.pos) and len(set(qs) - {base.goal, door.pos}) >= 2)
    death = replace(base, start=center, enemies=tuple(replace(e, hp=2, pos=p) for e, p in zip(base.enemies, sides)),
                    doors=(replace(door, requires=Condition("dead", 0)),))
    return [
        ("key", key, replace(key, doors=(replace(door, requires=Condition("item", 1)),))),
        ("weakness", replace(fight, affinities=(Affinity(1, 2, "critical", 3),)),
         replace(fight, affinities=(Affinity(1, 3, "critical", 3),))),
        ("protect", replace(fight, supports=(Support(0, 1, "protect", 1),)),
         replace(fight, supports=(Support(1, 0, "protect", 1),))),
        ("and_or", and_door, or_door),
        ("death", death, replace(death, doors=(replace(door, requires=Condition("dead", 1)),))),
    ]


def _query_map(world: World, state: State) -> dict[tuple, Query]:
    queries = list(game_queries(world, state))
    if world.max_combo >= 2 and (0, 1) == state.inventory[:2]:
        together = (door_matches(world, state, 0, (0, 1))
                    and not any(door_matches(world, state, 0, (i,)) for i in (0, 1)))
        queries.append(Query("combo_door", (uid("item", 0), uid("item", 1), uid("door", 0)), together, 0))
    return {(q.task, q.nodes): q for q in queries}


def local_oracle(kind: str, world: World, state: State, actions: tuple[Action, ...]) -> set[Action]:
    best = {}
    for action in actions:
        if kind in ("key", "and_or") and action.kind not in DOOR_ACTIONS:
            continue
        if kind in ("weakness", "protect", "death") and action.kind not in ("use", "attack"):
            continue
        after, _ = transition(world, state, action, checked=True)
        if after.hp <= 0:
            continue
        if kind in ("weakness", "protect"):
            damage = state.enemy_hp[1] - after.enemy_hp[1]
            if damage:
                best[action] = damage
        elif 0 in after.opened:
            best[action] = -len(action.items) if kind in ("key", "and_or") else 0
    peak = max(best.values(), default=None)
    return {a for a, score in best.items() if score == peak}


@torch.no_grad()
def counterfactual_benchmark(cfg: Config, model: Network, pairs: list[tuple[str, World, World]]) -> dict:
    model.eval()
    book = Metrics()
    examples = []
    for kind, before, after in pairs:
        left, right = initial_state(before), initial_state(after)
        obs = [observation(before, left), observation(after, right)]
        frames = model.encode([o.graph for o in obs], [o.events for o in obs])
        query_maps = [_query_map(before, left), _query_map(after, right)]
        common = sorted(query_maps[0].keys() & query_maps[1].keys())
        query_lists = [[q[key] for key in common] for q in query_maps]
        probs = [model.relation_scores(frame, qs).sigmoid().tolist() for frame, qs in zip(frames, query_lists)]
        changed = 0
        for a, b, p, q in zip(*query_lists, *probs):
            category = "changed" if a.label != b.label else "unchanged"
            flip = (p >= .5) != (q >= .5)
            correct = (p >= .5) == a.label and (q >= .5) == b.label
            for prefix in (f"query.{category}", f"{kind}.query.{category}"):
                book.mean(f"{prefix}.n", 1., 1)
                book.mean(f"{prefix}.both_correct", float(correct))
                book.mean(f"{prefix}.flip_rate", float(flip))
                book.mean(f"{prefix}.mean_absolute_shift", abs(q - p))
                if category == "changed":
                    book.mean(f"{prefix}.direction_correct", float((q - p) * (b.label - a.label) > 0))
            changed += a.label != b.label
        actions = [o.actions for o in obs]
        logits, _ = model.policy(frames, actions)
        selected = [a[int(index)] for a, index in zip(actions, logits.argmax(1))]
        allowed = [local_oracle(kind, w, s, a) for w, s, a in zip((before, after), (left, right), actions)]
        oracle_changed = allowed[0] != allowed[1]
        selected_changed = selected[0] != selected[1]
        policy_ok = all(a in options for a, options in zip(selected, allowed)) if all(allowed) else None
        oracle_mass = None
        if policy_ok is not None:
            category = "changed" if oracle_changed else "unchanged"
            distributions = logits.softmax(dim=1)
            oracle_mass = [sum(float(p) for a, p in zip(row, probs) if a in targets)
                           for row, probs, targets in zip(actions, distributions, allowed)]
            for prefix in (f"action.{category}", f"{kind}.action.{category}"):
                book.mean(f"{prefix}.both_correct", float(policy_ok))
                book.mean(f"{prefix}.switch_rate", float(selected_changed))
                book.mean(f"{prefix}.oracle_probability", fmean(oracle_mass))
        examples.append({"kind": kind, "seed": before.seed, "queries": len(common), "changed_labels": changed,
                         "unchanged_labels": len(common) - changed, "oracle_action_set_changed": oracle_changed,
                         "model_action_changed": selected_changed, "both_actions_correct": policy_ok,
                         "oracle_probability": oracle_mass,
                         "nodes_before": len(obs[0].graph.nodes), "nodes_after": len(obs[1].graph.nodes),
                         "edges_before": len(obs[0].graph.edges), "edges_after": len(obs[1].graph.edges),
                         "actions": [asdict(a) for a in selected]})
    metrics = book.flat()
    for name, (_, count) in book.means.items():
        if name.endswith(".n"):
            metrics[name] = count
        elif name.endswith(".both_correct") and ".action." in "." + name:
            metrics[name.rsplit(".", 1)[0] + ".n"] = count
    return {"metrics": metrics, "examples": examples}


@torch.no_grad()
def rename_benchmark(model: Network, worlds: list[Generated]) -> dict:
    maxima = {"policy": 0., "value": 0., "relations": 0.}
    checks = 0
    matching_actions = 0
    for obj in worlds:
        worlds_pair = [obj.world, rename(obj.world, derived_seed(obj.world.seed, "rename"), "test")]
        state = initial_state(obj.world)
        memory, prior = [None, None], [None, None]
        for action in obj.solution.actions:
            obs = [observation(w, state, g) for w, g in zip(worlds_pair, prior)]
            if obs[0] != obs[1]:
                raise AssertionError("Renaming changed the model inputs")
            frames = model.encode([o.graph for o in obs], [o.events for o in obs], memory)
            logits, _ = model.policy(frames, [o.actions for o in obs])
            values = model.values(frames)
            queries = game_queries(obj.world, state)
            if queries:
                relation = [model.relation_scores(frame, queries) for frame in frames]
                maxima["relations"] = max(maxima["relations"], (relation[0] - relation[1]).abs().max().item())
            maxima["policy"] = max(maxima["policy"], (logits[0] - logits[1]).abs().max().item())
            maxima["value"] = max(maxima["value"], (values[0] - values[1]).abs().item())
            matching_actions += int(logits[0].argmax() == logits[1].argmax())
            memory, prior = [f.memory for f in frames], [o.graph for o in obs]
            state, _ = transition(obj.world, state, action, checked=True)
            checks += 1
    return {"max_absolute_difference": maxima, "states": checks, "matching_argmax": matching_actions}


def evaluate(cfg: Config, models: dict[str, Network], output: Path, episodes: int,
             *, repeats: int | None = None, with_ood: bool = True) -> dict:
    if episodes < 1 or repeats is not None and repeats < 1:
        raise ValueError("Evaluation episode and repeat counts must be positive")
    id_worlds = held_out(cfg, episodes)
    ood_worlds = held_out(cfg, episodes, ood=True) if with_ood else []
    pairs = [pair for i in range(episodes)
             for pair in interventions(cfg, derived_seed(cfg.runtime.seed, "counterfactual-test", i))]
    report = {"seed": cfg.runtime.seed, "id": game_benchmark(cfg, id_worlds, models),
              "relations": relation_benchmark(cfg, models, id_worlds, ood_worlds, repeats or cfg.data.axis_repeats),
              "counterfactual": {label: counterfactual_benchmark(cfg, model, pairs) for label, model in models.items()},
              "rename": {label: rename_benchmark(model, id_worlds) for label, model in models.items()}}
    if ood_worlds:
        report["ood"] = game_benchmark(cfg, ood_worlds, models)
    write_json(output, report)
    return report
