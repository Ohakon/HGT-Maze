"""Supervision derived from explicit rules, transitions and search certificates."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from heapq import heappop, heappush
from itertools import combinations, permutations
from random import Random

from .game import (SUPPORT, STATUS, Action, Condition, Reward, State, World, active_affinities,
                   choices, door_matches, legal_actions, outcome, transition)
from .generate import derived_seed
from .graph import (CONDITIONS, Builder, Event, Graph, Query, diff, observe, tag, uid)
from .solver import Search, Solution, combat_possible, reach

POOL_NAMES = ("reachable_rooms", "openable_doors", "defeatable_enemies", "conditional_doors",
              "goal_blockers", "goal_reachable", "solvable")
AXES = {"nodes": (0, 8, 24), "edges": (0, 4, 12), "node_types": (1, 3, 5), "edge_types": (1, 2, 3),
        "hops": (1, 2, 3, 4, 5), "branches": (1, 2, 3), "ands": (0, 2, 4), "ors": (0, 2, 4),
        "many_to_many": (0, 2, 4), "crossings": (0, 2, 4), "enemy_links": (0, 2, 4)}


def gate_kind(condition: Condition) -> str:
    return condition.op if condition.op in ("and", "or") else "none"


def game_queries(world: World, state: State) -> tuple[Query, ...]:
    rows = []
    available = tuple(i for i, stock in enumerate(state.stock) if stock)
    keys = {i for i in available if world.items[i].key}
    alive = tuple(i for i, hp in enumerate(state.enemy_hp) if hp)

    def add(task: str, label: bool, *nodes: str, gate: str = "none") -> None:
        rows.append(Query(task, tuple(nodes), bool(label), 0, gate))

    def matches(door: int, bundles: tuple[tuple[int, ...], ...], current: State) -> bool:
        return any(door_matches(world, current, door, bundle) for bundle in bundles)

    for i, door in enumerate(world.doors):
        if i in state.opened:
            continue
        d = uid("door", i)
        gate = gate_kind(door.requires)
        single = {k: door_matches(world, state, i, (k,)) for k in available}
        for k in available:
            add("key_door" if k in keys else "item_door", single[k], uid("item", k), d, gate=gate)
        if world.max_combo >= 2:
            for a, b in combinations(available, 2):
                if not single[a] and not single[b]:
                    add("combo_door", door_matches(world, state, i, (a, b)),
                        uid("item", a), uid("item", b), d, gate=gate)
        owned = tuple(choices(state.inventory, world.max_combo, include_empty=True))
        before = matches(i, owned, state)
        for e in alive:
            killed = replace(state, enemy_hp=tuple(0 if j == e else hp for j, hp in enumerate(state.enemy_hp)))
            add("death_door", not before and matches(i, owned, killed), uid("enemy", e), d, gate=gate)
        for j, chest in enumerate(world.chests):
            if j not in state.looted:
                bundles = tuple(choices(tuple(sorted(chest.items)), world.max_combo))
                add("chest_access", matches(i, bundles, state), uid("chest", j), d, gate=gate)
    for e in alive:
        for k in available:
            if k in keys:
                continue
            current = active_affinities(world, state, e, k, (k,))
            bonus = {a.kind for a in current if a.kind != "resist"}
            add("weakness", bool(bonus), uid("item", k), uid("enemy", e))
            for index in range(len(STATUS)):
                timers = list(state.statuses[e])
                timers[index] = max(1, timers[index])
                hypothetical = active_affinities(world, state, e, k, (k,), tuple(timers))
                added = any(a not in current and a.kind != "resist" for a in hypothetical)
                add("status_bonus", added, uid("item", k), uid("enemy", e), uid("status_type", index))
    links = {(s.source, s.target, s.kind) for s in world.supports}
    for a, b in permutations(alive, 2):
        for kind in SUPPORT:
            add(kind, (a, b, kind) in links, uid("enemy", a), uid("enemy", b))
        direct = any((a, b, kind) in links for kind in SUPPORT)
        reverse = any((b, a, kind) in links for kind in SUPPORT)
        if direct != reverse:
            add("direction", direct, uid("enemy", a), uid("enemy", b))
    return tuple(rows)


def balanced(queries: tuple[Query, ...], rng: Random, limit: int) -> tuple[Query, ...]:
    strata = defaultdict(list)
    for q in queries:
        strata[(q.task, q.label)].append(q)
    selected = []
    for task in sorted({q.task for q in queries}):
        pos, neg = strata[(task, True)], strata[(task, False)]
        size = min(len(pos), len(neg), limit)
        selected.extend(rng.sample(pos, size) + rng.sample(neg, size))
    rng.shuffle(selected)
    return tuple(selected)


@dataclass(frozen=True)
class RelationCase:
    graph: Graph
    queries: tuple[Query, ...]
    axis: str = "game"
    level: int = 0


def controlled(seed: int, split: str, axis: str, level: int) -> RelationCase:
    """Mirrored logical proofs with identical query-node features and balanced labels."""
    if axis not in (*AXES, "curriculum"):
        raise ValueError(f"Unknown axis {axis}")
    rng = Random(derived_seed("controlled", split, seed))
    builder = Builder(seed)
    depth = level if axis in ("hops", "curriculum") else 4
    # All exact-hop cases share the same five available levels per side.
    capacity = max(5, depth)
    values = {}
    operands = {}
    ops = {}

    def add(key: str, op: str, truth: bool = False, inputs: tuple[str, ...] = ()) -> str:
        builder.node(key, "condition", tag(CONDITIONS.index(op), float(truth)))
        values[key] = truth
        ops[key] = op
        operands[key] = inputs
        for child in inputs:
            builder.link(key, "operand", child)
        return key

    neutral_shared = [add(f"shared:{i}", "fact", True) for i in range(capacity)]
    roots = []
    truth_first = bool(rng.getrandbits(1))
    gate_counts = []
    for side, truth in enumerate((truth_first, not truth_first)):
        token = f"branch:{side}"
        parent = add(f"{token}:input", "fact", truth)
        padding = add(f"{token}:padding", "fact", True) if depth < capacity or axis == "hops" else ""
        if axis == "ands":
            kinds = ["and"] * level + ["relay"] * (capacity - level)
        elif axis == "ors":
            kinds = ["or"] * level + ["relay"] * (capacity - level)
        elif axis == "curriculum":
            choices_for_level = ("relay",) if level < 2 else ("and", "or") if level >= 3 else ("and",)
            kinds = [rng.choice(choices_for_level) for _ in range(capacity)] if side == 0 else gate_counts
        else:
            kinds = ["and"] * capacity
        gate_counts = list(kinds)
        for index, op in enumerate(kinds):
            neutral = op != "or"
            local = [add(f"{token}:neutral:{index}:0", "fact", neutral)]
            pad = f"{token}:neutral:{index}:1"
            branch = level if axis == "branches" else (1 if axis == "curriculum" and level < 2 else 2)
            add(pad, "relay", inputs=(local[0],)) if op == "relay" and branch == 2 else add(pad, "fact", neutral)
            local.append(pad)
            extra = local[:branch - 1]
            if op == "relay":
                extra = []
            if axis == "crossings" and index < level:
                extra = [neutral_shared[index]]
            if axis == "branches":
                # Keep the total edge count fixed as operands move onto the queried path.
                add(f"{token}:unqueried:{index}", "and", inputs=tuple(local[branch - 1:]))
            # For the exact-hop axis the tail retains its edges and uses a disconnected seed.
            if index == depth and depth < capacity:
                parent = padding
            parent = add(f"{token}:gate:{index}", op, inputs=(parent, *extra))
            if index + 1 == depth:
                roots.append(parent)

    def prove(node: str) -> bool:
        if ops[node] == "fact":
            return values[node]
        facts = tuple(prove(child) for child in operands[node])
        return any(facts) if ops[node] == "or" else all(facts) if ops[node] == "and" else facts[0]

    extra_nodes = 1 + level if axis == "nodes" else 8
    other_types = ("passage", "room", "material", "chest", "key")
    for i in range(extra_nodes):
        types = level if axis == "node_types" else 1
        builder.node(f"extra:{i}", other_types[i % types])
    for i in range(8):
        builder.node(f"item:{i}", "item")
        builder.node(f"effect:{i}", "effect")
        builder.node(f"enemy:{i}", "enemy")
        builder.node(f"support:{i}", "support")
    stride = rng.randrange(1, 5)
    for i in range(8):
        builder.link(f"item:{i}", "produces", f"effect:{i}")
        if axis == "many_to_many":
            other = (i - 1) if i // 2 < level and i % 2 else (i + 1) % 8
        else:
            other = (i + stride) % 8
        builder.link(f"item:{i}", "produces", f"effect:{other}")
    n_links = 5 if axis == "enemy_links" else 4
    labels = ("boosts", "protects", "heals")
    for i in range(n_links):
        j = i % 8
        label = labels[i % level] if axis == "edge_types" else "boosts"
        if axis == "enemy_links" and i > level:
            builder.link(f"item:{j}", "produces", f"effect:{(j + stride + 1) % 8}")
        else:
            target = j if axis == "enemy_links" else j % 4
            builder.link(f"enemy:{j}", label, f"support:{target}")
    for i in range(4):
        builder.link(f"support:{i}", "benefits", f"enemy:{(i + 1) % 8}")
    if axis == "enemy_links":
        builder.link("support:4", "benefits", "enemy:5")
    if axis == "edges":
        for i in range(level):
            source = i % 8
            target = (source + stride + 1 + i // 8) % 8
            builder.link(f"item:{source}", "produces", f"effect:{target}")
    kind = "and" if axis in ("ands", "branches", "crossings", "hops") else "or" if axis == "ors" else "none"
    queries = tuple(Query("logic", (root,), prove(root), depth, kind) for root in roots)
    return RelationCase(builder.freeze(), queries, axis, level)


@dataclass(frozen=True)
class Observation:
    graph: Graph
    events: tuple[Event, ...]
    actions: tuple[Action, ...]


def observation(world: World, state: State, prior: Graph | None = None) -> Observation:
    graph = observe(world, state)
    return Observation(graph, diff(prior, graph), legal_actions(world, state))


def pool_targets(world: World, state: State, result: Solution, search: Search) -> tuple[float | None, ...]:
    area = reach(world, state)
    rooms = sum(any(tile in area for tile in room) for room in world.rooms)
    held = tuple(choices(state.inventory, world.max_combo, include_empty=True))
    all_items = tuple(choices(tuple(i for i, stock in enumerate(state.stock) if stock),
                              world.max_combo, include_empty=True))
    dead = replace(state, enemy_hp=(0,) * len(world.enemies))
    closed = [(i, door) for i, door in enumerate(world.doors) if i not in state.opened]
    now = sum(any(p in area for p in world.floor_neighbors[door.pos])
              and any(door_matches(world, state, i, bundle) for bundle in held) for i, door in closed)
    eventual = sum(any(door_matches(world, dead, i, bundle) for bundle in all_items) for i, _ in closed)
    combat = [combat_possible(world, state, i, search.combat_budget) if outcome(world, state) == "running" else False
              for i, hp in enumerate(state.enemy_hp) if hp]
    blocked = {door.pos for i, door in closed if not any(door_matches(world, state, i, bundle) for bundle in held)}
    agenda = [(0, state.pos)]
    distances = {state.pos: 0}
    while agenda:
        count, pos = heappop(agenda)
        if pos == world.goal:
            break
        for nxt in world.floor_neighbors[pos]:
            score = count + int(nxt in blocked)
            if score < distances.get(nxt, len(world.floor) + 1):
                distances[nxt] = score
                heappush(agenda, (score, nxt))
    possible = 1. if result.found else 0. if result.status == "impossible" else None
    return (float(rooms), float(now), None if None in combat else float(sum(combat)), float(eventual),
            float(distances[world.goal]), float(world.goal in area), possible)


@dataclass(frozen=True)
class ValueTarget:
    value: float
    weight: float
    exact: bool


def discount_sum(gamma: float, length: int) -> float:
    return float(length) if gamma == 1. else (1. - gamma ** length) / (1. - gamma)


def value_target(world: World, state: State, result: Solution, reward: Reward, gamma: float,
                 approximate_weight: float) -> ValueTarget | None:
    if outcome(world, state) != "running":
        return ValueTarget(0., 1., True)
    if result.status == "unknown":
        return None
    exact = (result.status == "optimal" and reward.hp == 0 and reward.item == 0
             and reward.death >= reward.turn * discount_sum(gamma, world.max_turns - state.turn))
    if exact:
        turns = len(result.actions)
        return ValueTarget(gamma ** (turns - 1) * reward.goal - reward.turn * discount_sum(gamma, turns), 1., True)
    total, t = 0., 0
    current = state
    while outcome(world, current) == "running":
        action = result.actions[t] if t < len(result.actions) else Action("wait")
        current, score = transition(world, current, action, reward)
        total += gamma ** t * score
        t += 1
    return ValueTarget(total, approximate_weight, False)
