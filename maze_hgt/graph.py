"""Typed, fully observable game graphs and diffs between formal observations."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property

from .game import (AFFINITY, DIRECTIONS, MATERIALS, STATUS, Action, Condition,
                   Pos, State, World)

NODE_TYPES = ("player", "enemy", "item", "key", "chest", "door", "room", "passage", "goal",
              "condition", "material", "status_type", "status", "effect", "affinity", "support")
FORWARD = ("north", "east", "south", "west", "at", "contains_tile", "holds", "contains", "drops",
           "on_ground", "made_of", "requires", "operand", "needs_item", "needs_death",
           "needs_status", "needs_material", "produces", "affects", "applies_status", "matches_material",
           "affinity_item", "affinity_enemy", "boosts", "protects", "heals", "benefits",
           "has_status", "status_kind")
RELATIONS = FORWARD + tuple("rev_" + key for key in FORWARD)
EVENT_TYPES = ("node_add", "node_remove", "node_change", "edge_add", "edge_remove", "edge_change")
NODE_ID = {key: i for i, key in enumerate(NODE_TYPES)}
REL_ID = {key: i for i, key in enumerate(RELATIONS)}
EVENT_ID = {key: i for i, key in enumerate(EVENT_TYPES)}
CONDITIONS = ("true", "and", "or", "item", "dead", "status", "breach", "fact", "relay")
EFFECTS = ("damage", *STATUS, "heal", "breach")
FEATURES = 12
ZEROS = (0.,) * FEATURES


def attrs(*numbers: float) -> tuple[float, ...]:
    return tuple(numbers) + (0.,) * (FEATURES - len(numbers))


def tag(index: int, a: float = 0., b: float = 0., c: float = 0.) -> tuple[float, ...]:
    values = list(attrs(a, b, c))
    values[3 + index] = 1.
    return tuple(values)


def uid(kind: str, number: int) -> str:
    return f"{kind}:{number}"


def cell(pos: Pos) -> str:
    return f"p:{pos[0]}:{pos[1]}"


def status_slot(enemy: int, kind: int) -> str:
    return f"status:{enemy}:{kind}"


PLAYER = "player"
GOAL = "goal"


@dataclass(frozen=True)
class Node:
    id: str
    kind: str
    x: tuple[float, ...] = ZEROS


@dataclass(frozen=True, order=True)
class Edge:
    src: str
    rel: str
    dst: str
    value: float = 1.

    @property
    def key(self) -> tuple[str, str, str]:
        return self.src, self.rel, self.dst


@dataclass(frozen=True)
class Graph:
    scope: int
    nodes: tuple[Node, ...]
    edges: tuple[Edge, ...]

    @cached_property
    def index(self) -> dict[str, int]:
        return {n.id: i for i, n in enumerate(self.nodes)}

    @cached_property
    def table(self) -> dict[str, Node]:
        return {n.id: n for n in self.nodes}

    @cached_property
    def relations(self) -> dict[tuple[str, str, str], Edge]:
        return {edge.key: edge for edge in self.edges}

    @cached_property
    def related(self) -> dict[str, tuple[str, ...]]:
        neighbors = defaultdict(set)
        for e in self.edges:
            neighbors[e.src].add(e.dst)
            neighbors[e.dst].add(e.src)
        return {key: tuple(sorted(value)) for key, value in neighbors.items()}


class Builder:
    def __init__(self, scope: int):
        self.scope = scope
        self.nodes: dict[str, Node] = {}
        self.edges: dict[tuple[str, str, str], Edge] = {}

    def node(self, key: str, kind: str, x: tuple[float, ...] = ZEROS) -> str:
        self.nodes[key] = Node(key, kind, x)
        return key

    def link(self, src: str, rel: str, dst: str, value: float = 1., *, reverse: bool = True) -> None:
        if src not in self.nodes or dst not in self.nodes:
            return
        edge = Edge(src, rel, dst, value)
        self.edges[edge.key] = edge
        if reverse:
            back = Edge(dst, "rev_" + rel, src, value)
            self.edges[back.key] = back

    def freeze(self) -> Graph:
        return Graph(self.scope, tuple(self.nodes[key] for key in sorted(self.nodes)),
                     tuple(self.edges[key] for key in sorted(self.edges)))


def add_condition(builder: Builder, rule: Condition, key: str, state: State) -> str:
    truth = float(rule.op == "dead" and state.enemy_hp[rule.ref] == 0)
    builder.node(key, "condition", tag(CONDITIONS.index(rule.op), truth))
    if rule.op in ("and", "or"):
        for i, child in enumerate(rule.children):
            builder.link(key, "operand", add_condition(builder, child, f"{key}.{i}", state))
    elif rule.op in ("item", "dead", "status", "breach"):
        relation, kind = {"item": ("needs_item", "item"), "dead": ("needs_death", "enemy"),
                          "status": ("needs_status", "status_type"), "breach": ("needs_material", "material")}[rule.op]
        builder.link(key, relation, uid(kind, rule.ref))
    return key


def observe(world: World, state: State) -> Graph:
    g = Builder(world.seed)
    g.node(PLAYER, "player", attrs(state.hp / 32., world.max_hp / 32., state.turn / world.max_turns,
                                   (world.max_turns - state.turn) / 100., world.max_combo / 4.))
    g.node(GOAL, "goal")
    for pos in sorted(world.floor):
        g.node(cell(pos), "passage")
    g.link(PLAYER, "at", cell(state.pos))
    g.link(GOAL, "at", cell(world.goal))
    for pos in sorted(world.floor):
        for (dr, dc), label in zip(DIRECTIONS, FORWARD[:4]):
            g.link(cell(pos), label, cell((pos[0] + dr, pos[1] + dc)), reverse=False)
    for i, tiles in enumerate(world.rooms):
        room = g.node(uid("room", i), "room")
        for pos in tiles:
            g.link(room, "contains_tile", cell(pos))
    for i, item in enumerate(world.items):
        remaining = state.stock[i]
        if remaining:
            g.node(uid("item", i), "key" if item.key else "item", attrs(max(0, remaining) / 5., float(remaining < 0)))
    for i in state.inventory:
        g.link(PLAYER, "holds", uid("item", i), max(0, state.stock[i]) / 5.)
    for pos, i in state.ground:
        g.link(cell(pos), "on_ground", uid("item", i))
    for i in range(len(MATERIALS)):
        g.node(uid("material", i), "material")
    for i in range(len(STATUS)):
        g.node(uid("status_type", i), "status_type", tag(i))
    for i, enemy in enumerate(world.enemies):
        if state.enemy_hp[i]:
            enemy_node = g.node(uid("enemy", i), "enemy",
                                attrs(state.enemy_hp[i] / 16., enemy.hp / 16., enemy.attack / 8.))
            g.link(enemy_node, "at", cell(enemy.pos))
            for item in enemy.loot:
                g.link(enemy_node, "drops", uid("item", item))
            for k, duration in enumerate(state.statuses[i]):
                if duration:
                    node = g.node(status_slot(i, k), "status", attrs(duration / 8.))
                    g.link(enemy_node, "has_status", node, duration / 8.)
                    g.link(node, "status_kind", uid("status_type", k))
    for i, chest in enumerate(world.chests):
        if i not in state.looted:
            node = g.node(uid("chest", i), "chest")
            g.link(node, "at", cell(chest.pos))
            for item in chest.items:
                g.link(node, "contains", uid("item", item))
    for i, door in enumerate(world.doors):
        if i not in state.opened:
            node = g.node(uid("door", i), "door")
            g.link(node, "at", cell(door.pos))
            g.link(node, "made_of", uid("material", door.material))
            g.link(node, "requires", add_condition(g, door.requires, f"c:door:{i}", state))
    for i, effect in enumerate(world.effects):
        item = uid("item", effect.item)
        if item not in g.nodes:
            continue
        if effect.kind not in ("breach", "heal") and effect.target >= 0 and not state.enemy_hp[effect.target]:
            continue
        node = g.node(uid("effect", i), "effect", tag(EFFECTS.index(effect.kind), effect.power / 8.,
                                                     float(effect.target == -1 and effect.kind != "heal")))
        g.link(item, "produces", node)
        if effect.kind == "breach":
            g.link(node, "matches_material", uid("material", effect.target))
        elif effect.kind == "heal":
            g.link(node, "affects", PLAYER)
        else:
            g.link(node, "affects", uid("enemy", effect.target))
            if effect.kind in STATUS:
                g.link(node, "applies_status", uid("status_type", STATUS.index(effect.kind)))
        g.link(node, "requires", add_condition(g, effect.requires, f"c:effect:{i}", state))
    for i, affinity in enumerate(world.affinities):
        enemy, item = uid("enemy", affinity.enemy), uid("item", affinity.item)
        if enemy in g.nodes and item in g.nodes:
            node = g.node(uid("affinity", i), "affinity",
                          tag(AFFINITY.index(affinity.kind), affinity.numerator / 4., affinity.denominator / 4.))
            g.link(node, "affinity_item", item)
            g.link(node, "affinity_enemy", enemy)
            g.link(node, "requires", add_condition(g, affinity.requires, f"c:affinity:{i}", state))
    for i, support in enumerate(world.supports):
        source, target = uid("enemy", support.source), uid("enemy", support.target)
        if source in g.nodes and target in g.nodes:
            node = g.node(uid("support", i), "support", attrs(support.power / 4.))
            g.link(source, {"boost": "boosts", "protect": "protects", "heal": "heals"}[support.kind], node)
            g.link(node, "benefits", target)
    return g.freeze()


@dataclass(frozen=True)
class Event:
    kind: str
    subject: str
    subject_type: str
    before: tuple[float, ...] = ZEROS
    after: tuple[float, ...] = ZEROS
    relation: str | None = None
    other: str | None = None
    other_type: str | None = None
    related: tuple[str, ...] = ()

    def receivers(self) -> tuple[tuple[str, int], ...]:
        primary = ((self.subject, 0), (self.other, 1)) if self.other is not None else ((self.subject, 0),)
        return primary + tuple((i, 2) for i in self.related)


def diff(before: Graph | None, after: Graph) -> tuple[Event, ...]:
    events = []
    previous = before.table if before is not None else {}
    for key in sorted(previous.keys() | after.table.keys()):
        old, new = previous.get(key), after.table.get(key)
        if old == new:
            continue
        related = set(before.related.get(key, ())) if before is not None else set()
        related.update(after.related.get(key, ()))
        events.append(Event("node_add" if old is None else "node_remove" if new is None else "node_change",
                            key, (new or old).kind, old.x if old else ZEROS, new.x if new else ZEROS,
                            related=tuple(sorted(related))))
    old_edges = before.relations if before is not None else {}
    for key in sorted(old_edges.keys() | after.relations.keys()):
        old, new = old_edges.get(key), after.relations.get(key)
        if old == new:
            continue
        src, rel, dst = key
        a = after.table.get(src) or previous[src]
        b = after.table.get(dst) or previous[dst]
        events.append(Event("edge_add" if old is None else "edge_remove" if new is None else "edge_change",
                            src, a.kind, attrs(old.value) if old else ZEROS, attrs(new.value) if new else ZEROS,
                            rel, dst, b.kind))
    return tuple(events)


def action_nodes(action: Action) -> tuple[str, tuple[str, ...]]:
    if action.kind == "move":
        target = cell(action.target)
    elif action.kind == "chest":
        target = uid("chest", action.target)
    elif action.kind in ("key", "breach", "combine"):
        target = uid("door", action.target)
    elif action.kind in ("attack", "use"):
        target = uid("enemy", action.target)
    else:
        target = PLAYER
    return target, tuple(uid("item", i) for i in action.items)


@dataclass(frozen=True)
class TransitionSpec:
    nodes: tuple[str, ...]
    birth_slots: tuple[tuple[str, str, str], ...]
    edges: tuple[tuple[str, str, str], ...]


def transition_spec(graph: Graph) -> TransitionSpec:
    """Enumerate possible changes from the current graph and schema alone."""
    edges = {e.key for e in graph.edges if e.rel in FORWARD[4:]}
    slots = []
    enemy_locations = {}
    drops = []
    for e in graph.edges:
        if e.rel == "at" and graph.table[e.src].kind == "enemy":
            enemy_locations[e.src] = e.dst
        elif e.rel == "drops":
            drops.append((e.src, e.dst))
    for node in graph.nodes:
        if node.kind == "passage":
            edges.add((PLAYER, "at", node.id))
        elif node.kind in ("item", "key"):
            edges.add((PLAYER, "holds", node.id))
        elif node.kind == "enemy":
            enemy_id = int(node.id.split(":")[1])
            for i in range(len(STATUS)):
                slot, kind = status_slot(enemy_id, i), uid("status_type", i)
                edges.add((node.id, "has_status", slot))
                edges.add((slot, "status_kind", kind))
                if slot not in graph.table:
                    slots.append((slot, node.id, kind))
    edges.update((enemy_locations[enemy], "on_ground", item) for enemy, item in drops)
    return TransitionSpec(tuple(graph.index), tuple(slots), tuple(sorted(edges)))


RELATION_TASKS = ("item_door", "key_door", "weakness", "combo_door", "status_bonus", "boost", "protect",
                  "heal", "death_door", "chest_access", "direction", "logic")


@dataclass(frozen=True)
class Query:
    task: str
    nodes: tuple[str, ...]
    label: bool
    hops: int
    gate: str = "none"
