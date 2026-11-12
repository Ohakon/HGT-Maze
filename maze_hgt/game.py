"""The immutable game state and the sole transition function."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from functools import cached_property
from itertools import combinations
from typing import Iterator

Pos = tuple[int, int]
STATUS = ("poison", "weaken", "vulnerable")
DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))
SUPPORT = ("boost", "protect", "heal")
AFFINITY = ("weak", "critical", "resist")
MATERIALS = ("赤", "青", "緑", "黄", "紫", "白")


def neighbors(pos: Pos) -> Iterator[Pos]:
    for row, col in DIRECTIONS:
        yield pos[0] + row, pos[1] + col


def distance(a: Pos, b: Pos) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


@dataclass(frozen=True)
class Condition:
    op: str
    ref: int = -1
    children: tuple[Condition, ...] = ()


ALWAYS = Condition("true")


def all_of(*parts: Condition) -> Condition:
    return Condition("and", children=parts)


def any_of(*parts: Condition) -> Condition:
    return Condition("or", children=parts)


@dataclass(frozen=True)
class Item:
    name: str
    key: bool = False
    charges: int = -1


@dataclass(frozen=True)
class Effect:
    item: int
    kind: str
    target: int
    power: int
    requires: Condition = ALWAYS


@dataclass(frozen=True)
class Affinity:
    enemy: int
    item: int
    kind: str
    numerator: int
    denominator: int = 1
    requires: Condition = ALWAYS


@dataclass(frozen=True)
class Support:
    source: int
    target: int
    kind: str
    power: int


@dataclass(frozen=True)
class Door:
    pos: Pos
    material: int
    requires: Condition


@dataclass(frozen=True)
class Enemy:
    pos: Pos
    hp: int
    attack: int
    loot: tuple[int, ...] = ()


@dataclass(frozen=True)
class Chest:
    pos: Pos
    items: tuple[int, ...]


@dataclass(frozen=True)
class World:
    grid: tuple[str, ...]
    start: Pos
    goal: Pos
    rooms: tuple[tuple[Pos, ...], ...] = ()
    items: tuple[Item, ...] = ()
    effects: tuple[Effect, ...] = ()
    affinities: tuple[Affinity, ...] = ()
    supports: tuple[Support, ...] = ()
    doors: tuple[Door, ...] = ()
    enemies: tuple[Enemy, ...] = ()
    chests: tuple[Chest, ...] = ()
    starting_items: tuple[int, ...] = ()
    ground: tuple[tuple[Pos, int], ...] = ()
    max_hp: int = 24
    max_turns: int = 80
    max_combo: int = 2
    seed: int = 0
    critical_doors: tuple[int, ...] = ()
    recipe: str = ""

    @cached_property
    def floor(self) -> frozenset[Pos]:
        return frozenset((r, c) for r, row in enumerate(self.grid) for c, char in enumerate(row) if char != "#")

    @cached_property
    def floor_neighbors(self) -> dict[Pos, tuple[Pos, ...]]:
        return {p: tuple(q for q in neighbors(p) if q in self.floor) for p in sorted(self.floor)}

    @cached_property
    def door_at(self) -> dict[Pos, int]:
        return {door.pos: i for i, door in enumerate(self.doors)}

    @cached_property
    def enemy_at(self) -> dict[Pos, int]:
        return {enemy.pos: i for i, enemy in enumerate(self.enemies)}

    @cached_property
    def chest_at(self) -> dict[Pos, int]:
        return {chest.pos: i for i, chest in enumerate(self.chests)}

    @cached_property
    def effects_by_item(self) -> tuple[tuple[Effect, ...], ...]:
        return tuple(tuple(e for e in self.effects if e.item == i) for i in range(len(self.items)))

    @cached_property
    def affinity_by_enemy(self) -> tuple[tuple[Affinity, ...], ...]:
        return tuple(tuple(a for a in self.affinities if a.enemy == i) for i in range(len(self.enemies)))


@dataclass(frozen=True)
class State:
    pos: Pos
    hp: int
    inventory: tuple[int, ...]
    stock: tuple[int, ...]
    enemy_hp: tuple[int, ...]
    statuses: tuple[tuple[int, ...], ...]
    opened: frozenset[int] = frozenset()
    looted: frozenset[int] = frozenset()
    ground: tuple[tuple[Pos, int], ...] = ()
    turn: int = 0

    def search_key(self) -> tuple:
        return (self.pos, self.hp, self.inventory, self.stock, self.enemy_hp,
                self.statuses, self.opened, self.looted, self.ground)


@dataclass(frozen=True)
class Action:
    kind: str
    target: Pos | int | None = None
    items: tuple[int, ...] = ()


WAIT = Action("wait")
ACTION_KINDS = ("move", "chest", "key", "breach", "combine", "attack", "use", "heal", "wait")
DOOR_ACTIONS = ("key", "breach", "combine")


@dataclass(frozen=True)
class Reward:
    goal: float = 1.0
    death: float = 1.0
    turn: float = 0.01
    hp: float = 0.0
    item: float = 0.0

    def __post_init__(self) -> None:
        if self.goal <= 0 or min(self.death, self.turn, self.hp, self.item) < 0:
            raise ValueError("goal must be positive and all costs nonnegative")


def outcome(world: World, state: State) -> str:
    if state.hp <= 0:
        return "death"
    if state.pos == world.goal:
        return "success"
    return "timeout" if state.turn >= world.max_turns else "running"


def choices(items: tuple[int, ...], maximum: int, include_empty: bool = False) -> Iterator[tuple[int, ...]]:
    for size in range(0 if include_empty else 1, min(maximum, len(items)) + 1):
        yield from combinations(items, size)


def condition_met(rule: Condition, selected: tuple[int, ...], hp: tuple[int, ...],
                  timers: tuple[int, ...] = (), breaches: frozenset[int] = frozenset()) -> bool:
    if rule.op == "and":
        return all(condition_met(c, selected, hp, timers, breaches) for c in rule.children)
    if rule.op == "or":
        return any(condition_met(c, selected, hp, timers, breaches) for c in rule.children)
    if rule.op == "item":
        return rule.ref in selected
    if rule.op == "dead":
        return hp[rule.ref] == 0
    if rule.op == "status":
        return bool(timers and timers[rule.ref])
    if rule.op == "breach":
        return rule.ref in breaches
    if rule.op == "true":
        return True
    raise ValueError(f"Unknown condition {rule.op}")


def door_matches(world: World, state: State, door_id: int, selected: tuple[int, ...]) -> bool:
    door = world.doors[door_id]
    breached = any(
        effect.kind == "breach" and effect.target == door.material
        and condition_met(effect.requires, selected, state.enemy_hp)
        for item in selected for effect in world.effects_by_item[item]
    )
    materials = frozenset((door.material,)) if breached else frozenset()
    return condition_met(door.requires, selected, state.enemy_hp, breaches=materials)


def auto_open(world: World, state: State) -> State:
    return replace(state, opened=state.opened.union(
        i for i in range(len(world.doors)) if i not in state.opened and door_matches(world, state, i, ())
    ))


def initial_state(world: World) -> State:
    state = State(world.start, world.max_hp, tuple(sorted(world.starting_items)),
                  tuple(item.charges for item in world.items), tuple(enemy.hp for enemy in world.enemies),
                  tuple((0,) * len(STATUS) for _ in world.enemies), ground=tuple(sorted(world.ground)))
    return auto_open(world, state)


def tile_free(world: World, state: State, pos: Pos) -> bool:
    return (pos in world.floor
            and (pos not in world.door_at or world.door_at[pos] in state.opened)
            and (pos not in world.enemy_at or state.enemy_hp[world.enemy_at[pos]] == 0))


def legal_actions(world: World, state: State) -> tuple[Action, ...]:
    """Physical affordances. A wrong item attempt is legal and uses a turn."""
    if outcome(world, state) != "running":
        return ()
    actions = []
    owned = state.inventory
    weapons = tuple(i for i in owned if not world.items[i].key)
    for pos in world.floor_neighbors[state.pos]:
        if tile_free(world, state, pos):
            actions.append(Action("move", pos))
        if pos in world.door_at and world.door_at[pos] not in state.opened:
            for selected in choices(owned, world.max_combo):
                kind = "combine" if len(selected) > 1 else ("key" if world.items[selected[0]].key else "breach")
                actions.append(Action(kind, world.door_at[pos], selected))
        if pos in world.enemy_at and state.enemy_hp[world.enemy_at[pos]] > 0:
            enemy = world.enemy_at[pos]
            actions.append(Action("attack", enemy))
            actions.extend(Action("use", enemy, selected) for selected in choices(weapons, world.max_combo))
    for pos in (state.pos, *world.floor_neighbors[state.pos]):
        if pos in world.chest_at and world.chest_at[pos] not in state.looted:
            actions.append(Action("chest", world.chest_at[pos]))
    actions.extend(Action("heal", None, (item,)) for item in weapons)
    actions.append(WAIT)
    return tuple(actions)


def active_affinities(world: World, state: State, enemy: int, item: int,
                      selected: tuple[int, ...], timers: tuple[int, ...] | None = None) -> tuple[Affinity, ...]:
    timers = state.statuses[enemy] if timers is None else timers
    return tuple(a for a in world.affinity_by_enemy[enemy] if a.item == item
                 and condition_met(a.requires, selected, state.enemy_hp, timers))


def transition(world: World, state: State, action: Action, reward: Reward = Reward(),
               *, checked: bool = False) -> tuple[State, float]:
    if not checked and action not in legal_actions(world, state):
        raise ValueError(f"Illegal action {action!r}")
    pos, health = state.pos, state.hp
    owned, stock = set(state.inventory), list(state.stock)
    hp, timers = list(state.enemy_hp), [list(values) for values in state.statuses]
    opened, looted, ground = set(state.opened), set(state.looted), list(state.ground)
    used = set()
    target = action.target

    if action.kind == "move":
        pos = target
        owned.update(i for tile, i in ground if tile == pos)
        ground = [(tile, i) for tile, i in ground if tile != pos]
    elif action.kind == "chest":
        looted.add(target)
        owned.update(world.chests[target].items)
    elif action.kind in DOOR_ACTIONS and door_matches(world, state, target, action.items):
        opened.add(target)
        used.update(action.items)
    elif action.kind in ("attack", "use"):
        damage = int(action.kind == "attack")
        for item in action.items:
            for effect in world.effects_by_item[item]:
                if effect.target not in (-1, target) or effect.kind not in ("damage", *STATUS):
                    continue
                if not condition_met(effect.requires, action.items, state.enemy_hp, state.statuses[target]):
                    continue
                used.add(item)
                if effect.kind == "damage":
                    hit = effect.power
                    for affinity in active_affinities(world, state, target, item, action.items):
                        hit = hit * affinity.numerator // affinity.denominator
                    damage += hit
                else:
                    k = STATUS.index(effect.kind)
                    timers[target][k] = max(timers[target][k], effect.power)
        if state.statuses[target][2]:
            damage += (damage + 1) // 2
        armor = sum(link.power for link in world.supports if link.kind == "protect"
                    and link.target == target and hp[link.source] > 0)
        hp[target] = max(0, hp[target] - max(0, damage - armor))
    elif action.kind == "heal":
        for effect in world.effects_by_item[action.items[0]]:
            if effect.kind == "heal" and condition_met(effect.requires, action.items, state.enemy_hp):
                used.add(effect.item)
                health = min(world.max_hp, health + effect.power)

    spent = 0
    for item in used:
        if stock[item] > 0:
            spent += 1
            stock[item] -= 1
            if stock[item] == 0:
                owned.remove(item)

    for i in range(len(hp)):
        if hp[i] > 0 and timers[i][0]:
            hp[i] = max(0, hp[i] - 1)
        if hp[i] == 0 and state.enemy_hp[i] > 0:
            ground.extend((world.enemies[i].pos, item) for item in world.enemies[i].loot)

    healed = [0] * len(hp)
    boosted = [0] * len(hp)
    for link in world.supports:
        if hp[link.source] and hp[link.target]:
            if link.kind == "heal":
                healed[link.target] += link.power
            elif link.kind == "boost":
                boosted[link.target] += link.power
    hits = 0
    for i, enemy in enumerate(world.enemies):
        if hp[i] > 0:
            hp[i] = min(enemy.hp, hp[i] + healed[i])
            if distance(enemy.pos, pos) == 1:
                attack = enemy.attack + boosted[i]
                hits += attack // 2 if timers[i][1] else attack
    taken = min(health, hits)
    health -= taken
    after = auto_open(world, State(pos, health, tuple(sorted(owned)), tuple(stock), tuple(hp),
                                   tuple(tuple(max(0, t - 1) if hp[i] else 0 for t in values)
                                         for i, values in enumerate(timers)),
                                   frozenset(opened), frozenset(looted), tuple(sorted(ground)), state.turn + 1))
    result = outcome(world, after)
    score = -reward.turn - reward.hp * taken - reward.item * spent
    score += reward.goal if result == "success" else -reward.death if result == "death" else 0
    return after, score


def world_dict(world: World) -> dict:
    return asdict(world)


def condition_from(data: dict) -> Condition:
    return Condition(data["op"], data["ref"], tuple(condition_from(x) for x in data["children"]))


def world_from(data: dict) -> World:
    values = dict(data)
    for key in ("grid", "start", "goal", "starting_items", "critical_doors"):
        values[key] = tuple(data[key])
    values["rooms"] = tuple(tuple(tuple(pos) for pos in room) for room in data["rooms"])
    values["ground"] = tuple((tuple(pos), item) for pos, item in data["ground"])
    values["items"] = tuple(Item(**i) for i in data["items"])
    for field, cls in (("effects", Effect), ("affinities", Affinity)):
        values[field] = tuple(cls(**(entry | {"requires": condition_from(entry["requires"])})) for entry in data[field])
    values["supports"] = tuple(Support(**s) for s in data["supports"])
    values["doors"] = tuple(Door(tuple(d["pos"]), d["material"], condition_from(d["requires"])) for d in data["doors"])
    values["enemies"] = tuple(Enemy(tuple(e["pos"]), e["hp"], e["attack"], tuple(e["loot"])) for e in data["enemies"])
    values["chests"] = tuple(Chest(tuple(c["pos"]), tuple(c["items"])) for c in data["chests"])
    return World(**values)


def action_from(data: dict) -> Action:
    target = data["target"]
    return Action(data["kind"], tuple(target) if isinstance(target, list) else target, tuple(data["items"]))
