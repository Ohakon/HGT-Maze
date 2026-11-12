"""Pareto-distributed maze generation, disjoint data splits and solvability checks."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace
from functools import lru_cache
from hashlib import blake2b
from itertools import product
from random import Random
from string import ascii_lowercase

from .game import (AFFINITY, MATERIALS, STATUS, SUPPORT, Affinity, Chest, Condition, Door, Effect,
                   Enemy, Item, Pos, Support, World, all_of, any_of, neighbors)
from .solver import Search, Solution, solve


@dataclass(frozen=True)
class Generation:
    min_height: int = 7
    max_height: int = 13
    min_width: int = 7
    max_width: int = 13
    pareto: float = 1.9
    min_depth: int = 1
    max_depth: int = 3
    min_items: int = 4
    max_items: int = 10
    min_enemies: int = 2
    max_enemies: int = 5
    min_chests: int = 1
    max_chests: int = 5
    max_extra_doors: int = 2
    max_rooms: int = 6
    room_span: int = 3
    density: float = 0.3
    room_density: float = 0.35
    max_hp: int = 30
    max_turns: int = 90
    max_combo: int = 2
    finite_fraction: float = 0.3
    attempts: int = 120
    recipes: tuple[str, ...] = ("key", "breach", "combo", "death", "mixed")
    required_recipe: str = ""

    def __post_init__(self) -> None:
        for lower, upper in ((self.min_height, self.max_height), (self.min_width, self.max_width),
                             (self.min_depth, self.max_depth), (self.min_items, self.max_items),
                             (self.min_enemies, self.max_enemies), (self.min_chests, self.max_chests)):
            if lower < 0 or upper < lower:
                raise ValueError("Invalid generation limits")
        if min(self.min_height, self.min_width) < 5 or self.pareto <= 0:
            raise ValueError("Dimensions must be at least 5 and Pareto alpha must be positive")
        if not self.recipes or set(self.recipes) - {"key", "breach", "combo", "death", "mixed", "compound"}:
            raise ValueError("Unknown gate recipe")
        if self.required_recipe and (self.required_recipe not in self.recipes or self.max_depth == 0):
            raise ValueError("The required gate recipe must be in the available recipes and have a gate slot")
        if min(self.max_turns, self.max_hp, self.attempts, self.max_combo) < 1:
            raise ValueError("Invalid horizon, HP, attempts or combo limit")
        if any(not 0 <= p <= 1 for p in (self.density, self.room_density, self.finite_fraction)):
            raise ValueError("Generation densities must be between 0 and 1")


@dataclass(frozen=True)
class Generated:
    world: World
    solution: Solution | None
    attempt: int


def derived_seed(*parts: object) -> int:
    return int.from_bytes(blake2b(repr(parts).encode(), digest_size=8).digest(), "big")


def split_bucket(*parts: object) -> str:
    bucket = derived_seed(*parts) % 100
    return "train" if bucket < 70 else "val" if bucket < 85 else "test"


@lru_cache(maxsize=4)
def names(split: str) -> tuple[str, ...]:
    words = ("".join(parts) for parts in product(ascii_lowercase, repeat=3))
    return tuple(w for w in words if split == "demo" or split_bucket("item-name", w) == split)


def rename(world: World, seed: int, split: str = "demo") -> World:
    current = {item.name for item in world.items}
    pool = [word for word in names(split) if word not in current]
    return replace(world, items=tuple(replace(item, name=word)
                                     for item, word in zip(world.items, Random(seed).sample(pool, len(world.items)))))


def tail(rng: Random, lo: int, hi: int, alpha: float, scale: float = 2.) -> int:
    return min(hi, lo + int(scale * (rng.paretovariate(alpha) - 1)))


def odd_size(rng: Random, low: int, high: int, alpha: float) -> int:
    low += low % 2 == 0
    high -= high % 2 == 0
    if low > high:
        raise ValueError("Size bounds must contain an odd integer")
    return low + 2 * tail(rng, 0, (high - low) // 2, alpha)


def path_in(floor: set[Pos], start: Pos) -> dict[Pos, Pos | None]:
    parents = {start: None}
    pending = deque([start])
    while pending:
        p = pending.popleft()
        for q in neighbors(p):
            if q in floor and q not in parents:
                parents[q] = p
                pending.append(q)
    return parents


def recover(parents: dict[Pos, Pos | None], end: Pos) -> tuple[Pos, ...]:
    path = []
    while end is not None:
        path.append(end)
        end = parents[end]
    return tuple(reversed(path))


def carve(rng: Random, height: int, width: int) -> tuple[set[Pos], list[Pos]]:
    anchors = list(product(range(1, height - 1, 2), range(1, width - 1, 2)))
    anchor_set = set(anchors)
    start = rng.choice(anchors)
    floor = {start}
    stack = [start]
    while stack:
        r, c = stack[-1]
        nxt = [(2 * p - r, 2 * q - c) for p, q in neighbors((r, c))]
        candidates = [pos for pos in nxt if pos in anchor_set and pos not in floor]
        if not candidates:
            stack.pop()
            continue
        p, q = rng.choice(candidates)
        floor.update(((p, q), ((p + r) // 2, (q + c) // 2)))
        stack.append((p, q))
    return floor, anchors


def layout(rng: Random, cfg: Generation) -> tuple | None:
    height = odd_size(rng, cfg.min_height, cfg.max_height, cfg.pareto)
    width = odd_size(rng, cfg.min_width, cfg.max_width, cfg.pareto)
    floor, anchors = carve(rng, height, width)
    start = rng.choice(anchors)
    parents = path_in(floor, start)
    depth = tail(rng, cfg.min_depth, cfg.max_depth, cfg.pareto)
    paths = sorted((recover(parents, p) for p in anchors if p != start),
                   key=lambda path: (-len(path), path[-1]))
    paths = [path for path in paths if len(path) >= max(5, 4 * depth + 3)]
    if not paths:
        return None
    path = paths[tail(rng, 0, len(paths) - 1, cfg.pareto, 3.)]
    anchors_on_route = []
    for i in range(depth):
        nominal = (len(path) - 1) * (i + 1) // (depth + 1)
        lo = max(3, nominal - 1, anchors_on_route[-1] + 4 if anchors_on_route else 0)
        hi = min(len(path) - 3 - 4 * (depth - i - 1), max(lo, nominal + 1))
        anchors_on_route.append(rng.randint(lo, hi))
    gates = tuple(path[i] for i in anchors_on_route)
    regions = {}
    ungated = floor - set(gates)
    for group, index in enumerate((0, *(i + 1 for i in anchors_on_route))):
        regions.update({pos: group for pos in path_in(ungated, path[index])})
    # A gate placed on a junction may also isolate a side branch.
    next_group = len(gates) + 1
    while (unassigned := ungated - regions.keys()):
        regions.update({pos: next_group for pos in path_in(ungated, min(unassigned))})
        next_group += 1

    def extend(pos: Pos) -> bool:
        r, c = pos
        if not (0 < r < height - 1 and 0 < c < width - 1):
            return False
        if pos in floor:
            return pos in regions
        adjacent = tuple(neighbors(pos))
        groups = {regions[p] for p in adjacent if p in regions}
        if len(groups) != 1 or any(p in gates for p in adjacent):
            return False
        floor.add(pos)
        regions[pos] = groups.pop()
        return True

    rooms = []
    assigned = set()
    for center in rng.sample(anchors, min(len(anchors), tail(rng, min(1, cfg.max_rooms), cfg.max_rooms, cfg.pareto))):
        if center in gates or center in assigned:
            continue
        room = [center]
        span = tail(rng, 1, cfg.room_span, cfg.pareto)
        dr, dc = rng.choice((-1, 1)), rng.choice((-1, 1))
        for r, c in product(range(span), repeat=2):
            pos = center[0] + r * dr, center[1] + c * dc
            if pos not in assigned and pos not in room and rng.random() < cfg.room_density and extend(pos):
                if regions[pos] == regions[center] and any(q in room for q in neighbors(pos)):
                    room.append(pos)
        assigned.update(room)
        rooms.append(tuple(sorted(room)))
    loops = [pos for pos in product(range(1, height - 1), range(1, width - 1)) if pos not in floor
             and sum(q in floor for q in neighbors(pos)) > 1]
    for pos in rng.sample(loops, min(len(loops), tail(rng, 0, len(loops), cfg.pareto, cfg.room_density * 4))):
        extend(pos)
    grid = tuple("".join("." if (r, c) in floor else "#" for c in range(width)) for r in range(height))
    return grid, path, gates, regions, tuple(rooms)


@dataclass
class Contents:
    items: list[Item] = field(default_factory=list)
    effects: list[Effect] = field(default_factory=list)
    affinities: list[Affinity] = field(default_factory=list)
    supports: list[Support] = field(default_factory=list)
    doors: list[Door] = field(default_factory=list)
    enemies: list[Enemy] = field(default_factory=list)
    chests: list[Chest] = field(default_factory=list)
    starters: list[int] = field(default_factory=list)
    ground: list[tuple[Pos, int]] = field(default_factory=list)


def build(seed: int, cfg: Generation, split: str) -> World | None:
    rng = Random(seed)
    maze = layout(rng, cfg)
    if maze is None:
        return None
    grid, route, gates, regions, rooms = maze
    if split != "demo" and split_bucket("layout", grid) != split:
        return None
    c = Contents()
    taken = {route[0], route[-1], *gates}
    gate_recipes = []
    required_gate = rng.randrange(len(gates)) if cfg.required_recipe and gates else -1
    if cfg.required_recipe and required_gate < 0:
        return None

    def location(group: int | None = None, main: bool = False) -> Pos | None:
        eligible = [p for p in (route if main else sorted(regions))
                    if p not in taken and (group is None or regions.get(p) == group)]
        if not eligible:
            return None
        pos = rng.choice(eligible)
        taken.add(pos)
        return pos

    def item(key: bool = False, reusable: bool = True) -> int:
        i = len(c.items)
        charges = -1 if key or reusable else tail(rng, 1, 5, cfg.pareto)
        c.items.append(Item("", key, charges))
        return i

    def enemy(pos: Pos, hard: bool = False) -> int:
        i = len(c.enemies)
        c.enemies.append(Enemy(pos, tail(rng, 2, 9 if hard else 6, cfg.pareto),
                               tail(rng, 1, 3 if hard else 2, cfg.pareto)))
        return i

    def provide(ids: tuple[int, ...], group: int, drops: tuple[int, ...] = ()) -> None:
        if not ids:
            return
        if drops and rng.random() < .35:
            index = rng.choice(drops)
            c.enemies[index] = replace(c.enemies[index], loot=(*c.enemies[index].loot, *ids))
            return
        if len(c.chests) < cfg.max_chests and rng.random() < .78:
            pos = location(group, main=rng.random() < .85) or location(group)
            if pos is not None:
                c.chests.append(Chest(pos, ids))
                return
        if group and rng.random() < .7:
            pos = location(group, main=True) or location(group)
            if pos is not None:
                c.ground.extend((pos, i) for i in ids)
                return
        c.starters.extend(ids)

    available_keys = []
    for group, pos in enumerate(gates):
        available_recipes = (cfg.required_recipe,) if group == required_gate else cfg.recipes
        eligible_recipes = [r for r in available_recipes if (
            (r not in ("death", "mixed", "compound") or len(c.enemies) < cfg.max_enemies)
            and (r not in ("combo", "compound") or cfg.max_combo >= 2)
            and len(c.items) + {"key": 1, "breach": 1, "combo": 2, "death": 0,
                               "mixed": 1, "compound": 3}[r] <= cfg.max_items)]
        if not eligible_recipes:
            return None
        recipe = rng.choice(eligible_recipes)
        gate_recipes.append(recipe)
        material = rng.randrange(len(MATERIALS))
        kills = []
        if recipe in ("death", "mixed", "compound"):
            max_kills = min(2, cfg.max_enemies - len(c.enemies))
            count = tail(rng, 1, max_kills, cfg.pareto, 4.)
            if recipe == "compound":
                count = max_kills
            for _ in range(count):
                pos_enemy = location(group, main=rng.random() < .6) or location(group)
                if pos_enemy is None:
                    return None
                kills.append(enemy(pos_enemy))
        required = [Condition("dead", i) for i in kills]
        loot = []
        if recipe in ("key", "mixed", "compound"):
            key = item(key=True)
            loot.append(key)
            aliases = [Condition("item", key)]
            if available_keys and rng.random() < .65:
                aliases.append(Condition("item", rng.choice(available_keys)))
            if len(c.items) < cfg.max_items - 2 * (len(gates) - group - 1) and rng.random() < .35:
                alias = item(key=True)
                aliases.append(Condition("item", alias))
                provide((alias,), group + 1)
                available_keys.append(alias)
            available_keys.append(key)
            required.append(any_of(*aliases) if len(aliases) > 1 else aliases[0])
        if recipe in ("breach", "combo", "compound"):
            if len(c.items) + (2 if recipe in ("combo", "compound") else 1) > cfg.max_items:
                return None
            first = item()
            loot.append(first)
            auxiliary = Condition("true")
            if recipe in ("combo", "compound"):
                second = item()
                loot.append(second)
                auxiliary = Condition("item", second)
            c.effects.append(Effect(first, "breach", material, tail(rng, 1, 4, cfg.pareto), auxiliary))
            if recipe == "compound":
                c.effects.append(Effect(first, "damage", -1, tail(rng, 4, 8, cfg.pareto)))
            required.append(Condition("breach", material))
            if recipe == "combo":
                required.append(auxiliary)
        if recipe == "compound":
            required = [*required[:-2], any_of(*required[-2:])]
        provide(tuple(loot), group, () if recipe == "compound" else tuple(kills))
        c.doors.append(Door(pos, material, all_of(*required) if len(required) > 1 else required[0]))

    item_target = max(len(c.items), tail(rng, cfg.min_items, cfg.max_items, cfg.pareto, 4.))
    enemy_target = max(len(c.enemies), tail(rng, cfg.min_enemies, cfg.max_enemies, cfg.pareto, 3.))
    while len(c.enemies) < enemy_target:
        pos = location()
        if pos is None:
            return None
        enemy(pos, hard=True)
    chests_target = min(cfg.max_chests, max(cfg.min_chests, len(c.chests)) + tail(rng, 0, cfg.max_chests, cfg.pareto))
    while len(c.chests) < chests_target:
        pos = location()
        if pos is None:
            if len(c.chests) < cfg.min_chests:
                return None
            break
        c.chests.append(Chest(pos, ()))
    while len(c.items) < item_target:
        i = item(key=rng.random() < .22, reusable=rng.random() >= cfg.finite_fraction)
        if c.items[i].key:
            available_keys.append(i)
        owner = rng.choice(("start", "chest", "enemy", "floor"))
        if owner == "chest" and c.chests:
            j = rng.randrange(len(c.chests))
            c.chests[j] = replace(c.chests[j], items=(*c.chests[j].items, i))
        elif owner == "enemy" and c.enemies:
            j = rng.randrange(len(c.enemies))
            c.enemies[j] = replace(c.enemies[j], loot=(*c.enemies[j].loot, i))
        elif owner == "floor" and (pos := location()) is not None:
            c.ground.append((pos, i))
        else:
            c.starters.append(i)

    extra_count = tail(rng, 0, cfg.max_extra_doors, cfg.pareto, 3.)
    for _ in range(extra_count):
        pos = location()
        if pos is None:
            break
        material = rng.randrange(len(MATERIALS))
        alternatives = [Condition("breach", material)]
        alternatives.extend(Condition("item", i) for i in rng.sample(available_keys, min(2, len(available_keys))))
        if c.enemies and rng.random() < .5:
            alternatives.append(Condition("dead", rng.randrange(len(c.enemies))))
        c.doors.append(Door(pos, material, any_of(*alternatives) if len(alternatives) > 1 else alternatives[0]))

    usable = [i for i, obj in enumerate(c.items) if not obj.key]
    density = min(1., cfg.density * (0.6 + .25 * rng.paretovariate(cfg.pareto)))
    for i in usable:
        effects_n = tail(rng, 1, 4, cfg.pareto, 2.0 * density)
        for _ in range(effects_n):
            kind = rng.choice(("damage", "damage", *STATUS, "heal", "breach"))
            target = -1
            if kind == "breach":
                target = rng.randrange(len(MATERIALS))
            elif c.enemies and kind != "heal" and rng.random() < .55:
                target = rng.randrange(len(c.enemies))
            power = tail(rng, 2, 7, cfg.pareto)
            requirements = []
            if kind not in ("breach", "heal") and rng.random() < density:
                requirements.append(Condition("status", rng.randrange(len(STATUS))))
            if cfg.max_combo >= 2 and len(usable) > 1 and rng.random() < density / 2:
                requirements.append(Condition("item", rng.choice([j for j in usable if j != i])))
            rule = all_of(*requirements) if len(requirements) > 1 else (
                requirements[0] if requirements else Condition("true"))
            c.effects.append(Effect(i, kind, target, power, rule))
    for e in range(len(c.enemies)):
        for i in usable:
            if rng.random() < density:
                kind = rng.choice(AFFINITY)
                factor = (1, 2) if kind == "resist" else (2, 1) if kind == "weak" else (3, 1)
                rule = Condition("status", rng.randrange(len(STATUS))) if rng.random() < .5 else Condition("true")
                c.affinities.append(Affinity(e, i, kind, *factor, rule))
    for a in range(len(c.enemies)):
        for b in range(len(c.enemies)):
            if a != b and rng.random() < density:
                c.supports.append(Support(a, b, rng.choice(SUPPORT), tail(rng, 1, 3, cfg.pareto)))

    chosen_names = Random(derived_seed(seed, "names")).sample(names(split), len(c.items))
    items = tuple(replace(obj, name=name) for obj, name in zip(c.items, chosen_names))
    return World(grid, route[0], route[-1], rooms, items, tuple(c.effects), tuple(c.affinities),
                 tuple(c.supports), tuple(c.doors), tuple(c.enemies), tuple(c.chests),
                 tuple(c.starters), tuple(c.ground), cfg.max_hp, cfg.max_turns, cfg.max_combo,
                 seed, tuple(range(len(gates))), "/".join(gate_recipes))


def generate(seed: int, cfg: Generation = Generation(), search: Search = Search(),
             split: str = "demo", *, verify: bool = True) -> Generated:
    if split not in ("demo", "train", "val", "test"):
        raise ValueError("Unknown data split")
    for attempt in range(cfg.attempts):
        world = build(derived_seed(split, seed, attempt), cfg, split)
        if world is None:
            continue
        if not verify:
            return Generated(world, None, attempt)
        result = solve(world, config=search)
        if result.found:
            return Generated(world, result, attempt)
    raise RuntimeError(f"Could not generate a {'verified ' if verify else ''}maze for seed {seed} "
                       f"in {cfg.attempts} attempts. Check generation limits and search budget.")


def deadlock(world: World) -> World:
    """Move a required item behind the first articulation gate for a certified negative."""
    if not world.critical_doors or not world.items:
        raise ValueError("A deadlock variant requires an item and an articulation gate")
    i = world.critical_doors[0]
    item = 0
    doors = tuple(replace(d, requires=Condition("item", item)) if j == i else d for j, d in enumerate(world.doors))
    return replace(world, doors=doors, starting_items=tuple(k for k in world.starting_items if k != item),
                   chests=tuple(replace(c, items=tuple(k for k in c.items if k != item)) for c in world.chests),
                   enemies=tuple(replace(e, loot=tuple(k for k in e.loot if k != item)) for e in world.enemies),
                   ground=tuple((p, k) for p, k in world.ground if k != item) + ((world.goal, item),))
