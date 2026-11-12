"""Shortest-path certificates for small mazes and bounded guided search for larger ones."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from heapq import heappop, heappush
from itertools import count

from .game import (DOOR_ACTIONS, Action, Pos, State, World, choices, door_matches,
                   initial_state, legal_actions, outcome, tile_free, transition)


@dataclass(frozen=True)
class Search:
    mode: str = "auto"
    budget: int = 8000
    exact_tiles: int = 64
    weight: float = 2.0
    combat_budget: int = 500


@dataclass(frozen=True)
class Solution:
    status: str
    actions: tuple[Action, ...] = ()
    best_actions: tuple[Action, ...] = ()
    expanded: int = 0

    @property
    def found(self) -> bool:
        return self.status in ("optimal", "feasible")


def flood(world: World, pos: Pos, blocked: set[Pos] | frozenset[Pos]) -> dict[Pos, int]:
    depths = {pos: 0}
    frontier = deque([pos])
    while frontier:
        current = frontier.popleft()
        for other in world.floor_neighbors[current]:
            if other not in depths and other not in blocked:
                depths[other] = depths[current] + 1
                frontier.append(other)
    return depths


def reach(world: World, state: State) -> dict[Pos, int]:
    blocked = {pos for pos in (*world.door_at, *world.enemy_at) if not tile_free(world, state, pos)}
    return flood(world, state.pos, blocked)


def optimistic_reach(world: World, state: State) -> set[Pos]:
    """Ignore HP, spending, and combat and let reachable enemies yield loot/death."""
    opened, held, enemy_hp = set(state.opened), set(state.inventory), list(state.enemy_hp)
    while True:
        area = set(flood(world, state.pos, {d.pos for i, d in enumerate(world.doors) if i not in opened}))

        def adjacent(pos: Pos) -> bool:
            return pos in area or any(p in area for p in world.floor_neighbors[pos])

        for i, enemy in enumerate(world.enemies):
            if enemy_hp[i] and adjacent(enemy.pos):
                enemy_hp[i] = 0
                held.update(enemy.loot)
        for i, chest in enumerate(world.chests):
            if i not in state.looted and adjacent(chest.pos):
                held.update(chest.items)
        held.update(item for pos, item in state.ground if pos in area)
        proxy = replace(state, enemy_hp=tuple(enemy_hp))
        bundles = tuple(choices(tuple(sorted(held)), world.max_combo, include_empty=True))
        unlocked = {i for i, door in enumerate(world.doors) if i not in opened and adjacent(door.pos)
                    and any(door_matches(world, proxy, i, bundle) for bundle in bundles)}
        if not unlocked:
            return area
        opened.update(unlocked)


def solve(world: World, state: State | None = None, config: Search = Search(),
          *, ties: bool = False) -> Solution:
    state = initial_state(world) if state is None else state
    result = outcome(world, state)
    if result != "running":
        return Solution("optimal" if result == "success" else "impossible")
    if world.goal not in optimistic_reach(world, state):
        return Solution("impossible")
    exact = config.mode == "exact" or (config.mode == "auto" and len(world.floor) <= config.exact_tiles)
    if config.mode not in ("auto", "exact", "guided"):
        raise ValueError("search mode must be auto, exact or guided")
    goal_dist = flood(world, world.goal, set())
    blocked_gate_bounds = {}
    if exact:
        for i, door in enumerate(world.doors):
            if i not in state.opened and not door_matches(
                    world, replace(state, enemy_hp=(0,) * len(state.enemy_hp)), i, ()):
                blocked_gate_bounds[i] = flood(world, world.goal, {door.pos})

    def estimate(current: State) -> float:
        remaining = goal_dist[current.pos]
        if exact:
            return remaining + sum(i not in current.opened and current.pos not in side
                                   for i, side in blocked_gate_bounds.items())
        progress = (len(current.opened) * 8 + len(current.looted) * 4
                    + sum(a - b for a, b in zip(state.enemy_hp, current.enemy_hp)) * 1.5
                    + len(current.inventory) * 3)
        return config.weight * remaining - progress + (state.hp - current.hp) * 0.1

    serial = count()
    start_key = state.search_key()
    agenda = [(estimate(state), 0, next(serial), state, (), frozenset())]
    visited = {start_key: (0, frozenset())}
    winning = None
    firsts = set()
    expanded = 0
    while agenda:
        rank, _, _, current, path, roots = heappop(agenda)
        if winning is not None and rank > len(winning):
            return Solution("optimal", winning, tuple(a for a in legal_actions(world, state) if a in firsts), expanded)
        key = current.search_key()
        g = len(path)
        if visited[key] != (g, roots):
            continue
        if outcome(world, current) == "success":
            if winning is None:
                winning = path
            firsts.update(roots)
            if not exact or not ties:
                return Solution("optimal" if exact else "feasible", winning, winning[:1], expanded)
            continue
        if outcome(world, current) != "running" or g + goal_dist[current.pos] > world.max_turns - state.turn:
            continue
        if expanded >= config.budget:
            return (Solution("feasible", winning, winning[:1], expanded)
                    if winning is not None else Solution("unknown", expanded=expanded))
        expanded += 1
        for action in legal_actions(world, current):
            if (action.kind in DOOR_ACTIONS and (path or not ties)
                    and not door_matches(world, current, action.target, action.items)):
                continue  # The unsuccessful transition is also represented by wait.
            after, _ = transition(world, current, action, checked=True)
            if after.hp <= 0:
                continue
            successor = after.search_key()
            roots_after = roots if path else frozenset((action,))
            prior = visited.get(successor)
            if prior and (prior[0] < g + 1 or (prior[0] == g + 1 and
                                               (not ties or roots_after <= prior[1]))):
                continue
            if prior and prior[0] == g + 1:
                roots_after = roots_after | prior[1]
            visited[successor] = (g + 1, roots_after)
            heappush(agenda, (g + 1 + estimate(after), -(g + 1), next(serial),
                             after, (*path, action), roots_after))
    if winning is not None:
        return Solution("optimal", winning, tuple(a for a in legal_actions(world, state) if a in firsts), expanded)
    return Solution("impossible" if exact else "unknown", expanded=expanded)


def combat_possible(world: World, state: State, enemy: int, budget: int) -> bool | None:
    """Can the target be killed using movement, current supplies and waiting?"""
    reachable = reach(world, state)
    can_approach = any(pos in reachable for pos in world.floor_neighbors[world.enemies[enemy].pos])
    pending_poison = any(timers[0] for timers in state.statuses)
    if not can_approach and not pending_poison:
        return False
    agenda = deque([state])
    seen = {state.search_key()}
    while agenda:
        if budget <= 0:
            return None
        budget -= 1
        current = agenda.popleft()
        for action in legal_actions(world, current):
            if action.kind not in ("move", "heal", "wait") and not (
                    action.kind in ("attack", "use") and action.target == enemy):
                continue
            after, _ = transition(world, current, action, checked=True)
            if after.hp <= 0 or any(i not in current.inventory for i in after.inventory):
                continue
            if not after.enemy_hp[enemy]:
                return True
            key = after.search_key()
            if outcome(world, after) == "running" and key not in seen:
                seen.add(key)
                agenda.append(after)
    return False
