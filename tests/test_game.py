from dataclasses import replace
import json
import unittest

from maze_hgt.game import (Action, Affinity, Chest, Condition, Door, Effect, Enemy, Item, Reward, Support,
                           WAIT, World, all_of, any_of, door_matches, initial_state, legal_actions, outcome,
                           transition, world_dict, world_from)
from maze_hgt.generate import Generation, deadlock, generate, rename
from maze_hgt.graph import FORWARD, diff, observe, transition_spec
from maze_hgt.solver import Search, Solution, combat_possible, solve
from maze_hgt.tasks import AXES, controlled, game_queries, pool_targets, value_target


GRID = ("#########", "#.......#", "#.......#", "#.......#", "#########")
ITEMS = (Item("abc", True), Item("def", True), Item("ghi"), Item("jkl", charges=2), Item("mno"))


def lab(**changes):
    return replace(World(GRID, (2, 2), (2, 7), items=ITEMS, starting_items=(0, 1, 2, 3, 4),
                         max_hp=30, max_turns=35), **changes)


class RulesTests(unittest.TestCase):
    def test_keys_or_material_and_combinations(self):
        material = Condition("breach", 2)
        key0, key1, required = (Condition("item", i) for i in (0, 1, 4))
        world = lab(doors=(Door((2, 3), 2, any_of(key0, key1)),
                           Door((1, 2), 2, any_of(key0, material)),
                           Door((3, 2), 2, all_of(material, required))),
                    effects=(Effect(2, "breach", 2, 1, required),))
        state = initial_state(world)
        for door, items, correct in ((0, (0,), True), (0, (1,), True), (1, (0,), True),
                                     (0, (2,), False), (1, (2, 4), True), (2, (2,), False),
                                     (2, (4,), False), (2, (2, 4), True)):
            self.assertEqual(door_matches(world, state, door, items), correct)
        wrong = Action("breach", 0, (2,))
        self.assertIn(wrong, legal_actions(world, state))
        unchanged, _ = transition(world, state, wrong)
        self.assertFalse(unchanged.opened)
        after, _ = transition(world, state, Action("combine", 2, (2, 4)))
        self.assertIn(2, after.opened)
        with self.assertRaises(ValueError):
            transition(world, state, Action("move", (4, 2)))

    def test_directed_enemy_boost_protect_and_heal(self):
        world = lab(enemies=(Enemy((1, 4), 9, 1), Enemy((2, 3), 12, 2)),
                    effects=(Effect(2, "damage", -1, 6),),
                    supports=(Support(0, 1, "boost", 3), Support(0, 1, "protect", 2), Support(0, 1, "heal", 1)))
        state = initial_state(world)
        hit = Action("use", 1, (2,))
        after, _ = transition(world, state, hit)
        self.assertEqual(after.enemy_hp, (9, 9))
        self.assertEqual(after.hp, 25)
        after, _ = transition(world, replace(state, enemy_hp=(0, 12)), hit)
        self.assertEqual(after.enemy_hp, (0, 6))
        self.assertEqual(after.hp, 28)

    def test_status_conditional_damage_and_edge_attribute_events(self):
        world = lab(enemies=(Enemy((2, 3), 20, 4),),
                    effects=(Effect(3, "poison", 0, 4), Effect(3, "weaken", 0, 4),
                             Effect(2, "damage", 0, 2)),
                    affinities=(Affinity(0, 2, "critical", 3, requires=Condition("status", 0)),))
        state = initial_state(world)
        raw, _ = transition(world, state, Action("use", 0, (2,)))
        poisoned, _ = transition(world, state, Action("use", 0, (3,)))
        critical, _ = transition(world, poisoned, Action("use", 0, (2,)))
        self.assertEqual(raw.enemy_hp[0], 18)
        self.assertEqual(poisoned.enemy_hp[0], 19)
        self.assertEqual(poisoned.hp, 28)
        self.assertEqual(critical.enemy_hp[0], 12)
        before, after = observe(world, state), observe(world, poisoned)
        creation = diff(before, after)
        self.assertTrue(any(e.kind == "node_add" and e.subject_type == "status" for e in creation))
        valid_edges = set(transition_spec(before).edges)
        self.assertTrue(all((e.subject, e.relation, e.other) in valid_edges for e in creation
                            if e.kind == "edge_add" and e.relation in FORWARD))
        second = diff(after, observe(world, critical))
        self.assertTrue(any(e.kind == "edge_change" and e.relation == "has_status" for e in second))
        self.assertTrue(any(e.kind == "edge_change" and e.relation == "holds" for e in creation))
        for _ in range(2):
            critical, _ = transition(world, critical, WAIT)
        self.assertFalse(any(critical.statuses[0]))
        self.assertNotIn("status:0:0", observe(world, critical).index)

    def test_deaths_open_gate_drop_and_transfer_ownership(self):
        world = lab(enemies=(Enemy((1, 2), 2, 1, (0,)), Enemy((2, 3), 2, 1)),
                    doors=(Door((3, 2), 0, all_of(Condition("dead", 0), Condition("dead", 1))),),
                    effects=(Effect(2, "damage", -1, 4),),
                    chests=(Chest((2, 1), (1,)),), starting_items=(2, 3, 4))
        state = initial_state(world)
        chested, _ = transition(world, state, Action("chest", 0))
        chest_events = diff(observe(world, state), observe(world, chested))
        self.assertTrue(any(e.kind == "edge_add" and e.other == "item:1" and e.relation == "holds" for e in chest_events))
        self.assertFalse(any(e.kind.startswith("node_") and e.subject == "item:1" for e in chest_events))
        killed, _ = transition(world, chested, Action("use", 0, (2,)))
        self.assertNotIn(0, killed.opened)
        self.assertIn(((1, 2), 0), killed.ground)
        events = diff(observe(world, chested), observe(world, killed))
        removal = next(e for e in events if e.kind == "node_remove" and e.subject == "enemy:0")
        self.assertIn("item:0", removal.related)
        self.assertNotIn("enemy:0", observe(world, killed).index)
        collected, _ = transition(world, killed, Action("move", (1, 2)))
        self.assertIn(0, collected.inventory)
        self.assertFalse(collected.ground)
        killed, _ = transition(world, killed, Action("use", 1, (2,)))
        self.assertIn(0, killed.opened)
        self.assertNotIn("door:0", observe(world, killed).index)

    def test_consumable_and_reward_incentives(self):
        world = lab(items=(*ITEMS[:3], replace(ITEMS[3], charges=1), ITEMS[4]),
                    enemies=(Enemy((2, 3), 4, 1),), effects=(Effect(3, "damage", 0, 8),),
                    chests=(Chest((2, 1), ()),))
        state = initial_state(world)
        after, hit = transition(world, state, Action("use", 0, (3,)))
        _, wait = transition(world, state, WAIT)
        _, chest = transition(world, state, Action("chest", 0))
        self.assertEqual(hit, wait)
        self.assertEqual(chest, wait)
        self.assertNotIn(3, after.inventory)
        self.assertNotIn("item:3", observe(world, after).index)
        _, paid = transition(world, state, Action("use", 0, (3,)), Reward(item=.2))
        self.assertAlmostEqual(paid, hit - .2)
        doomed = replace(state, hp=1)
        self.assertEqual(outcome(world, transition(world, doomed, WAIT)[0]), "death")


class SolverAndDatasetTests(unittest.TestCase):
    def test_defeatability_considers_poison_that_opens_a_route(self):
        world = World(("#######", "#.....#", "#######"), (1, 1), (1, 5),
                      enemies=(Enemy((1, 2), 1, 1), Enemy((1, 4), 2, 1)),
                      doors=(Door((1, 3), 0, Condition("dead", 0)),))
        state = initial_state(world)
        self.assertFalse(combat_possible(world, state, 1, 200))
        poisoned = replace(state, statuses=((2, 0, 0), (0, 0, 0)))
        self.assertTrue(combat_possible(world, poisoned, 1, 200))

    def test_shortest_path_all_ties_unknown_and_discounted_value(self):
        world = lab(goal=(3, 3), starting_items=(), items=())
        state = initial_state(world)
        solution = solve(world, state, Search(mode="exact"), ties=True)
        self.assertEqual(solution.status, "optimal")
        self.assertEqual(len(solution.actions), 2)
        self.assertEqual(set(solution.best_actions), {Action("move", (3, 2)), Action("move", (2, 3))})
        self.assertEqual(solve(world, state, Search(mode="exact", budget=0)).status, "unknown")
        target = value_target(world, state, solution, Reward(turn=.05), .9, .3)
        self.assertTrue(target.exact)
        self.assertAlmostEqual(target.value, -.05 + .9 * .95)
        approximate = value_target(world, state, solution, Reward(hp=.2), .9, .3)
        self.assertFalse(approximate.exact)
        self.assertEqual(approximate.weight, .3)
        self.assertIsNone(value_target(world, state, Solution("unknown"), Reward(), .9, .3))

    def test_seed_roundtrip_disjoint_names_and_verified_worlds(self):
        cfg = Generation(max_height=9, max_width=9, max_items=7, max_depth=2, max_enemies=3)
        train = generate(27, cfg, split="train")
        self.assertEqual(train, generate(27, cfg, split="train"))
        val = generate(27, cfg, split="val")
        test = generate(27, cfg, split="test")
        all_worlds = (train.world, val.world, test.world)
        self.assertEqual(len({w.grid for w in all_worlds}), 3)
        self.assertEqual(sum(len(w.items) for w in all_worlds), len({i.name for w in all_worlds for i in w.items}))
        world = world_from(json.loads(json.dumps(world_dict(train.world))))
        self.assertEqual(world, train.world)
        state = initial_state(world)
        for action in train.solution.actions:
            state, _ = transition(world, state, action)
        self.assertEqual(outcome(world, state), "success")
        negative = deadlock(world)
        certificate = solve(negative, config=Search(mode="guided", budget=0))
        self.assertEqual(certificate.status, "impossible")
        self.assertEqual(pool_targets(negative, initial_state(negative), certificate, Search())[-1], 0.)

    def test_rule_labels_do_not_use_direct_answer_edges_or_names(self):
        world = lab(doors=(Door((2, 3), 2, Condition("breach", 2)),
                           Door((2, 5), 1, Condition("breach", 1))),
                    effects=(Effect(2, "breach", 2, 1),))
        state = initial_state(world)
        graph = observe(world, state)
        labels = {(q.task, q.nodes): q.label for q in game_queries(world, state)}
        self.assertTrue(labels["item_door", ("item:2", "door:0")])
        self.assertFalse(labels["item_door", ("item:2", "door:1")])
        self.assertFalse(any(e.src == "item:2" and e.dst.startswith("door:") for e in graph.edges))
        self.assertEqual(graph, observe(rename(world, 42), state))

    def test_controlled_hops_have_balanced_labels_and_constant_graph_size(self):
        hops = [controlled(18, "test", "hops", h) for h in range(1, 6)]
        self.assertEqual(len({(len(case.graph.nodes), len(case.graph.edges)) for case in hops}), 1)
        for axis, levels in AXES.items():
            for level in levels:
                case = controlled(99, "test", axis, level)
                self.assertEqual({q.label for q in case.queries}, {False, True})
                self.assertTrue(all(q.nodes[0] in case.graph.index for q in case.queries))
