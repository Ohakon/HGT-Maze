from pathlib import Path
import unittest

from maze_hgt.config import load
from maze_hgt.evaluate import _query_map, held_out, interventions, local_oracle
from maze_hgt.game import initial_state, legal_actions
from maze_hgt.graph import observe
from maze_hgt.tasks import controlled


CFG = load(Path(__file__).resolve().parents[1] / "config/smoke.toml")


class EvaluationTests(unittest.TestCase):
    def test_intervention_pairs_keep_graph_size_and_flip_relevant_labels(self):
        for kind, first, second in interventions(CFG, 420):
            with self.subTest(kind=kind):
                left, right = initial_state(first), initial_state(second)
                a, b = observe(first, left), observe(second, right)
                self.assertEqual((len(a.nodes), len(a.edges)), (len(b.nodes), len(b.edges)))
                self.assertEqual(first.grid, second.grid)
                self.assertEqual(first.items, second.items)
                lq, rq = _query_map(first, left), _query_map(second, right)
                paired = [(lq[key].label, rq[key].label) for key in lq.keys() & rq.keys()]
                self.assertTrue(any(x != y for x, y in paired))
                self.assertTrue(any(x == y for x, y in paired))
                if kind in ("key", "weakness", "and_or", "death"):
                    before = local_oracle(kind, first, left, legal_actions(first, left))
                    after = local_oracle(kind, second, right, legal_actions(second, right))
                    self.assertTrue(before)
                    self.assertTrue(after)
                    self.assertNotEqual(before, after)

    def test_ood_uses_verified_larger_mazes_with_held_out_recipe(self):
        row = held_out(CFG, 1, ood=True)[0]
        self.assertTrue(row.solution.found)
        self.assertIn("compound", row.world.recipe.split("/"))
        self.assertGreater(len(row.world.grid), CFG.generation.max_height)
        self.assertGreater(len(row.world.critical_doors), CFG.generation.max_depth)

    def test_controlled_distractor_edges_leave_node_counts_and_types_fixed(self):
        cases = [controlled(19, "test", "edges", level) for level in (0, 4, 12)]
        self.assertEqual(len({len(case.graph.nodes) for case in cases}), 1)
        self.assertEqual(len({len({e.rel for e in case.graph.edges}) for case in cases}), 1)
        self.assertEqual([len(c.graph.edges) - len(cases[0].graph.edges) for c in cases], [0, 8, 24])
