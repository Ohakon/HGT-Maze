from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

import torch

from maze_hgt.config import Config
from maze_hgt.game import Action, Chest, Door, Condition, Effect, Enemy, initial_state, legal_actions, transition
from maze_hgt.graph import diff, observe, transition_spec
from maze_hgt.io import setup
from maze_hgt.metrics import Metrics
from maze_hgt.model import Architecture, Network, mask_logits
from maze_hgt.play import play, replay
from maze_hgt.rl import assign_targets, burn_in, collect, gae, importance_ratio, replay_loss, segments
from maze_hgt.solver import Search
from test_game import lab


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.cfg = replace(Config(), model=Architecture(dim=12, heads=3, layers=2))
        self.model = Network(self.cfg.model).to(setup(self.cfg))
        self.model.eval()

    def test_memory_matches_node_ids_resets_and_removes(self):
        world = lab(enemies=(Enemy((2, 3), 1, 1),), effects=(Effect(2, "damage", 0, 5),))
        state = initial_state(world)
        first = observe(world, state)
        frame = self.model.encode([first], [diff(None, first)])[0]
        self.assertIn("enemy:0", frame.memory.nodes)
        after, _ = transition(world, state, Action("use", 0, (2,)))
        graph = observe(world, after)
        events = diff(first, graph)
        original = self.model.encode([graph], [events], [frame.memory])[0]
        reordered = replace(graph, nodes=tuple(reversed(graph.nodes)))
        alt = self.model.encode([reordered], [events], [frame.memory])[0]
        self.assertNotIn("enemy:0", original.memory.nodes)
        for key in original.memory.nodes:
            torch.testing.assert_close(original.memory.nodes[key], alt.memory.nodes[key])
        clean, reset = (self.model.encode([first], [diff(None, first)], [memory])[0]
                        for memory in (None, frame.memory))
        torch.testing.assert_close(clean.nodes, reset.nodes)

    def test_actor_padding_and_illegal_action_mask(self):
        a = lab(chests=(Chest((2, 1), ()),))
        b = lab(starting_items=())
        worlds = [a, b]
        states = list(map(initial_state, worlds))
        graphs = [observe(w, s) for w, s in zip(worlds, states)]
        frames = self.model.encode(graphs)
        logits, mask = self.model.policy(frames, [legal_actions(w, s) for w, s in zip(worlds, states)])
        self.assertTrue((logits.softmax(1)[~mask] == 0).all())
        self.assertTrue(mask[0].sum() > mask[1].sum())
        self.assertTrue(torch.isneginf(mask_logits(torch.tensor([[0., 3.]]), torch.tensor([[True, False]]))[0, 1]))
        with self.assertRaises(ValueError):
            mask_logits(torch.zeros(1, 2), torch.zeros(1, 2, dtype=torch.bool))

    def test_balanced_accuracy_detects_majority_only_solvability(self):
        metrics = Metrics()
        metrics.binary("solvable", torch.ones(10), torch.tensor([True] * 9 + [False]))
        self.assertGreater(metrics.flat()["solvable.f1"], .94)
        self.assertEqual(metrics.flat()["solvable.balanced_accuracy"], .5)
        self.assertEqual(metrics.flat()["solvable.specificity"], 0.)

    def test_transition_heads_use_present_graph_and_action(self):
        world = lab(enemies=(Enemy((2, 3), 12, 1),), effects=(Effect(2, "poison", 0, 4),))
        state = initial_state(world)
        graph = observe(world, state)
        spec = transition_spec(graph)
        frame = self.model.encode([graph], [diff(None, graph)])[0]
        action = Action("use", 0, (2,))
        prediction = self.model.predict_transition(frame, action, spec)
        self.assertEqual(tuple(prediction.alive.shape), (len(graph.nodes),))
        self.assertGreater(len(prediction.birth), 0)
        self.assertEqual(prediction.edges.shape, (len(spec.edges), 4))
        prediction.edges.mean().backward()
        self.assertIsNotNone(self.model.hgt.features.weight.grad)

    def test_ppo_likelihood_ratio_and_recurrent_prefix_recomputation(self):
        world = lab(max_turns=6, enemies=(Enemy((2, 3), 8, 1),), effects=(Effect(2, "poison", 0, 4),))
        other = lab(starting_items=(), max_turns=5)
        rows = collect(self.model, [world, other], self.cfg, [0, 1])
        schedule = segments(rows, 2)
        self.assertTrue(any(chunk.start for chunk in schedule))
        stateful = burn_in(self.model, schedule)
        self.assertTrue(all((memory is not None) == (chunk.start > 0) for memory, chunk in zip(stateful, schedule)))
        assign_targets(rows, self.cfg, "ppo")
        book = Metrics()
        loss = replay_loss(self.model, schedule, self.cfg, "ppo", book, 5.)
        self.assertAlmostEqual(book.flat()["ratio"], 1., places=5)
        self.assertAlmostEqual(book.flat()["kl"], 0., places=6)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(self.model.gru.weight_ih.grad.norm().item(), 0.)
        self.assertAlmostEqual(importance_ratio(torch.tensor(-.2), torch.tensor(-.9)).item(), 2.01375, places=4)

    def test_gae_and_grpo_group_relative_returns(self):
        advantages, returns = gae([0., 1.], [.2, .4], [False, True], .9, .8)
        self.assertAlmostEqual(advantages[0], .592)
        self.assertAlmostEqual(returns[1], 1.)
        truncation, _ = gae([1.], [.1], [False], .9, .8, bootstrap=.5)
        self.assertAlmostEqual(truncation[0], 1.35)
        world = lab(max_turns=5)
        rows = collect(self.model, [world, world], self.cfg, [0, 0], self.model)
        rows[0].steps[-1].reward += 1.
        assign_targets(rows, self.cfg, "grpo")
        self.assertAlmostEqual(rows[0].steps[0].advantage, 1.)
        self.assertAlmostEqual(rows[1].steps[0].advantage, -1.)
        loss = replay_loss(self.model, segments(rows, 2), self.cfg, "grpo", Metrics(), 5.)
        self.assertTrue(torch.isfinite(loss))

    def test_play_and_replay_roundtrip(self):
        world = lab(items=(), starting_items=(), max_turns=8)
        with TemporaryDirectory() as tmp:
            path = Path(tmp)
            original = play(world, self.cfg, path, policy="solver", quiet=True)
            self.assertEqual(original["outcome"], "success")
            self.assertEqual(original, replay(path / "trace.json", path, quiet=True))
            trace = json.loads((path / "trace.json").read_text())
            trace["steps"][0]["reward"] = 10.
            (path / "trace.json").write_text(json.dumps(trace))
            with self.assertRaisesRegex(ValueError, "diverged"):
                replay(path / "trace.json", path, quiet=True)
