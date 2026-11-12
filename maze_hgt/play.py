"""Complete episode logs and exact replay using the formal transition function."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from random import Random
import json

import torch
from torch.distributions import Categorical

from .config import Config
from .game import Reward, State, World, action_from, initial_state, outcome, transition, world_dict, world_from
from .graph import diff, observe
from .io import write_json
from .model import Network
from .render import Viewer, action_text, ascii_board, rules_text
from .solver import Solution, solve
from .tasks import observation


def snapshot(state: State) -> dict:
    raw = asdict(state)
    for key in ("opened", "looted"):
        raw[key] = sorted(raw[key])
    return json.loads(json.dumps(raw))


@torch.no_grad()
def play(world: World, cfg: Config, output: Path, *, policy: str, model: Network | None = None,
         sample: bool = False, mode: str = "ascii", quiet: bool = False,
         solution: Solution | None = None) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    if policy == "model" and model is None:
        raise ValueError("A model checkpoint is required")
    if model:
        model.eval()
    reference = solution or solve(world, config=cfg.search)
    if policy == "solver" and not reference.found:
        raise ValueError(f"Solver {reference.status}: cannot play a solution")
    state, memory = initial_state(world), None
    obs = observation(world, state)
    rules = rules_text(world)
    (output / "rules.txt").write_text(rules + "\n", encoding="utf-8")
    if not quiet:
        print(rules)
    trace = {"format": 1, "world": world_dict(world), "policy": policy, "sample": sample,
             "reward": asdict(cfg.reward), "gamma": cfg.optimizer.gamma, "initial": snapshot(state),
             "initial_graph": asdict(obs.graph), "initial_events": [asdict(e) for e in obs.events],
             "reference": {"status": reference.status, "turns": len(reference.actions) if reference.found else None,
                           "expanded": reference.expanded, "actions": [asdict(a) for a in reference.actions]},
             "steps": []}
    viewer = Viewer(mode, output / "frames")
    rng = Random(cfg.runtime.seed)
    viewer.frame(world, state, obs.graph, print_ascii=not quiet)
    try:
        while outcome(world, state) == "running":
            actions = obs.actions
            value = None
            if policy == "model":
                frames = model.encode([obs.graph], [obs.events], [memory])
                memory = frames[0].memory
                logits, _ = model.policy(frames, [actions])
                dist = Categorical(logits=logits[0])
                probs = dist.probs.tolist()
                index = int(dist.sample()) if sample else int(logits[0].argmax())
                value = model.values(frames)[0].item()
            elif policy == "solver":
                index = actions.index(reference.actions[state.turn])
                probs = [float(i == index) for i in range(len(actions))]
            else:
                index = rng.randrange(len(actions))
                probs = [1. / len(actions)] * len(actions)
            action = actions[index]
            after, reward = transition(world, state, action, cfg.reward, checked=True)
            following = observation(world, after, obs.graph)
            trace["steps"].append({"turn": state.turn, "before_ascii": ascii_board(world, state),
                                   "ascii": ascii_board(world, after), "chosen": index, "action": asdict(action),
                                   "label": action_text(world, action),
                                   "candidates": [{"action": asdict(a), "probability": p,
                                                   "label": action_text(world, a)} for a, p in zip(actions, probs)],
                                   "value": value, "reward": reward, "state": snapshot(after),
                                   "events": [asdict(event) for event in following.events]})
            if not quiet:
                print(f"行動 {action_text(world, action)} / p={probs[index]:.3f} / "
                      f"V={value if value is None else round(value, 4)} / 報酬 {reward:+.3f}")
            state, obs = after, following
            viewer.frame(world, state, obs.graph, print_ascii=not quiet)
    finally:
        viewer.close()
    trace["summary"] = {"outcome": outcome(world, state), "turns": state.turn,
                        "return": sum(s["reward"] for s in trace["steps"]),
                        "discounted_return": sum(cfg.optimizer.gamma ** t * s["reward"]
                                                 for t, s in enumerate(trace["steps"]))}
    write_json(output / "trace.json", trace)
    return trace["summary"]


def replay(path: Path, output: Path, mode: str = "ascii", quiet: bool = False) -> dict:
    trace = json.loads(path.read_text(encoding="utf-8"))
    world = world_from(trace["world"])
    reward = Reward(**trace["reward"])
    state = initial_state(world)
    if snapshot(state) != trace["initial"]:
        raise ValueError("Initial state differs from the recorded state")
    if not quiet:
        print(rules_text(world))
    graph = observe(world, state)
    viewer = Viewer(mode, output / "frames")
    viewer.frame(world, state, graph, print_ascii=not quiet)
    try:
        for row in trace["steps"]:
            action = action_from(row["action"])
            state, score = transition(world, state, action, reward)
            after = observe(world, state)
            events = json.loads(json.dumps([asdict(e) for e in diff(graph, after)]))
            if (snapshot(state) != row["state"] or score != row["reward"] or events != row["events"]
                    or ascii_board(world, state) != row["ascii"]):
                raise ValueError(f"Replay diverged at turn {state.turn}")
            if not quiet:
                print(f"再生 {action_text(world, action)} / 報酬 {score:+.3f}")
            viewer.frame(world, state, after, print_ascii=not quiet)
            graph = after
    finally:
        viewer.close()
    if outcome(world, state) != trace["summary"]["outcome"] or state.turn != trace["summary"]["turns"]:
        raise ValueError("Replay outcome differs from the recorded outcome")
    return trace["summary"]
