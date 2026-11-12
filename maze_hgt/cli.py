"""Independent entry points for generation, training, evaluation and play."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

from .config import load
from .evaluate import evaluate, held_out, relation_benchmark
from .game import world_dict, world_from
from .generate import generate
from .io import load_model, setup, write_json
from .play import play, replay
from .render import curves
from .rl import reinforce
from .solver import solve
from .train import pretrain


def source(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--world", type=Path, help="Previously exported world.json")
    group.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", choices=("demo", "train", "val", "test"), default="demo")


def rendering(parser: argparse.ArgumentParser, default: str) -> None:
    parser.add_argument("--render", choices=("ascii", "frames", "live"), default=default)
    parser.add_argument("--quiet", action="store_true")


def checkpoints(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint", action="append", default=[], metavar="LABEL=PATH")
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--axis-repeats", type=int)
    parser.add_argument("--no-ood", action="store_true")


def parser_for_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="maze-hgt", description="Relational maze and recurrent HGT experiments")
    p.add_argument("--config", type=Path, help="TOML settings, e.g. config/smoke.toml")
    commands = p.add_subparsers(dest="command", required=True)
    gen = commands.add_parser("generate", help="Generate verified mazes and export complete rules")
    gen.add_argument("--seed", type=int, default=42)
    gen.add_argument("--count", type=int, default=1)
    gen.add_argument("--split", choices=("demo", "train", "val", "test"), default="demo")
    gen.add_argument("--unverified", action="store_true")
    gen.add_argument("--out", type=Path, required=True)
    solver = commands.add_parser("solve", help="Return a reference solution or a search status")
    source(solver)
    solver.add_argument("--mode", choices=("auto", "exact", "guided"))
    solver.add_argument("--ties", action="store_true")
    solver.add_argument("--out", type=Path)
    check = commands.add_parser("check", help="Generate, solve, display and replay an episode")
    source(check)
    rendering(check, "frames")
    check.add_argument("--out", type=Path, required=True)
    game = commands.add_parser("play", help="Play with the model, reference solver or random policy")
    source(game)
    rendering(game, "frames")
    game.add_argument("--policy", choices=("model", "solver", "random"))
    game.add_argument("--model", type=Path)
    game.add_argument("--sample", action="store_true")
    game.add_argument("--out", type=Path, required=True)
    replay_cmd = commands.add_parser("replay", help="Replay and verify an exported trace")
    replay_cmd.add_argument("trace", type=Path)
    replay_cmd.add_argument("--out", type=Path, default=Path("replay"))
    rendering(replay_cmd, "ascii")
    train = commands.add_parser("pretrain", help="Train one supervised stage or the complete sequence")
    train.add_argument("--stage", choices=("all", "1", "2", "3", "4", "5"), default="all")
    train.add_argument("--out", type=Path, required=True)
    initial = train.add_mutually_exclusive_group()
    initial.add_argument("--init", type=Path)
    initial.add_argument("--resume", type=Path)
    for name in ("ppo", "grpo"):
        rl = commands.add_parser(name, help=f"Run recurrent {name.upper()}")
        initial = rl.add_mutually_exclusive_group(required=True)
        initial.add_argument("--init", type=Path)
        initial.add_argument("--resume", type=Path)
        initial.add_argument("--from-scratch", action="store_true")
        rl.add_argument("--allow-unready", action="store_true", help="Run an experiment with unmet pretraining targets")
        rl.add_argument("--out", type=Path, required=True)
    bench = commands.add_parser("evaluate", help="Compare held-out games, interventions and reasoning")
    checkpoints(bench)
    bench.add_argument("--out", type=Path, required=True)
    relations = commands.add_parser("relations", help="Evaluate only HGT relation heads")
    checkpoints(relations)
    relations.add_argument("--out", type=Path, required=True)
    ablate = commands.add_parser("ablate", help="Train and compare recurrent, stateless and event-input variants")
    ablate.add_argument("--base-run", type=Path, help="Directory with pretrain/, ppo/ and grpo/ runs to reuse")
    ablate.add_argument("--episodes", type=int, default=8)
    ablate.add_argument("--axis-repeats", type=int)
    ablate.add_argument("--no-ood", action="store_true")
    ablate.add_argument("--allow-unready", action="store_true")
    ablate.add_argument("--out", type=Path, required=True)
    plot = commands.add_parser("plot", help="Plot supervised and reinforcement learning metrics")
    plot.add_argument("--run", type=Path, required=True)
    plot.add_argument("--out", type=Path, required=True)
    return p


def input_world(args, cfg):
    if args.world:
        return world_from(json.loads(args.world.read_text(encoding="utf-8"))), None
    obj = generate(args.seed, cfg.generation, cfg.search, args.split)
    return obj.world, obj.solution


def train_if_needed(cfg, device, output: Path, allow_unready: bool, algorithms: tuple[str, ...]) -> dict[str, Path]:
    path = output / "pretrain" / "stage5" / "best.pt"
    if not path.exists():
        path = pretrain(cfg, device, output / "pretrain")
    entries = {"pretrained": path}
    for algorithm in algorithms:
        best = output / algorithm / "best.pt"
        if not best.exists():
            best = reinforce(cfg, setup(cfg), output / algorithm, algorithm, path, allow_unready=allow_unready)
        entries[algorithm] = best
    return entries


def ablations(args, cfg, device) -> dict:
    base = args.base_run or args.out / "full"
    paths = train_if_needed(cfg, device, base, args.allow_unready, ("ppo", "grpo"))
    for name, model in (("stateless", replace(cfg.model, gru=False, events=False)),
                        ("no_events", replace(cfg.model, events=False))):
        alt = replace(cfg, model=model)
        row = train_if_needed(alt, setup(alt), args.out / name, args.allow_unready, ("ppo",))
        paths.update({f"{name}_{key}": value for key, value in row.items()})
    scratch = args.out / "scratch" / "ppo"
    if not (scratch / "best.pt").exists():
        reinforce(cfg, setup(cfg), scratch, "ppo", None, from_scratch=True)
    paths["scratch_ppo"] = scratch / "best.pt"
    models = {key: load_model(path, device)[0] for key, path in paths.items()}
    return evaluate(cfg, models, args.out / "evaluation.json", args.episodes,
                    repeats=args.axis_repeats, with_ood=not args.no_ood)


def model_set(values: list[str], device) -> dict:
    result = {}
    for item in values:
        label, equals, name = item.partition("=")
        if not equals or not label or not name or label in {"reference", "random", *result}:
            raise ValueError("Use unique --checkpoint LABEL=PATH values")
        result[label] = load_model(Path(name), device)[0]
    if not result:
        raise ValueError("Supply at least one --checkpoint LABEL=PATH")
    return result


def dispatch(args) -> None:
    cfg = load(args.config)
    cmd = args.command
    if cmd == "generate":
        if args.count < 1:
            raise ValueError("Count must be positive")
        manifest = []
        for index in range(args.count):
            seed = args.seed + index
            obj = generate(seed, cfg.generation, cfg.search, args.split, verify=not args.unverified)
            path = args.out / f"world-{seed}.json"
            write_json(path, world_dict(obj.world))
            manifest.append({"seed": seed, "split": args.split, "file": path.name, "attempt": obj.attempt,
                             "world_seed": obj.world.seed, "reference": asdict(obj.solution) if obj.solution else None})
        write_json(args.out / "manifest.json", manifest)
        print(f"Generated {len(manifest)} mazes in {args.out}")
        return
    if cmd == "solve":
        if args.world:
            world, _ = input_world(args, cfg)
        else:
            world = generate(args.seed, cfg.generation, cfg.search, args.split, verify=False).world
        settings = replace(cfg.search, mode=args.mode) if args.mode else cfg.search
        result = solve(world, config=settings, ties=args.ties)
        raw = asdict(result) | {"turns": len(result.actions) if result.found else None}
        if args.out:
            write_json(args.out / "world.json", world_dict(world))
            write_json(args.out / "solution.json", raw)
        print(json.dumps(raw, ensure_ascii=False))
        return
    if cmd == "replay":
        print(json.dumps(replay(args.trace, args.out, args.render, args.quiet), ensure_ascii=False))
        return
    if cmd == "plot":
        curves(args.run, args.out)
        print(args.out)
        return
    device = setup(cfg)
    if cmd == "pretrain":
        if args.init and args.resume:
            raise ValueError("Select --init or --resume")
        stage = None if args.stage == "all" else int(args.stage)
        print(pretrain(cfg, device, args.out, stage, args.init, args.resume))
    elif cmd in ("ppo", "grpo"):
        print(reinforce(cfg, device, args.out, cmd, args.init, allow_unready=args.allow_unready,
                        from_scratch=args.from_scratch, resume=args.resume))
    elif cmd in ("check", "play"):
        world, solution = input_world(args, cfg)
        model_path = getattr(args, "model", None)
        policy = "solver" if cmd == "check" else args.policy or ("model" if model_path else "solver")
        model = load_model(model_path, device)[0] if model_path else None
        result = play(world, cfg, args.out, policy=policy, model=model, sample=getattr(args, "sample", False),
                      mode=args.render, quiet=args.quiet, solution=solution)
        if cmd == "check":
            assert result == replay(args.out / "trace.json", args.out, "ascii", quiet=True)
        print(json.dumps(result, ensure_ascii=False))
    elif cmd == "relations":
        models = model_set(args.checkpoint, device)
        report = relation_benchmark(cfg, models, held_out(cfg, args.episodes),
                                    [] if args.no_ood else held_out(cfg, args.episodes, ood=True),
                                    args.axis_repeats or cfg.data.axis_repeats)
        write_json(args.out, report)
        print(args.out)
    else:
        report = (ablations(args, cfg, device) if cmd == "ablate"
                  else evaluate(cfg, model_set(args.checkpoint, device), args.out, args.episodes,
                                repeats=args.axis_repeats, with_ood=not args.no_ood))
        print(json.dumps({split: {label: row["clear_rate"] for label, row in report[split].items()}
                          for split in ("id", "ood") if split in report}, ensure_ascii=False))


def main() -> None:
    parser = parser_for_cli()
    args = parser.parse_args()
    try:
        dispatch(args)
    except (ValueError, RuntimeError, FileNotFoundError) as error:
        parser.exit(2, f"maze-hgt: {error}\n")


if __name__ == "__main__":
    main()
