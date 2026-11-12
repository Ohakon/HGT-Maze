"""TOML configuration for generation, supervised learning and RL."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TypeVar
import tomllib

from .game import Reward
from .generate import Generation
from .model import Architecture
from .solver import Search


@dataclass(frozen=True)
class Runtime:
    seed: int = 1729
    device: str = "auto"
    threads: int = 2
    deterministic: bool = True


@dataclass(frozen=True)
class Data:
    relation_train: int = 192
    relation_val: int = 64
    queries_per_class: int = 12
    train_worlds: int = 32
    val_worlds: int = 12
    validation_rollouts: int = 8
    explorations: int = 1
    explore_turns: int = 8
    label_stride: int = 3
    negative_every: int = 4
    axis_repeats: int = 24


@dataclass(frozen=True)
class Optimizer:
    lr: float = 0.001
    graph_batch: int = 8
    sequence_batch: int = 4
    bptt: int = 12
    grad_clip: float = 1.0
    gamma: float = 0.99
    approximate_weight: float = 0.35


@dataclass(frozen=True)
class Stage:
    epochs: int = 16
    patience: int = 5
    stable: int = 2
    min_delta: float = 0.002
    max_seconds: float = 0.
    freeze_epochs: int = 0
    freeze_modules: tuple[str, ...] = ()
    score: str = "loss"
    goals: dict[str, dict[str, float]] = field(default_factory=dict)


def stages() -> dict[int, Stage]:
    return {
        1: Stage(score="macro_f1", goals={"macro_f1": {"min": .78}, "logic.f1": {"min": .78},
                                          "hop.3.accuracy": {"min": .70}}),
        2: Stage(score="event_f1", freeze_epochs=1, freeze_modules=("hgt",),
                 goals={"birth.f1": {"min": .65}, "remove.f1": {"min": .75},
                        "edge.add.f1": {"min": .70}, "edge.remove.f1": {"min": .70},
                        "attributes.mae": {"max": .08}}),
        3: Stage(score="pool_mae", freeze_epochs=1, freeze_modules=("hgt", "events", "gru", "memory_norm"),
                 goals={"pool_mae": {"max": .75}, "solvable.f1": {"min": .75},
                        "solvable.balanced_accuracy": {"min": .75}}),
        4: Stage(score="value_exact.mae", freeze_epochs=1, freeze_modules=("hgt", "events", "gru", "memory_norm", "pool"),
                 goals={"value_exact.mae": {"max": .12}}),
        5: Stage(score="optimal.accuracy", freeze_epochs=1,
                 freeze_modules=("hgt", "events", "gru", "memory_norm", "pool"),
                 goals={"optimal.accuracy": {"min": .80}, "rollout.clear_rate": {"min": .65}}),
    }


@dataclass(frozen=True)
class Reinforcement:
    updates: int = 100
    environments: int = 12
    ppo_epochs: int = 4
    learning_rate: float = 0.0003
    sequence_batch: int = 8
    unroll: int = 16
    gae_lambda: float = 0.95
    clip: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    target_kl: float = 0.04
    group_size: int = 4
    group_worlds: int = 3
    grpo_kl: float = 0.02
    grpo_value_coef: float = 0.0
    curriculum_levels: int = 3
    promote_rate: float = 0.7
    promote_window: int = 4
    validate_every: int = 5
    validation_worlds: int = 8


@dataclass(frozen=True)
class Config:
    runtime: Runtime = Runtime()
    generation: Generation = Generation()
    search: Search = Search()
    model: Architecture = Architecture()
    data: Data = Data()
    optimizer: Optimizer = Optimizer()
    reward: Reward = Reward()
    rl: Reinforcement = Reinforcement()
    stages: dict[int, Stage] = field(default_factory=stages)

    def __post_init__(self) -> None:
        if not 0 < self.optimizer.gamma <= 1.:
            raise ValueError("Discount must be in (0, 1]")
        if not 0 < self.optimizer.approximate_weight <= 1.:
            raise ValueError("Approximate label weight must be in (0, 1]")
        if min(self.runtime.threads, self.optimizer.graph_batch, self.optimizer.sequence_batch,
               self.optimizer.bptt, self.data.label_stride, self.data.relation_train,
               self.data.relation_val, self.data.train_worlds, self.data.val_worlds,
               self.data.validation_rollouts, self.data.queries_per_class, self.data.axis_repeats) < 1:
            raise ValueError("Thread, sample, batch and sequence counts must be positive")
        if min(self.rl.updates, self.rl.environments, self.rl.ppo_epochs, self.rl.sequence_batch,
               self.rl.unroll, self.rl.group_worlds, self.rl.promote_window, self.rl.validate_every,
               self.rl.validation_worlds) < 1 or self.rl.group_size < 2:
            raise ValueError("Invalid rollout, group or update counts")
        if not 0 <= self.rl.gae_lambda <= 1 or not 0 < self.rl.clip < 1:
            raise ValueError("Invalid GAE lambda or clipping range")
        for stage in self.stages.values():
            if min(stage.epochs, stage.patience, stage.stable) < 1 or stage.freeze_epochs < 0:
                raise ValueError("Invalid stage limits")
            if any(not g or set(g) - {"min", "max"} for g in stage.goals.values()):
                raise ValueError("Goals must specify min and/or max")

    def plain(self) -> dict:
        return asdict(self)


T = TypeVar("T")


def configure(base: T, values: dict) -> T:
    raw = asdict(base)
    unknown = values.keys() - raw.keys()
    if unknown:
        raise ValueError(f"Unknown keys for {type(base).__name__}: {sorted(unknown)}")
    raw.update((k, tuple(v) if isinstance(raw[k], tuple) else v) for k, v in values.items())
    return type(base)(**raw)


def load(path: Path | None) -> Config:
    if path is None:
        return Config()
    with path.open("rb") as file:
        values = tomllib.load(file)
    cfg = Config()
    unknown = values.keys() - cfg.plain().keys()
    if unknown:
        raise ValueError(f"Unknown configuration sections: {sorted(unknown)}")
    sections = {name: configure(getattr(cfg, name), raw) for name, raw in values.items() if name != "stages"}
    custom = dict(cfg.stages)
    for key, stage in values.get("stages", {}).items():
        number = int(key)
        if number not in custom:
            raise ValueError(f"Unknown stage {number}")
        custom[number] = configure(custom[number], stage)
    return Config(**sections, stages=custom)
