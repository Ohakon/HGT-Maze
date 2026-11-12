"""Seeds, device setup, checkpoints and structured logs."""
from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
from random import seed as python_seed
import random

import torch

from .config import Config
from .model import Architecture, Network


def setup(cfg: Config) -> torch.device:
    device = cfg.runtime.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was selected and no CUDA device is available")
    if cfg.runtime.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(cfg.runtime.deterministic)
    torch.set_num_threads(cfg.runtime.threads)
    python_seed(cfg.runtime.seed)
    torch.manual_seed(cfg.runtime.seed)
    return torch.device(device)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".tmp")
    partial.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    partial.replace(path)


def append_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def checkpoint(path: Path, model: Network, optimizer: torch.optim.Optimizer, cfg: Config, **metadata) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".tmp")
    torch.save({"architecture": asdict(model.cfg), "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "config": cfg.plain(), "metadata": metadata,
                "rng": torch.get_rng_state(), "python_rng": random.getstate(),
                "cuda_rng": torch.cuda.get_rng_state_all() if model.device.type == "cuda" else []}, partial)
    partial.replace(path)


def read_checkpoint(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=True)


def load_model(path: Path, device: torch.device) -> tuple[Network, dict]:
    data = read_checkpoint(path)
    model = Network(Architecture(**data["architecture"])).to(device)
    model.load_state_dict(data["model"])
    return model, data


def restore_rng(checkpoint_data: dict, device: torch.device) -> None:
    torch.set_rng_state(checkpoint_data["rng"])
    random.setstate(checkpoint_data["python_rng"])
    if device.type == "cuda" and checkpoint_data["cuda_rng"]:
        torch.cuda.set_rng_state_all(checkpoint_data["cuda_rng"])
