"""Shared helpers: configuration, seeding, I/O, logging, schedulers."""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import copy
import json
import math
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- config
def load_config(path: str | os.PathLike | None = None) -> dict:
    path = Path(path) if path else REPO_ROOT / "configs" / "onconet.yaml"
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    for key in ("raw_dir", "processed_dir", "output_dir"):
        p = Path(cfg["paths"][key])
        cfg["paths"][key] = str(p if p.is_absolute() else (REPO_ROOT / p))
    cfg["_config_path"] = str(path)
    return cfg


def cfg_path(cfg: dict, key: str, *parts) -> Path:
    p = Path(cfg["paths"][key]).joinpath(*parts)
    return p


# --------------------------------------------------------------------------- seeding
def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int) -> None:  # DataLoader worker seeding
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- I/O
def ensure_dir(p) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _to_jsonable(o):
    if isinstance(o, dict):
        return {str(k): _to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_to_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    if isinstance(o, Path):
        return str(o)
    return o


def save_json(obj, path) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_to_jsonable(obj), fh, indent=2)


def load_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def environment_info() -> dict:
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    try:
        import torchvision, timm  # noqa: E401

        info["torchvision"] = torchvision.__version__
        info["timm"] = timm.__version__
    except Exception:  # pragma: no cover
        pass
    return info


class Logger:
    """Prints and appends to a log file so every run leaves a text trace."""

    def __init__(self, path):
        self.path = Path(path)
        ensure_dir(self.path.parent)

    def __call__(self, *msg):
        line = " ".join(str(m) for m in msg)
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(line, flush=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(f"[{stamp}] {line}\n")


# --------------------------------------------------------------------------- schedules
class WarmupCosine:
    """Per-epoch linear warm-up followed by cosine decay to ``min_lr`` (Eq. 13)."""

    def __init__(self, optimizer, base_lr, warmup_epochs, max_epochs, min_lr):
        self.opt = optimizer
        self.base_lr = base_lr
        self.warmup = warmup_epochs
        self.total = max_epochs
        self.min_lr = min_lr
        self.step(0)

    def lr_at(self, epoch: int) -> float:
        if epoch < self.warmup:
            return self.base_lr * (epoch + 1) / self.warmup
        t = epoch - self.warmup
        T = max(1, self.total - self.warmup)
        return self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * t / T))

    def step(self, epoch: int) -> float:
        lr = self.lr_at(epoch)
        for g in self.opt.param_groups:
            g["lr"] = lr
        return lr


def current_lr(optimizer) -> float:
    return optimizer.param_groups[0]["lr"]


class EarlyStopping:
    def __init__(self, patience: int, mode: str = "max"):
        self.patience = patience
        self.mode = mode
        self.best = None
        self.bad = 0

    def improved(self, value: float) -> bool:
        if self.best is None:
            self.best = value
            self.bad = 0
            return True
        better = value > self.best if self.mode == "max" else value < self.best
        if better:
            self.best = value
            self.bad = 0
            return True
        self.bad += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.bad >= self.patience


def deep_update(base: dict, upd: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out
