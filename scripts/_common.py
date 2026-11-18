"""Path bootstrap + shared helpers for the numbered scripts."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from onconet.data import load_manifest  # noqa: E402
from onconet.utils import Logger, ensure_dir, environment_info, load_config, load_json, save_json  # noqa: E402


def base_parser(desc: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--config", default=str(REPO / "configs" / "onconet.yaml"))
    return p


def setup(args, name: str):
    cfg = load_config(args.config)
    out = ensure_dir(Path(cfg["paths"]["output_dir"]))
    log = Logger(out / "logs" / f"{name}.log")
    log(f"===== {name} =====")
    log("environment:", environment_info())
    return cfg, out, log


def processed(cfg) -> Path:
    return Path(cfg["paths"]["processed_dir"])


def norm_stats(cfg, stain_norm=True):
    meta = load_json(processed(cfg) / "meta.json")
    s = meta["norm_stats"]["reinhard" if stain_norm else "raw"]
    return s["mean"], s["std"]


def class_weights(cfg):
    return load_json(processed(cfg) / "meta.json")["class_weights"]


def image_dir(cfg, stain_norm=True) -> Path:
    return processed(cfg) / ("images_256" if stain_norm else "images_256_raw")
