"""Classification stages of the proposed framework, shared by the hold-out scripts and the
cross-validation driver: DeiT fine-tuning -> embeddings -> AE-MLP -> error-guided FT."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .engine import (error_guided_finetune, extract_embeddings, predict_aemlp, train_aemlp,
                     train_image_classifier)
from .metrics import cls_metrics
from .utils import ensure_dir, save_json


def deit_stage(cfg, tr, va, eval_sets: dict, roi_dir, mean, std, cw, out_dir, log):
    out_dir = ensure_dir(out_dir)
    model, hist, best_epoch = train_image_classifier(cfg, "deit_mr", tr, va, roi_dir, mean, std, cw, out_dir, log)
    emb = {"train": extract_embeddings(model, tr, roi_dir, cfg, mean, std,
                                       train_views=cfg["deit"]["train_embedding_views"], seed=cfg["seed"]),
           "val": extract_embeddings(model, va, roi_dir, cfg, mean, std)}
    for k, d in eval_sets.items():
        emb[k] = extract_embeddings(model, d, roi_dir, cfg, mean, std)
    np.savez_compressed(out_dir / "embeddings.npz",
                        **{f"{s}_{f}": v for s, e in emb.items() for f, v in e.items()})
    a = model.alphas.detach().cpu().numpy()
    save_json({"best_epoch": best_epoch, "alpha_local": float(a[0]), "alpha_global": float(a[1]),
               "embedding_dim": int(emb["val"]["X"].shape[1])}, out_dir / "deit_summary.json")
    return model, emb, best_epoch


def load_embeddings(path) -> dict:
    z = np.load(path, allow_pickle=True)
    emb = {}
    for key in z.files:
        s, f = key.split("_", 1)
        emb.setdefault(s, {})[f] = z[key]
    return emb


def aemlp_stage(cfg, emb, cw, out_dir, log, eval_keys=("test",)):
    out_dir = ensure_dir(out_dir)
    names = cfg["dataset"]["class_names"]
    model, hist, best_epoch = train_aemlp(cfg, emb["train"]["X"], emb["train"]["y"], emb["val"]["X"],
                                          emb["val"]["y"], cw, log, out_path=Path(out_dir) / "aemlp_base.pt")
    pd.DataFrame(hist).to_csv(Path(out_dir) / "aemlp_history.csv", index=False)
    res = {"best_epoch": best_epoch, "val": cls_metrics(emb["val"]["y"], predict_aemlp(model, emb["val"]["X"]), names)}
    for k in eval_keys:
        if k in emb:
            res[k] = cls_metrics(emb[k]["y"], predict_aemlp(model, emb[k]["X"]), names)
    save_json(res, Path(out_dir) / "aemlp_base_metrics.json")
    return model, best_epoch


def egft_stage(cfg, model, emb, cw, base_epoch, out_dir, log, eval_keys=("test",)):
    out_dir = ensure_dir(out_dir)
    eval_sets = {k: (emb[k]["X"], emb[k]["y"]) for k in eval_keys if k in emb}
    model, rows = error_guided_finetune(cfg, model, emb["train"], emb["val"]["X"], emb["val"]["y"],
                                        cw, base_epoch, log, eval_sets)
    torch.save({"state_dict": model.state_dict(), "in_dim": int(emb["val"]["X"].shape[1])},
               Path(out_dir) / "aemlp_final.pt")
    save_json(rows, Path(out_dir) / "error_guided_iterations.json")
    return model, rows
