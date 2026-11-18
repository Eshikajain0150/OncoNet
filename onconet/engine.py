"""Training / inference loops shared by all scripts (Section 3.3)."""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset

from .data import ROIDataset, SegDataset, collate_views, imread_gray, mix_batch
from .losses import SegLoss, SupConLoss, weighted_soft_ce
from .metrics import boundary_f1, cls_metrics, seg_confusion, seg_image_metrics, summarise_seg
from .models import AEMLP, MultiResDeiT, build_resnet50, build_seg_model, build_vit_small
from .utils import EarlyStopping, WarmupCosine, current_lr, get_device, seed_worker, set_seed


def _loader(ds, bs, shuffle, cfg, seed):
    g = torch.Generator()
    g.manual_seed(seed)
    drop_last = shuffle and len(ds) % bs == 1  # avoid a BatchNorm batch of size one
    kw = dict(batch_size=bs, shuffle=shuffle, num_workers=cfg["runtime"]["num_workers"],
              worker_init_fn=seed_worker, generator=g, drop_last=drop_last)
    if isinstance(ds, ROIDataset):
        kw["collate_fn"] = collate_views
    return DataLoader(ds, **kw)


# =========================================================================== segmentation
def train_segmentation(cfg, variant, train_df, val_df, img_dir, mask_dir, mean, std,
                       out_dir: Path, log, seed=None):
    seed = cfg["seed"] if seed is None else seed
    set_seed(seed, cfg["runtime"]["deterministic"])
    s, dev = cfg["segmentation"], get_device()
    model = build_seg_model(variant, s["channels"]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=s["lr"], betas=tuple(s["betas"]),
                           weight_decay=s["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=s["plateau_factor"],
                                                       patience=s["plateau_patience"], min_lr=s["min_lr"])
    crit = SegLoss(s["loss_dice_weight"], s["loss_bce_weight"])
    dl_tr = _loader(SegDataset(train_df, img_dir, mask_dir, cfg, True, mean, std), s["batch_size"], True, cfg, seed)
    stopper = EarlyStopping(s["early_stop_patience"], "max")
    hist, best_path = [], Path(out_dir) / "best.pt"
    micro = _probe_micro_batch(model, crit, s["batch_size"], dev, log, variant)
    for epoch in range(1, s["max_epochs"] + 1):
        model.train()
        tl, n = 0.0, 0
        for x, y, _ in dl_tr:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad(set_to_none=True)
            for xs, ys in zip(x.split(micro), y.split(micro)):  # gradient accumulation if micro < batch
                loss = crit(model(xs), ys) * len(xs) / len(x)
                loss.backward()
                tl += loss.item() * len(x)
            opt.step()
            n += len(x)
        val = evaluate_segmentation(model, val_df, img_dir, mask_dir, cfg, mean, std, loss_fn=crit)
        sched.step(val["summary"]["dice"])
        improved = stopper.improved(val["summary"]["dice"])
        if improved:
            torch.save({"variant": variant, "state_dict": model.state_dict(), "epoch": epoch}, best_path)
        row = {"epoch": epoch, "train_loss": tl / n, "val_loss": val["loss"], "lr": current_lr(opt), "micro_batch": micro,
               **{f"val_{k}": v for k, v in val["summary"].items()}, "best": improved}
        hist.append(row)
        log(f"[seg:{variant}] ep {epoch:02d} train_loss {row['train_loss']:.4f} val_loss {row['val_loss']:.4f} "
            f"val_dice {row['val_dice']:.4f} val_iou {row['val_iou']:.4f} lr {row['lr']:.2e}{' *' if improved else ''}")
        if stopper.should_stop:
            log(f"[seg:{variant}] early stopping at epoch {epoch}")
            break
    pd.DataFrame(hist).to_csv(Path(out_dir) / "history.csv", index=False)
    ck = torch.load(best_path, map_location=dev)
    model.load_state_dict(ck["state_dict"])
    return model, hist, ck["epoch"]


def _probe_micro_batch(model, crit, batch_size, dev, log, tag):
    """Largest physical batch (<= batch_size) whose forward/backward fits in GPU memory.
    If it is smaller than batch_size, gradients are accumulated so the optimiser still sees
    the configured batch; the choice is logged. Model weights are left untouched."""
    if dev.type != "cuda":
        return batch_size
    import copy as _copy
    state = _copy.deepcopy(model.state_dict())
    mb = batch_size
    while mb >= 1:
        try:
            x = torch.randn(mb, 3, 256, 256, device=dev)
            y = (torch.rand(mb, 1, 256, 256, device=dev) > 0.5).float()
            model.train()
            crit(model(x), y).backward()
            model.zero_grad(set_to_none=True)
            del x, y
            torch.cuda.empty_cache()
            break
        except torch.cuda.OutOfMemoryError:
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            mb //= 2
    model.load_state_dict(state)  # undo BatchNorm running-stat updates from the probe
    if mb < batch_size:
        log(f"[seg:{tag}] GPU memory: physical batch {mb} x {batch_size // mb} accumulation steps "
            f"(effective batch {batch_size})")
    return max(mb, 1)


@torch.no_grad()
def predict_segmentation(model, df, img_dir, cfg, mean, std, batch_size=8):
    """Probability maps {image_id: HxW float32} for un-augmented images."""
    dev = get_device()
    model.eval()
    ds = SegDataset(df, img_dir, None, cfg, False, mean, std)
    out = {}
    for s in range(0, len(ds), batch_size):
        items = [ds.image_tensor(i) for i in range(s, min(s + batch_size, len(ds)))]
        x = torch.stack([t for t, _ in items]).to(dev)
        p = torch.sigmoid(model(x))[:, 0].float().cpu().numpy()
        for (_, iid), pm in zip(items, p):
            out[iid] = pm
    return out


def evaluate_segmentation(model, df, img_dir, mask_dir, cfg, mean, std, loss_fn=None, probs=None):
    thr = cfg["segmentation"]["threshold"]
    dev = get_device()
    if probs is None:
        probs = predict_segmentation(model, df, img_dir, cfg, mean, std)
    per, losses, pooled = [], [], np.zeros(4)
    for iid in df.image_id:
        gt = imread_gray(Path(mask_dir) / f"{iid}.png") > 127
        pr = probs[iid]
        pred = pr >= thr
        m = seg_image_metrics(pred, gt)
        m["boundary_f1"] = boundary_f1(pred, gt)
        m["image_id"] = iid
        per.append(m)
        pooled += np.array(seg_confusion(pred, gt), dtype=float)
        if loss_fn is not None:
            p = torch.from_numpy(np.clip(pr, 1e-6, 1 - 1e-6))[None, None]
            logit = torch.log(p / (1 - p))
            losses.append(loss_fn(logit, torch.from_numpy(gt.astype(np.float32))[None, None]).item())
    numeric = [{k: v for k, v in d.items() if k != "image_id"} for d in per]
    tp, fp, fn, tn = pooled
    pooled_m = {"dice": 2 * tp / (2 * tp + fp + fn), "iou": tp / (tp + fp + fn),
                "precision": tp / (tp + fp), "recall": tp / (tp + fn),
                "specificity": tn / (tn + fp), "accuracy": (tp + tn) / pooled.sum()}
    return {"summary": summarise_seg(numeric), "pooled": pooled_m, "per_image": per,
            "loss": float(np.mean(losses)) if losses else float("nan"), "probs": probs}


# =========================================================================== image classifiers
def _forward(kind, model, views):
    if kind == "deit_mr":
        logits, z = model(views[0], views[1])
        return logits, z
    return model(views[0]), None


def build_classifier(cfg, kind, num_classes):
    if kind == "deit_mr":
        d = cfg["deit"]
        return MultiResDeiT(d["backbone"], d["pretrained"], num_classes, d["proj_dim"])
    if kind == "resnet50":
        return build_resnet50(num_classes, pretrained=cfg["deit"]["pretrained"])
    if kind == "vit_small":
        return build_vit_small(cfg["baselines"]["vit_small"]["backbone"], num_classes,
                               pretrained=cfg["deit"]["pretrained"])
    raise ValueError(kind)


def train_image_classifier(cfg, kind, train_df, val_df, roi_dir, mean, std, class_weights,
                           out_dir: Path, log, seed=None):
    """kind: 'deit_mr' (proposed DeiT stage, WCE + 0.1 SupCon, two views),
    'resnet50' or 'vit_small' (baselines, WCE only, global view)."""
    seed = cfg["seed"] if seed is None else seed
    set_seed(seed, cfg["runtime"]["deterministic"])
    dev = get_device()
    names = cfg["dataset"]["class_names"]
    C = len(names)
    hp = cfg["deit"] if kind == "deit_mr" else cfg["baselines"][kind]
    views = ("local", "global") if kind == "deit_mr" else ("global",)
    model = build_classifier(cfg, kind, C).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"],
                            betas=tuple(hp.get("betas", (0.9, 0.999))))
    cosine = "warmup_epochs" in hp
    sched = (WarmupCosine(opt, hp["lr"], hp["warmup_epochs"], hp["max_epochs"], hp["min_lr"]) if cosine else
             torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=hp["plateau_factor"],
                                                        patience=hp["plateau_patience"]))
    supcon = SupConLoss(cfg["deit"]["supcon_temperature"]) if kind == "deit_mr" else None
    w = torch.tensor(class_weights, dtype=torch.float32, device=dev)
    dl_tr = _loader(ROIDataset(train_df, roi_dir, cfg, True, mean, std, views), hp["batch_size"], True, cfg, seed)
    rng = np.random.default_rng(seed)
    stopper = EarlyStopping(hp["early_stop_patience"], "max")
    hist, best_path, best_key = [], Path(out_dir) / "best.pt", None
    for epoch in range(1, hp["max_epochs"] + 1):
        if cosine:
            sched.step(epoch - 1)
        model.train()
        tl, correct, n = 0.0, 0, 0
        for vs, y, _ in dl_tr:
            vs, y = [v.to(dev) for v in vs], y.to(dev)
            onehot = F.one_hot(y, C).float()
            vs, target, mixed = mix_batch(vs, onehot, cfg, rng)
            opt.zero_grad(set_to_none=True)
            logits, z = _forward(kind, model, vs)
            loss = weighted_soft_ce(logits, target, w)
            if supcon is not None and mixed is None:
                loss = loss + cfg["deit"]["supcon_weight"] * supcon(model.proj(z), y)
            loss.backward()
            if hp.get("grad_clip"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), hp["grad_clip"])
            opt.step()
            tl += loss.item() * len(y)
            correct += (logits.argmax(1) == target.argmax(1)).sum().item()
            n += len(y)
        prob, yv, _ = predict_image_classifier(kind, model, val_df, roi_dir, cfg, mean, std)
        m = cls_metrics(yv, prob, names)
        vloss = weighted_soft_ce(torch.log(torch.from_numpy(prob).clamp_min(1e-8)),
                                 F.one_hot(torch.from_numpy(yv), C).float(), w.cpu()).item()
        if not cosine:
            sched.step(m["f1_macro"])
        key = (round(m["f1_macro"], 6), -vloss)
        improved = best_key is None or key > best_key
        stopper.improved(m["f1_macro"])
        if improved:
            best_key = key
            torch.save({"kind": kind, "state_dict": model.state_dict(), "epoch": epoch}, best_path)
        row = {"epoch": epoch, "train_loss": tl / n, "train_acc": correct / n, "val_loss": vloss,
               "val_acc": m["accuracy"], "val_precision": m["precision_macro"], "val_recall": m["recall_macro"],
               "val_f1": m["f1_macro"], "val_auc": m["auc_macro_ovr"], "lr": current_lr(opt), "best": improved}
        if kind == "deit_mr":
            a = model.alphas.detach().cpu().numpy()
            row["alpha_local"], row["alpha_global"] = float(a[0]), float(a[1])
        hist.append(row)
        log(f"[{kind}] ep {epoch:02d} loss {row['train_loss']:.4f} train_acc {row['train_acc']:.3f} "
            f"val_acc {row['val_acc']:.3f} val_f1 {row['val_f1']:.3f} val_auc {row['val_auc']:.3f} "
            f"lr {row['lr']:.2e}{' *' if improved else ''}")
        if stopper.should_stop:
            log(f"[{kind}] early stopping at epoch {epoch}")
            break
    pd.DataFrame(hist).to_csv(Path(out_dir) / "history.csv", index=False)
    ck = torch.load(best_path, map_location=dev)
    model.load_state_dict(ck["state_dict"])
    return model, hist, ck["epoch"]


@torch.no_grad()
def predict_image_classifier(kind, model, df, roi_dir, cfg, mean, std, batch_size=16):
    dev = get_device()
    model.eval()
    views = ("local", "global") if kind == "deit_mr" else ("global",)
    dl = DataLoader(ROIDataset(df, roi_dir, cfg, False, mean, std, views), batch_size=batch_size,
                    shuffle=False, num_workers=0, collate_fn=collate_views)
    P, Y, I = [], [], []
    for vs, y, ids in dl:
        logits, _ = _forward(kind, model, [v.to(dev) for v in vs])
        P.append(torch.softmax(logits, 1).float().cpu().numpy())
        Y.append(y.numpy())
        I += ids
    return np.concatenate(P), np.concatenate(Y), I


@torch.no_grad()
def extract_embeddings(model: MultiResDeiT, df, roi_dir, cfg, mean, std, train_views=0, seed=42):
    """768-d embeddings. ``train_views`` > 0 adds that many augmented views per image."""
    dev = get_device()
    model.eval()
    out = {"X": [], "y": [], "id": [], "view": []}
    passes = [(False, 0)] + [(True, k + 1) for k in range(train_views)]
    for aug, vidx in passes:
        torch.manual_seed(seed + vidx)
        dl = DataLoader(ROIDataset(df, roi_dir, cfg, aug, mean, std), batch_size=16, shuffle=False,
                        num_workers=0, collate_fn=collate_views)
        for vs, y, ids in dl:
            z = model.embed(vs[0].to(dev), vs[1].to(dev))
            out["X"].append(z.float().cpu().numpy())
            out["y"].append(y.numpy())
            out["id"] += ids
            out["view"] += [vidx] * len(ids)
    return {"X": np.concatenate(out["X"]), "y": np.concatenate(out["y"]),
            "id": np.array(out["id"]), "view": np.array(out["view"])}


# =========================================================================== AE-MLP
def train_aemlp(cfg, Xtr, ytr, Xva, yva, class_weights, log, out_path: Path | None = None,
                seed=None, init_state=None, sample_weights=None, lr=None, max_epochs=None,
                early_stop=True, select_including_init=False, tag="aemlp"):
    seed = cfg["seed"] if seed is None else seed
    set_seed(seed, cfg["runtime"]["deterministic"])
    a, dev = cfg["aemlp"], get_device()
    names = cfg["dataset"]["class_names"]
    C = len(names)
    model = AEMLP(Xtr.shape[1], tuple(a["hidden"]), C, a["dropout"]).to(dev)
    if init_state is not None:
        model.load_state_dict(init_state)
    lr = a["lr"] if lr is None else lr
    epochs = a["max_epochs"] if max_epochs is None else max_epochs
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=a["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=a["plateau_factor"],
                                                       patience=a["plateau_patience"])
    w = torch.tensor(class_weights, dtype=torch.float32, device=dev)
    sw = torch.ones(len(ytr)) if sample_weights is None else torch.as_tensor(sample_weights, dtype=torch.float32)
    ds = TensorDataset(torch.as_tensor(Xtr, dtype=torch.float32), torch.as_tensor(ytr, dtype=torch.long), sw)
    g = torch.Generator()
    g.manual_seed(seed)
    dl = DataLoader(ds, batch_size=a["batch_size"], shuffle=True, generator=g,
                    drop_last=len(ds) % a["batch_size"] == 1)
    stopper = EarlyStopping(a["early_stop_patience"], "max")
    hist, best_state, best_key, best_epoch = [], None, None, 0

    def _val():
        prob = predict_aemlp(model, Xva)
        m = cls_metrics(yva, prob, names)
        vloss = weighted_soft_ce(torch.log(torch.from_numpy(prob).clamp_min(1e-8)),
                                 F.one_hot(torch.as_tensor(yva), C).float(), w.cpu()).item()
        return m, vloss

    if select_including_init:
        m, vloss = _val()
        best_key, best_state, best_epoch = (round(m["f1_macro"], 6), -vloss), copy.deepcopy(model.state_dict()), 0
    for epoch in range(1, epochs + 1):
        model.train()
        tl, correct, n = 0.0, 0, 0
        for x, y, s in dl:
            x, y, s = x.to(dev), y.to(dev), s.to(dev)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = weighted_soft_ce(logits, F.one_hot(y, C).float(), w, s)
            loss.backward()
            opt.step()
            tl += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            n += len(y)
        m, vloss = _val()
        sched.step(m["f1_macro"])
        key = (round(m["f1_macro"], 6), -vloss)
        improved = best_key is None or key > best_key
        if improved:
            best_key, best_state, best_epoch = key, copy.deepcopy(model.state_dict()), epoch
        stopper.improved(m["f1_macro"])
        row = {"epoch": epoch, "train_loss": tl / n, "train_acc": correct / n, "val_loss": vloss,
               "val_acc": m["accuracy"], "val_precision": m["precision_macro"], "val_recall": m["recall_macro"],
               "val_f1": m["f1_macro"], "val_auc": m["auc_macro_ovr"], "lr": current_lr(opt), "best": improved}
        hist.append(row)
        if log is not None:
            log(f"[{tag}] ep {epoch:02d} loss {row['train_loss']:.4f} train_acc {row['train_acc']:.3f} "
                f"val_acc {row['val_acc']:.3f} val_f1 {row['val_f1']:.3f} val_auc {row['val_auc']:.3f}"
                f"{' *' if improved else ''}")
        if early_stop and stopper.should_stop:
            if log is not None:
                log(f"[{tag}] early stopping at epoch {epoch}")
            break
    model.load_state_dict(best_state)
    if out_path is not None:
        torch.save({"state_dict": best_state, "epoch": best_epoch, "in_dim": Xtr.shape[1]}, out_path)
    return model, hist, best_epoch


@torch.no_grad()
def predict_aemlp(model, X, batch_size=256):
    dev = get_device()
    model.eval()
    X = torch.as_tensor(X, dtype=torch.float32)
    out = [torch.softmax(model(X[i:i + batch_size].to(dev)), 1).cpu().numpy() for i in range(0, len(X), batch_size)]
    return np.concatenate(out)


def error_guided_finetune(cfg, model, emb_tr, Xva, yva, class_weights, base_epoch, log,
                          eval_sets: dict | None = None):
    """Error-analysis feedback loop (Section 4.8): in each iteration hard training images
    are mined with out-of-fold AE-MLP predictions *inside the training partition*
    (misclassified or margin < tau), then the current AE-MLP is fine-tuned with those
    images up-weighted. Checkpoints are chosen on validation macro-F1 only; ``eval_sets``
    are evaluated for reporting and never influence any choice."""
    e = cfg["error_guided"]
    names = cfg["dataset"]["class_names"]
    X, y, ids = emb_tr["X"], emb_tr["y"], emb_tr["id"]
    clean = emb_tr["view"] == 0
    uid, uy = ids[clean], y[clean]
    clean_row = {u: i for u, i in zip(ids[clean], np.where(clean)[0])}
    log_rows = []

    def _report(it, n_hard, hard_ids):
        row = {"iteration": it, "n_hard": n_hard, "hard_ids": hard_ids}
        row["val"] = cls_metrics(yva, predict_aemlp(model, Xva), names)
        for k, (Xe, ye) in (eval_sets or {}).items():
            row[k] = cls_metrics(ye, predict_aemlp(model, Xe), names)
        log_rows.append(row)
        msg = f"[egft] iter {it}: hard={n_hard} val_err={row['val']['misclassification_rate']:.4f}"
        for k in (eval_sets or {}):
            msg += f" {k}_err={row[k]['misclassification_rate']:.4f}"
        log(msg)

    _report(0, 0, [])
    for it in range(1, e["iterations"] + 1):
        skf = StratifiedKFold(n_splits=e["oof_folds"], shuffle=True, random_state=cfg["seed"] + it)
        oof = np.zeros((len(uid), len(names)))
        for k, (tr_u, te_u) in enumerate(skf.split(uid, uy)):
            tr_mask = np.isin(ids, uid[tr_u])
            m_k, _, _ = train_aemlp(cfg, X[tr_mask], y[tr_mask], Xva, yva, class_weights, None,
                                    seed=cfg["seed"] + 100 * it + k, max_epochs=max(base_epoch, 5),
                                    early_stop=False, tag=f"oof{it}.{k}")
            rows = [clean_row[u] for u in uid[te_u]]
            oof[te_u] = predict_aemlp(m_k, X[rows])
        p_true = oof[np.arange(len(uy)), uy]
        other = oof.copy()
        other[np.arange(len(uy)), uy] = -1
        margin = p_true - other.max(1)
        hard = (oof.argmax(1) != uy) | (margin < e["margin"])
        hard_ids = set(uid[hard].tolist())
        sw = np.where(np.isin(ids, list(hard_ids)), e["hard_weight"], 1.0)
        model, _, _ = train_aemlp(cfg, X, y, Xva, yva, class_weights, log, seed=cfg["seed"] + it,
                                  init_state=model.state_dict(), sample_weights=sw, lr=e["lr"],
                                  max_epochs=e["epochs"], early_stop=False, select_including_init=True,
                                  tag=f"egft{it}")
        _report(it, int(hard.sum()), sorted(hard_ids))
    return model, log_rows
