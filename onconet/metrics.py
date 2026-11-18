"""Evaluation metrics and statistics (Sections 3.4 and 3.4.2)."""
from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.ndimage import binary_erosion, distance_transform_edt
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)


# =========================================================================== segmentation
def seg_confusion(pred: np.ndarray, gt: np.ndarray):
    pred, gt = pred.astype(bool), gt.astype(bool)
    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    tn = np.logical_and(~pred, ~gt).sum()
    return tp, fp, fn, tn


def seg_image_metrics(pred, gt, eps=1e-7) -> dict:
    tp, fp, fn, tn = seg_confusion(pred, gt)
    return {
        "dice": (2 * tp + eps) / (2 * tp + fp + fn + eps),
        "iou": (tp + eps) / (tp + fp + fn + eps),
        "precision": (tp + eps) / (tp + fp + eps),
        "recall": (tp + eps) / (tp + fn + eps),          # sensitivity
        "specificity": (tn + eps) / (tn + fp + eps),
        "accuracy": (tp + tn) / (tp + tn + fp + fn),
    }


def boundary_f1(pred, gt, tol=2) -> float:
    """Boundary F1 (BF score, Csurka et al. 2013) with a ``tol``-pixel tolerance."""
    pred, gt = pred.astype(bool), gt.astype(bool)
    bp = pred & ~binary_erosion(pred)
    bg = gt & ~binary_erosion(gt)
    if bp.sum() == 0 and bg.sum() == 0:
        return 1.0
    if bp.sum() == 0 or bg.sum() == 0:
        return 0.0
    dg = distance_transform_edt(~bg)
    dp = distance_transform_edt(~bp)
    prec = (dg[bp] <= tol).mean()
    rec = (dp[bg] <= tol).mean()
    return 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)


def summarise_seg(per_image: list[dict]) -> dict:
    """Mean of per-image metrics (the reported segmentation values)."""
    keys = per_image[0].keys()
    return {k: float(np.mean([d[k] for d in per_image])) for k in keys}


# =========================================================================== classification
def cls_metrics(y_true, prob, class_names) -> dict:
    y_true = np.asarray(y_true)
    prob = np.asarray(prob)
    y_pred = prob.argmax(1)
    C = len(class_names)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(C)))
    out = {
        "n": int(len(y_true)),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "confusion_matrix": cm,
        "misclassification_rate": 1 - accuracy_score(y_true, y_pred),
    }
    try:
        out["auc_macro_ovr"] = roc_auc_score(y_true, prob, multi_class="ovr", average="macro")
    except ValueError:
        out["auc_macro_ovr"] = float("nan")
    per = {}
    for c, name in enumerate(class_names):
        tp = cm[c, c]
        fn = cm[c].sum() - tp
        fp = cm[:, c].sum() - tp
        tn = cm.sum() - tp - fn - fp
        per[name] = {
            "support": int(cm[c].sum()),
            "sensitivity": tp / max(tp + fn, 1),
            "specificity": tn / max(tn + fp, 1),
            "precision": tp / max(tp + fp, 1),
            "false_negative_rate": fn / max(tp + fn, 1),
            "false_positive_rate": fp / max(fp + tn, 1),
        }
    out["per_class"] = per
    # Blast-cell endpoints, reported under three explicit definitions:
    blast = [class_names.index(c) for c in ("AML", "ALL") if c in class_names]
    if blast:
        is_blast = np.isin(y_true, blast)
        out["blast_sensitivity_exact"] = float((y_pred[is_blast] == y_true[is_blast]).mean())
        out["blast_sensitivity_any_malignant"] = float(np.isin(y_pred[is_blast], blast).mean())
        minority = min(blast, key=lambda c: (y_true == c).sum())
        out["minority_blast_class"] = class_names[minority]
        out["minority_blast_sensitivity"] = per[class_names[minority]]["sensitivity"]
    if {"AML", "ALL"} <= set(class_names):
        a, l = class_names.index("AML"), class_names.index("ALL")
        sel = np.isin(y_true, [a, l])
        sub = prob[sel][:, [a, l]]
        pred_al = np.where(sub.argmax(1) == 0, a, l)
        out["aml_vs_all_accuracy"] = float((pred_al == y_true[sel]).mean())
    return out


def bootstrap_ci(y_true, prob, fn, n_boot=2000, seed=42, alpha=0.05):
    rng = np.random.default_rng(seed)
    y_true, prob = np.asarray(y_true), np.asarray(prob)
    vals = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            vals.append(fn(y_true[idx], prob[idx]))
        except ValueError:
            continue
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def t_interval(values, conf=0.95) -> dict:
    """Mean, sample SD and two-sided Student-t CI over folds (Eq. 19)."""
    v = np.asarray(values, dtype=float)
    n = len(v)
    m, s = v.mean(), v.std(ddof=1) if n > 1 else 0.0
    tcrit = stats.t.ppf(0.5 + conf / 2, n - 1) if n > 1 else float("nan")
    h = tcrit * s / np.sqrt(n) if n > 1 else float("nan")
    return {"mean": m, "sd": s, "ci_low": m - h, "ci_high": m + h, "t_crit": tcrit, "n": n}


def mcnemar_exact(correct_a, correct_b) -> dict:
    """Exact (binomial) McNemar test on paired per-image correctness."""
    a, b = np.asarray(correct_a, bool), np.asarray(correct_b, bool)
    n01 = int((a & ~b).sum())   # A right, B wrong
    n10 = int((~a & b).sum())   # A wrong, B right
    k, n = min(n01, n10), n01 + n10
    p = 1.0 if n == 0 else min(1.0, 2 * stats.binom.cdf(k, n, 0.5))
    return {"a_right_b_wrong": n01, "a_wrong_b_right": n10, "p_value": float(p)}
