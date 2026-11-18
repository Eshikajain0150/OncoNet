"""Step 5 - proposed classification stages on the primary hold-out split:
(a) multi-resolution DeiT-Small/16 fine-tuning (weighted CE + 0.1 SupCon),
(b) AE-MLP on the 768-d concatenated embeddings,
(c) error-guided fine-tuning (3 feedback iterations),
(d) final held-out test evaluation with bootstrap 95% CIs.
Requires: 03 (unetpp_ag_msf) and 04 (--seg-variant unetpp_ag_msf).

    python scripts/05_train_proposed.py              # all stages
    python scripts/05_train_proposed.py --skip-deit  # reuse saved DeiT embeddings
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from _common import base_parser, class_weights, ensure_dir, load_manifest, norm_stats, processed, save_json, setup
from onconet.data import split_frames
from onconet.engine import predict_aemlp, predict_image_classifier
from onconet.metrics import bootstrap_ci, cls_metrics
from onconet.models import AEMLP
from onconet.stages import aemlp_stage, deit_stage, egft_stage, load_embeddings
from onconet.utils import get_device


def main():
    p = base_parser(__doc__)
    p.add_argument("--skip-deit", action="store_true")
    args = p.parse_args()
    cfg, out, log = setup(args, "05_train_proposed")
    names = cfg["dataset"]["class_names"]
    df = load_manifest(cfg)
    tr, va, te = split_frames(df)
    mean, std = norm_stats(cfg)
    cw = class_weights(cfg)
    log("class weights (training partition):", dict(zip(names, [round(w, 3) for w in cw])))
    roi_dir = processed(cfg) / "rois" / "unetpp_ag_msf"
    pdir = ensure_dir(out / "proposed")

    if args.skip_deit:
        emb = load_embeddings(pdir / "deit" / "embeddings.npz")
        base_epoch = None
    else:
        deit, emb, _ = deit_stage(cfg, tr, va, {"test": te}, roi_dir, mean, std, cw, pdir / "deit", log)
        prob, y, _ = predict_image_classifier("deit_mr", deit, te, roi_dir, cfg, mean, std)
        save_json(cls_metrics(y, prob, names), pdir / "deit" / "deit_head_test_metrics.json")
        del deit
        torch.cuda.empty_cache()

    model, base_epoch = aemlp_stage(cfg, emb, cw, pdir, log)
    model, rows = egft_stage(cfg, model, emb, cw, base_epoch, pdir, log)

    prob = predict_aemlp(model, emb["test"]["X"])
    y = emb["test"]["y"]
    m = cls_metrics(y, prob, names)
    acc = lambda yt, pr: float((pr.argmax(1) == yt).mean())  # noqa: E731
    from sklearn.metrics import f1_score, roc_auc_score
    m["bootstrap_95ci"] = {
        "accuracy": bootstrap_ci(y, prob, acc),
        "f1_macro": bootstrap_ci(y, prob, lambda a, b: f1_score(a, b.argmax(1), average="macro")),
        "auc_macro_ovr": bootstrap_ci(y, prob, lambda a, b: roc_auc_score(a, b, multi_class="ovr")),
    }
    m["misclassification_before_egft"] = rows[0]["test"]["misclassification_rate"]
    m["misclassification_after_egft"] = rows[-1]["test"]["misclassification_rate"]
    save_json(m, pdir / "test_metrics.json")
    pd.DataFrame({"image_id": emb["test"]["id"], "y_true": y, "y_pred": prob.argmax(1),
                  **{f"p_{c}": prob[:, i] for i, c in enumerate(names)}}).to_csv(pdir / "test_predictions.csv", index=False)
    log(f"[proposed] TEST acc {m['accuracy']:.4f}  macroF1 {m['f1_macro']:.4f}  AUC {m['auc_macro_ovr']:.4f}  "
        f"blast-sens(exact) {m['blast_sensitivity_exact']:.4f}  "
        + "  ".join(f"{c}-sens {m['per_class'][c]['sensitivity']:.4f}" for c in names))
    log(f"[proposed] misclassification before/after error-guided FT: "
        f"{m['misclassification_before_egft']:.4f} -> {m['misclassification_after_egft']:.4f}")


if __name__ == "__main__":
    main()
