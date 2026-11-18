"""Step 7 - five-fold stratified cross-validation of the proposed framework on the 425-image
development pool (350 train + 75 val). In fold k the framework is trained on the other four
folds (a stratified 15% inner split of them is used only for checkpoint selection) and
evaluated on fold k. The 75-image held-out test set is never touched. Fold-wise Dice, IoU,
accuracy and macro-F1 are summarised as mean, sample SD and Student-t 95% CI (Eq. 19).

    python scripts/07_cross_validation.py            # all folds (resumes finished folds)
    python scripts/07_cross_validation.py --folds 0 1
"""
from __future__ import annotations

import torch

from _common import (base_parser, ensure_dir, image_dir, load_json, load_manifest, norm_stats, processed,
                     save_json, setup)
from onconet.data import class_weights_from_counts, split_frames
from onconet.engine import evaluate_segmentation, predict_aemlp, predict_segmentation, train_segmentation
from onconet.metrics import cls_metrics, t_interval
from onconet.pipeline import write_rois
from onconet.stages import aemlp_stage, deit_stage, egft_stage


def run_fold(cfg, out, log, k):
    names = cfg["dataset"]["class_names"]
    df = load_manifest(cfg)
    tr, va, ev = split_frames(df, fold=k, seed=cfg["seed"])
    log(f"[cv] fold {k}: train {len(tr)}  inner-val {len(va)}  eval {len(ev)}")
    mean, std = norm_stats(cfg)
    cw = class_weights_from_counts([int((tr.label == c).sum()) for c in names])
    fdir = ensure_dir(out / "cv" / f"fold{k}")
    idir, mdir = image_dir(cfg), processed(cfg) / "masks_cell"

    seg, _, seg_epoch = train_segmentation(cfg, "unetpp_ag_msf", tr, va, idir, mdir, mean, std,
                                           ensure_dir(fdir / "seg"), log)
    seg_res = evaluate_segmentation(seg, ev, idir, mdir, cfg, mean, std)
    dev_df = df[df.split != "test"]
    probs = predict_segmentation(seg, dev_df, idir, cfg, mean, std)
    roi_dir = processed(cfg) / "cv" / f"fold{k}" / "rois"
    write_rois(probs, dev_df, idir, roi_dir, cfg)
    del seg
    torch.cuda.empty_cache()

    deit, emb, _ = deit_stage(cfg, tr, va, {"eval": ev}, roi_dir, mean, std, cw, fdir / "deit", log)
    del deit
    torch.cuda.empty_cache()
    model, base_epoch = aemlp_stage(cfg, emb, cw, fdir, log, eval_keys=("eval",))
    model, _ = egft_stage(cfg, model, emb, cw, base_epoch, fdir, log, eval_keys=("eval",))
    cm = cls_metrics(emb["eval"]["y"], predict_aemlp(model, emb["eval"]["X"]), names)
    res = {"fold": k, "seg_best_epoch": seg_epoch, "dice": seg_res["summary"]["dice"], "iou": seg_res["summary"]["iou"],
           "accuracy": cm["accuracy"], "f1_macro": cm["f1_macro"], "auc_macro_ovr": cm["auc_macro_ovr"],
           "seg_summary": seg_res["summary"], "cls_metrics": cm}
    save_json(res, fdir / "fold_result.json")
    log(f"[cv] fold {k}: dice {res['dice']:.4f} iou {res['iou']:.4f} acc {res['accuracy']:.4f} f1 {res['f1_macro']:.4f}")
    return res


def main():
    p = base_parser(__doc__)
    p.add_argument("--folds", type=int, nargs="*")
    args = p.parse_args()
    cfg, out, log = setup(args, "07_cross_validation")
    K = cfg["dataset"]["cv_folds"]
    for k in (args.folds if args.folds else range(K)):
        if (out / "cv" / f"fold{k}" / "fold_result.json").exists():
            log(f"[cv] fold {k} already finished - skipping (delete fold_result.json to rerun)")
            continue
        run_fold(cfg, out, log, k)
    done = [load_json(out / "cv" / f"fold{k}" / "fold_result.json") for k in range(K)
            if (out / "cv" / f"fold{k}" / "fold_result.json").exists()]
    if len(done) == K:
        summ = {m: t_interval([r[m] for r in done]) for m in ("dice", "iou", "accuracy", "f1_macro", "auc_macro_ovr")}
        summ["folds"] = [{m: r[m] for m in ("fold", "dice", "iou", "accuracy", "f1_macro", "auc_macro_ovr")} for r in done]
        save_json(summ, out / "cv" / "cv_summary.json")
        for m in ("dice", "iou", "accuracy", "f1_macro"):
            s = summ[m]
            log(f"[cv] {m}: {s['mean']:.4f} ± {s['sd']:.4f}  95% CI {s['ci_low']:.4f}-{s['ci_high']:.4f}")


if __name__ == "__main__":
    main()
