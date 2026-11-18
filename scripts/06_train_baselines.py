"""Step 6 - end-to-end baselines of Section 3.3.5 / Table 7 on the primary hold-out split:
U-Net + ResNet-50 (CNN-based) and U-Net + ViT-Small/16 (transformer-based). Both use the
U-Net ROIs, the same class weights, preprocessing, augmentation and checkpoint rule.
Requires: 03 --variant unet and 04 --seg-variant unet.

    python scripts/06_train_baselines.py --model resnet50
    python scripts/06_train_baselines.py --model vit_small
"""
from __future__ import annotations

import pandas as pd

from _common import base_parser, class_weights, ensure_dir, load_manifest, norm_stats, processed, save_json, setup
from onconet.data import split_frames
from onconet.engine import predict_image_classifier, train_image_classifier
from onconet.metrics import bootstrap_ci, cls_metrics
from onconet.models import count_params


def main():
    p = base_parser(__doc__)
    p.add_argument("--model", choices=["resnet50", "vit_small"], required=True)
    args = p.parse_args()
    cfg, out, log = setup(args, f"06_baseline_{args.model}")
    names = cfg["dataset"]["class_names"]
    df = load_manifest(cfg)
    tr, va, te = split_frames(df)
    mean, std = norm_stats(cfg)
    cw = class_weights(cfg)
    roi_dir = processed(cfg) / "rois" / "unet"
    bdir = ensure_dir(out / "baselines" / args.model)
    model, hist, best_epoch = train_image_classifier(cfg, args.model, tr, va, roi_dir, mean, std, cw, bdir, log)
    prob, y, ids = predict_image_classifier(args.model, model, te, roi_dir, cfg, mean, std)
    m = cls_metrics(y, prob, names)
    m["best_epoch"] = best_epoch
    m["n_params_classifier"] = count_params(model)
    m["bootstrap_95ci"] = {"accuracy": bootstrap_ci(y, prob, lambda a, b: float((b.argmax(1) == a).mean()))}
    save_json(m, bdir / "test_metrics.json")
    pd.DataFrame({"image_id": ids, "y_true": y, "y_pred": prob.argmax(1),
                  **{f"p_{c}": prob[:, i] for i, c in enumerate(names)}}).to_csv(bdir / "test_predictions.csv", index=False)
    log(f"[{args.model}] TEST acc {m['accuracy']:.4f}  macroF1 {m['f1_macro']:.4f}  AUC {m['auc_macro_ovr']:.4f}  "
        f"blast-sens(exact) {m['blast_sensitivity_exact']:.4f}")


if __name__ == "__main__":
    main()
