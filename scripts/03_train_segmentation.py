"""Step 3 - train and evaluate the segmentation variants of Table 2 on the primary hold-out
split (350 train / 75 val / 75 test). The checkpoint with the highest validation Dice is
evaluated once on the held-out test set.

    python scripts/03_train_segmentation.py --variant unetpp_ag_msf
    python scripts/03_train_segmentation.py --all                # Table 2 (five variants)
    python scripts/03_train_segmentation.py --variant unetpp_ag_msf --no-stain-norm   # robustness
    python scripts/03_train_segmentation.py --variant otsu       # classical reference (Fig. 21)
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import binary_fill_holes

from _common import base_parser, ensure_dir, image_dir, load_manifest, norm_stats, processed, save_json, setup
from onconet.data import imread_rgb, split_frames
from onconet.engine import evaluate_segmentation, train_segmentation
from onconet.models import SEG_VARIANTS, count_params
from onconet.pipeline import select_cell_component


def otsu_probs(df, img_dir):
    """Classical reference: Otsu threshold on grey level (stained cell darker than background),
    centre component, hole filling. Returned as {id: {0,1} float map}."""
    out = {}
    for iid in df.image_id:
        g = cv2.cvtColor(imread_rgb(Path(img_dir) / f"{iid}.png"), cv2.COLOR_RGB2GRAY)
        _, b = cv2.threshold(cv2.GaussianBlur(g, (5, 5), 0), 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        m, _ = select_cell_component(b)
        out[iid] = binary_fill_holes(m).astype(np.float32)
    return out


def run_variant(cfg, out, log, variant, stain_norm):
    df = load_manifest(cfg)
    tr, va, te = split_frames(df)
    mean, std = norm_stats(cfg, stain_norm)
    idir, mdir = image_dir(cfg, stain_norm), processed(cfg) / "masks_cell"
    name = variant + ("" if stain_norm else "_nostain")
    vdir = ensure_dir(out / "segmentation" / name)
    if variant == "otsu":
        res = {s: evaluate_segmentation(None, d, idir, mdir, cfg, mean, std, probs=otsu_probs(d, idir))
               for s, d in (("val", va), ("test", te))}
        best_epoch, n_params = None, 0
    else:
        model, hist, best_epoch = train_segmentation(cfg, variant, tr, va, idir, mdir, mean, std, vdir, log)
        n_params = count_params(model)
        res = {s: evaluate_segmentation(model, d, idir, mdir, cfg, mean, std) for s, d in (("val", va), ("test", te))}
    for s in ("val", "test"):
        pd.DataFrame(res[s]["per_image"]).to_csv(vdir / f"{s}_per_image.csv", index=False)
        save_json({"variant": variant, "stain_norm": stain_norm, "best_epoch": best_epoch, "n_params": n_params,
                   "threshold": cfg["segmentation"]["threshold"], "n_images": len(res[s]["per_image"]),
                   "mean_per_image": res[s]["summary"], "pooled_pixels": res[s]["pooled"]},
                  vdir / f"{s}_metrics.json")
    t = res["test"]["summary"]
    log(f"[{name}] TEST  dice {t['dice']:.4f}  iou {t['iou']:.4f}  precision {t['precision']:.4f}  "
        f"recall {t['recall']:.4f}  accuracy {t['accuracy']:.4f}  boundaryF1 {t['boundary_f1']:.4f}")


def main():
    p = base_parser(__doc__)
    p.add_argument("--variant", choices=list(SEG_VARIANTS) + ["otsu"], default="unetpp_ag_msf")
    p.add_argument("--all", action="store_true", help="run the five Table-2 variants + Otsu")
    p.add_argument("--no-stain-norm", action="store_true")
    args = p.parse_args()
    cfg, out, log = setup(args, "03_train_segmentation")
    variants = (list(SEG_VARIANTS) + ["otsu"]) if args.all else [args.variant]
    for v in variants:
        run_variant(cfg, out, log, v, not args.no_stain_norm)


if __name__ == "__main__":
    main()
