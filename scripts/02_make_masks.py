"""Step 2 - generate the study-specific binary reference masks (cell and nucleus) for the 500
images at 256x256 and write a quality-control montage (see onconet/masks.py for the
procedure). Masks are computed on the un-normalised resized images so that they do not
depend on the Reinhard reference choice.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from _common import base_parser, ensure_dir, load_manifest, processed, save_json, setup
from onconet.data import imread_rgb, imwrite_gray, imwrite_rgb
from onconet.masks import MaskParams, reference_masks


def main():
    p = base_parser(__doc__)
    p.add_argument("--qc-per-class", type=int, default=8)
    args = p.parse_args()
    cfg, out, log = setup(args, "02_make_masks")
    df = load_manifest(cfg)
    P = processed(cfg)
    prm = MaskParams()
    cdir, ndir = ensure_dir(P / "masks_cell"), ensure_dir(P / "masks_nucleus")
    stats = []
    for r in df.itertuples():
        img = imread_rgb(P / "images_256_raw" / f"{r.image_id}.png")
        cell, nuc = reference_masks(img, prm)
        imwrite_gray(cdir / f"{r.image_id}.png", cell.astype(np.uint8) * 255)
        imwrite_gray(ndir / f"{r.image_id}.png", nuc.astype(np.uint8) * 255)
        stats.append({"image_id": r.image_id, "label": r.label, "src_folder": r.src_folder,
                      "cell_frac": cell.mean(), "nucleus_frac": nuc.mean(),
                      "nucleus_in_cell_frac": nuc.sum() / max(cell.sum(), 1)})
    st = pd.DataFrame(stats)
    st.to_csv(ensure_dir(out / "masks") / "mask_stats.csv", index=False)
    summary = {"params": prm.as_dict(),
               "by_folder": st.groupby("src_folder")[["cell_frac", "nucleus_frac", "nucleus_in_cell_frac"]].describe().round(4).to_dict(),
               "n_empty_cell_masks": int((st.cell_frac == 0).sum())}
    save_json(summary, out / "masks" / "mask_summary.json")
    log(st.groupby("src_folder")[["cell_frac", "nucleus_frac"]].mean().round(3).to_string())

    # QC montage: image | cell contour (green) + nucleus contour (blue)
    import cv2
    tiles = []
    for _, g in df.groupby("src_folder"):
        row = []
        for iid in g.image_id.head(args.qc_per_class):
            img = imread_rgb(P / "images_256_raw" / f"{iid}.png").copy()
            for d, col in ((cdir, (0, 255, 0)), (ndir, (0, 0, 255))):
                m = cv2.imdecode(np.fromfile(str(d / f"{iid}.png"), np.uint8), cv2.IMREAD_GRAYSCALE)
                cs, _ = cv2.findContours((m > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(img, cs, -1, col, 2)
            row.append(img)
        while len(row) < args.qc_per_class:
            row.append(np.zeros_like(row[0]))
        tiles.append(np.concatenate(row, 1))
    imwrite_rgb(out / "masks" / "mask_qc_montage.png", np.concatenate(tiles, 0))
    log(f"masks written; empty cell masks: {summary['n_empty_cell_masks']}; QC montage: outputs/masks/mask_qc_montage.png")


if __name__ == "__main__":
    main()
