"""Step 1 - build the 500-image task-specific subset, stratified 70/15/15 hold-out split and
five CV folds, resize to 256x256, Reinhard-normalise, compute training-only statistics and
class weights (manuscript Sections 3.1.1-3.1.2, 3.3.3, 3.3.4)."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from _common import base_parser, ensure_dir, processed, save_json, setup
from onconet.data import (ReinhardNormalizer, assign_splits, build_subset, choose_reference_image,
                          class_weights_from_counts, discover_class_folders, imread_rgb, imwrite_rgb,
                          resize_rgb)


def channel_stats(paths):
    s, ss, n = np.zeros(3), np.zeros(3), 0
    for p in paths:
        x = imread_rgb(p).reshape(-1, 3).astype(np.float64) / 255.0
        s += x.sum(0)
        ss += (x ** 2).sum(0)
        n += len(x)
    mean = s / n
    return mean.tolist(), np.sqrt(ss / n - mean ** 2).tolist()


def main():
    args = base_parser(__doc__).parse_args()
    cfg, out, log = setup(args, "01_prepare_data")
    raw = Path(cfg["paths"]["raw_dir"])
    pdir = ensure_dir(processed(cfg))

    # ---- audit of the full archive (transparency about the source data)
    folders = discover_class_folders(raw, cfg["dataset"]["image_extensions"])
    audit = {}
    for f, paths in sorted(folders.items()):
        shapes = Counter()
        for p in paths:
            h, w = imread_rgb(p).shape[:2]
            shapes[f"{w}x{h}"] += 1
        audit[f] = {"n_images": len(paths), "image_sizes": dict(shapes.most_common())}
        log(f"archive folder '{f}': {len(paths)} images, sizes {dict(shapes.most_common(3))}")
    save_json(audit, out / "data" / "archive_audit.json")

    # ---- subset + splits
    df = build_subset(cfg)
    df = assign_splits(df, cfg)
    df["src_path"] = [str(Path(p).relative_to(raw)) for p in df.src_path]

    # ---- resize to 256 (raw) and Reinhard-normalise
    size = cfg["preprocess"]["seg_size"]
    raw_dir, norm_dir = ensure_dir(pdir / "images_256_raw"), ensure_dir(pdir / "images_256")
    oh, ow = [], []
    for r in df.itertuples():
        img = imread_rgb(raw / r.src_path)
        oh.append(img.shape[0])
        ow.append(img.shape[1])
        imwrite_rgb(raw_dir / f"{r.image_id}.png", resize_rgb(img, size))
    df["orig_h"], df["orig_w"] = oh, ow

    train_ids = df[df.split == "train"].image_id.tolist()
    ref_i, _ = choose_reference_image([raw_dir / f"{i}.png" for i in train_ids])
    ref_id = train_ids[ref_i]
    rn = ReinhardNormalizer().fit(imread_rgb(raw_dir / f"{ref_id}.png"))
    for iid in df.image_id:
        imwrite_rgb(norm_dir / f"{iid}.png", rn.transform(imread_rgb(raw_dir / f"{iid}.png")))
    log(f"Reinhard reference image: {ref_id} (training partition)")

    # ---- training-partition statistics and class weights
    stats = {"reinhard": dict(zip(("mean", "std"), channel_stats([norm_dir / f"{i}.png" for i in train_ids]))),
             "raw": dict(zip(("mean", "std"), channel_stats([raw_dir / f"{i}.png" for i in train_ids])))}
    names = cfg["dataset"]["class_names"]
    tr_counts = [int(((df.split == "train") & (df.label == c)).sum()) for c in names]
    weights = class_weights_from_counts(tr_counts)

    table = {s: {c: int(((df.split == s) & (df.label == c)).sum()) for c in names} for s in ("train", "val", "test")}
    folds = {int(k): {c: int(((df.cv_fold == k) & (df.label == c)).sum()) for c in names}
             for k in sorted(df.cv_fold.unique()) if k >= 0}
    meta = {"n_images": len(df), "class_names": names, "mapping": cfg["dataset"]["mapping"],
            "split_counts": table, "cv_fold_counts": folds, "train_class_counts": tr_counts,
            "class_weights": weights, "reinhard_reference": ref_id, "reinhard_lab": rn.state(),
            "norm_stats": stats, "subset_source_folders": df.groupby("label").src_folder.value_counts().to_dict()}
    save_json(meta, pdir / "meta.json")
    df.to_csv(pdir / "manifest.csv", index=False)
    save_json(meta, out / "data" / "dataset_summary.json")
    log("split counts:", table)
    log("CV fold counts:", folds)
    log("training class counts:", dict(zip(names, tr_counts)), "-> class weights:",
        {c: round(w, 3) for c, w in zip(names, weights)})
    log("done")


if __name__ == "__main__":
    main()
