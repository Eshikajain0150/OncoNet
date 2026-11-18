"""Step 9 - controlled robustness study on the held-out test set (Section 4.6, Table 13).

Each un-normalised 256x256 test image is perturbed *before* preprocessing:
  normal            - unchanged
  low_stain         - optical density scaled by 0.6   (weaker staining)
  high_stain        - optical density scaled by 1.4   (stronger staining)
  low_resolution    - down-sampled x4 and up-sampled back (bilinear)
  high_noise        - additive Gaussian noise, sigma = 0.08 (intensities in [0, 1])
and passed end-to-end through (i) the proposed pipeline and (ii) the CNN baseline pipeline
(U-Net + ResNet-50), both with Reinhard normalisation. Segmentation error (1 - Dice against
the reference masks) is compared between UNet++-AG-MSF trained with and without stain
normalisation: reduction = 100 * (err_without - err_with) / err_without.
Requires: 05, 06 --model resnet50, 03 --variant unetpp_ag_msf [--no-stain-norm].
"""
from __future__ import annotations

import cv2
import numpy as np
import torch

from _common import base_parser, ensure_dir, load_json, load_manifest, norm_stats, processed, save_json, setup
from onconet.data import ReinhardNormalizer, imread_gray, imread_rgb, split_frames
from onconet.engine import build_classifier
from onconet.metrics import cls_metrics, seg_image_metrics
from onconet.models import AEMLP, build_seg_model
from onconet.pipeline import EndToEnd
from onconet.utils import get_device, set_seed


def perturb(img: np.ndarray, cond: str, cfg, rng) -> np.ndarray:
    r = cfg["robustness"]
    if cond == "normal":
        return img
    if cond in ("low_stain", "high_stain"):
        k = r["low_stain_od_scale"] if cond == "low_stain" else r["high_stain_od_scale"]
        od = -np.log((img.astype(np.float64) + 1.0) / 256.0)
        return np.clip(256.0 * np.exp(-od * k) - 1.0, 0, 255).astype(np.uint8)
    if cond == "low_resolution":
        f = r["low_resolution_factor"]
        h, w = img.shape[:2]
        small = cv2.resize(img, (w // f, h // f), interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    if cond == "high_noise":
        x = img.astype(np.float64) / 255.0 + rng.normal(0, r["high_noise_std"], img.shape)
        return (np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)
    raise ValueError(cond)


CONDITIONS = ["normal", "low_stain", "high_stain", "low_resolution", "high_noise"]


def load_seg(cfg, out, variant_dir, variant, dev):
    m = build_seg_model(variant, cfg["segmentation"]["channels"]).to(dev)
    m.load_state_dict(torch.load(out / "segmentation" / variant_dir / "best.pt", map_location=dev)["state_dict"])
    return m.eval()


def main():
    args = base_parser(__doc__).parse_args()
    cfg, out, log = setup(args, "09_robustness")
    set_seed(cfg["seed"])
    dev = get_device()
    names = cfg["dataset"]["class_names"]
    df = load_manifest(cfg)
    _, _, te = split_frames(df)
    P = processed(cfg)
    meta = load_json(P / "meta.json")
    rn = ReinhardNormalizer(meta["reinhard_lab"]["mean"], meta["reinhard_lab"]["std"])
    mean, std = norm_stats(cfg, True)
    mean_raw, std_raw = norm_stats(cfg, False)

    deit = build_classifier(cfg, "deit_mr", len(names))
    deit.load_state_dict(torch.load(out / "proposed" / "deit" / "best.pt", map_location=dev)["state_dict"])
    deit = deit.to(dev).eval()
    ck = torch.load(out / "proposed" / "aemlp_final.pt", map_location=dev)
    mlp = AEMLP(ck["in_dim"], tuple(cfg["aemlp"]["hidden"]), len(names), cfg["aemlp"]["dropout"]).to(dev)
    mlp.load_state_dict(ck["state_dict"])
    mlp.eval()
    proposed = EndToEnd(cfg, load_seg(cfg, out, "unetpp_ag_msf", "unetpp_ag_msf", dev),
                        lambda xl, xg: mlp(deit.embed(xl, xg)), mean, std, rn)
    res50 = build_classifier(cfg, "resnet50", len(names))
    res50.load_state_dict(torch.load(out / "baselines" / "resnet50" / "best.pt", map_location=dev)["state_dict"])
    res50 = res50.to(dev).eval()
    cnn = EndToEnd(cfg, load_seg(cfg, out, "unet", "unet", dev), lambda xl, xg: res50(xg), mean, std, rn)
    seg_nostain = None
    if (out / "segmentation" / "unetpp_ag_msf_nostain" / "best.pt").exists():
        seg_nostain = EndToEnd(cfg, load_seg(cfg, out, "unetpp_ag_msf_nostain", "unetpp_ag_msf", dev),
                               None, mean_raw, std_raw, None)
    else:
        log("unetpp_ag_msf_nostain not found - segmentation-error reduction will be skipped")

    raws = [imread_rgb(P / "images_256_raw" / f"{i}.png") for i in te.image_id]
    gts = [imread_gray(P / "masks_cell" / f"{i}.png") > 127 for i in te.image_id]
    y = te.label_idx.values
    results = {}
    for cond in CONDITIONS:
        rng = np.random.default_rng(cfg["seed"])
        imgs = [perturb(r, cond, cfg, rng) for r in raws]
        row = {}
        for tag, pipe in (("proposed", proposed), ("cnn_baseline", cnn)):
            probs, masks, _ = [], [], []
            for s in range(0, len(imgs), 8):
                p, m, _ = pipe.predict(imgs[s:s + 8])
                probs.append(p)
                masks += m
            prob = np.concatenate(probs)
            row[f"{tag}_accuracy"] = cls_metrics(y, prob, names)["accuracy"]
            row[f"{tag}_seg_dice"] = float(np.mean([seg_image_metrics(m, g)["dice"] for m, g in zip(masks, gts)]))
        if seg_nostain is not None:
            masks = []
            for s in range(0, len(imgs), 8):
                masks += seg_nostain.segment([seg_nostain.preprocess(i) for i in imgs[s:s + 8]])[0]
            row["nostain_seg_dice"] = float(np.mean([seg_image_metrics(m, g)["dice"] for m, g in zip(masks, gts)]))
            e_with, e_without = 1 - row["proposed_seg_dice"], 1 - row["nostain_seg_dice"]
            row["seg_error_reduction_pct"] = 100 * (e_without - e_with) / e_without if e_without > 0 else float("nan")
        results[cond] = row
        log(f"[{cond}] proposed acc {row['proposed_accuracy']:.4f}  CNN acc {row['cnn_baseline_accuracy']:.4f}  "
            f"seg dice {row['proposed_seg_dice']:.4f}"
            + (f"  no-stain dice {row['nostain_seg_dice']:.4f}  error reduction {row['seg_error_reduction_pct']:.1f}%"
               if "nostain_seg_dice" in row else ""))
    save_json({"settings": cfg["robustness"], "results": results}, ensure_dir(out / "robustness") / "robustness.json")


if __name__ == "__main__":
    main()
