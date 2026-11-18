"""Step 8 - quantitative interpretability on the 75-image held-out test set (Sections 3.2.4,
4.3, 4.5; Tables 10 and 12).

For every test image a Grad-CAM map is computed for the predicted class from the final
transformer block (proposed DeiT->AE-MLP pipeline, ViT-S baseline) or from layer4
(ResNet-50 baseline). Maps are rectified, bilinearly resized to 256x256, min-max normalised
and compared with the binary reference cell mask:
  * AFS (Eq. 9a/9b)        - % of normalised attention mass inside the reference cell mask
  * regional shares         - nucleus vs cytoplasm share of the in-cell attention, by predicted class
  * pointing game           - % images whose attention peak lies in the nucleus / in the cell
  * AML-vs-ALL accuracy      - accuracy restricted to AML and ALL test images (2-way argmax)
Primary AFS for the proposed model uses the global (256->224) view, which, like the
baselines' single view, covers the whole image; the fused global+local map is also reported.
Requires: 05, 06 (both baselines) and 02 (nucleus masks).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from _common import base_parser, ensure_dir, image_dir, load_manifest, norm_stats, processed, save_json, setup
from onconet.data import ROIDataset, collate_views, imread_gray, imread_rgb, imwrite_rgb, split_frames
from onconet.engine import build_classifier
from onconet.interpret import (attention_focus_score, gradcam_proposed, gradcam_resnet, gradcam_vit, overlay,
                               regional_shares, to_image_space)
from onconet.metrics import cls_metrics
from onconet.models import AEMLP, Pipeline
from onconet.utils import get_device, set_seed


def load_proposed(cfg, out, dev):
    names = cfg["dataset"]["class_names"]
    deit = build_classifier(cfg, "deit_mr", len(names))
    deit.load_state_dict(torch.load(out / "proposed" / "deit" / "best.pt", map_location=dev)["state_dict"])
    ck = torch.load(out / "proposed" / "aemlp_final.pt", map_location=dev)
    a = cfg["aemlp"]
    mlp = AEMLP(ck["in_dim"], tuple(a["hidden"]), len(names), a["dropout"])
    mlp.load_state_dict(ck["state_dict"])
    return Pipeline(deit, mlp).to(dev).eval()


def load_baseline(cfg, out, kind, dev):
    m = build_classifier(cfg, kind, len(cfg["dataset"]["class_names"]))
    m.load_state_dict(torch.load(out / "baselines" / kind / "best.pt", map_location=dev)["state_dict"])
    return m.to(dev).eval()


def main():
    p = base_parser(__doc__)
    p.add_argument("--n-figures", type=int, default=6)
    args = p.parse_args()
    cfg, out, log = setup(args, "08_interpretability")
    set_seed(cfg["seed"])
    dev = get_device()
    names = cfg["dataset"]["class_names"]
    df = load_manifest(cfg)
    _, _, te = split_frames(df)
    mean, std = norm_stats(cfg)
    P = processed(cfg)
    idir = ensure_dir(out / "interpretability")
    figdir = ensure_dir(idir / "examples")
    S = cfg["preprocess"]["seg_size"]
    L = cfg["preprocess"]["local_size"]

    models = {"proposed": load_proposed(cfg, out, dev)}
    for kind in ("resnet50", "vit_small"):
        if (out / "baselines" / kind / "best.pt").exists():
            models[kind] = load_baseline(cfg, out, kind, dev)
    roi_dirs = {"proposed": P / "rois" / "unetpp_ag_msf", "resnet50": P / "rois" / "unet", "vit_small": P / "rois" / "unet"}

    rows = []
    for name, model in models.items():
        views = ("local", "global") if name == "proposed" else ("global",)
        ds = ROIDataset(te, roi_dirs[name], cfg, False, mean, std, views)
        meta = pd.read_csv(roi_dirs[name] / "roi_meta.csv").set_index("image_id")
        for i in range(len(ds)):
            vs, y, iid = collate_views([ds[i]])
            vs = [v.to(dev) for v in vs]
            if name == "proposed":
                prob, cl, cg = gradcam_proposed(model, vs[0], vs[1])
                A = to_image_space(cg[0].cpu().numpy(), S)
                Al = to_image_space(cl[0].cpu().numpy(), S, region=(meta.loc[iid[0], "y0"], meta.loc[iid[0], "x0"]),
                                    local_size=L)
                Af = to_image_space(A + Al, S)  # fused (re-normalised) map
            elif name == "resnet50":
                prob, c = gradcam_resnet(model, vs[0])
                A = to_image_space(c[0].cpu().numpy(), S)
                Af = A
            else:
                prob, c = gradcam_vit(model, vs[0])
                A = to_image_space(c[0].cpu().numpy(), S)
                Af = A
            cell = imread_gray(P / "masks_cell" / f"{iid[0]}.png") > 127
            nuc = imread_gray(P / "masks_nucleus" / f"{iid[0]}.png") > 127
            pr = prob[0].cpu().numpy()
            r = {"model": name, "image_id": iid[0], "y_true": int(y[0]), "y_pred": int(pr.argmax()),
                 **{f"p_{c}": float(pr[j]) for j, c in enumerate(names)},
                 "afs": attention_focus_score(A, cell), "afs_fused": attention_focus_score(Af, cell),
                 **regional_shares(A, nuc, cell)}
            rows.append(r)
            if i < args.n_figures:
                img = imread_rgb(image_dir(cfg) / f"{iid[0]}.png")
                imwrite_rgb(figdir / f"{name}_{iid[0]}.png", np.concatenate([img, overlay(img, A)], axis=1))

    res = pd.DataFrame(rows)
    res.to_csv(idir / "per_image.csv", index=False)
    summary = {}
    for name, g in res.groupby("model"):
        prob = g[[f"p_{c}" for c in names]].values
        cm = cls_metrics(g.y_true.values, prob, names)
        s = {"n": len(g), "afs_mean": g.afs.mean(), "afs_sd": g.afs.std(ddof=1), "afs_fused_mean": g.afs_fused.mean(),
             "peak_in_nucleus_pct": 100 * g.peak_in_nucleus.mean(), "peak_in_cell_pct": 100 * g.peak_in_cell.mean(),
             "background_fraction_mean": g.background_fraction.mean(),
             "aml_vs_all_accuracy": cm.get("aml_vs_all_accuracy"),
             "regional_by_predicted_class": {names[c]: {"n": int((g.y_pred == c).sum()),
                                                        "nucleus_share": float(g[g.y_pred == c].nucleus_share.mean()),
                                                        "cytoplasm_share": float(g[g.y_pred == c].cytoplasm_share.mean())}
                                             for c in range(len(names)) if (g.y_pred == c).any()}}
        summary[name] = s
        log(f"[{name}] AFS {s['afs_mean']:.2f}% (fused {s['afs_fused_mean']:.2f}%)  peak-in-nucleus "
            f"{s['peak_in_nucleus_pct']:.1f}%  peak-in-cell {s['peak_in_cell_pct']:.1f}%  "
            f"AML-vs-ALL acc {s['aml_vs_all_accuracy']}")
        for c, v in s["regional_by_predicted_class"].items():
            log(f"[{name}]   predicted {c}: nucleus {v['nucleus_share']:.2f}%  cytoplasm {v['cytoplasm_share']:.2f}%  (n={v['n']})")
    if {"proposed", "resnet50"} <= set(res.model):
        a = res[res.model == "proposed"].set_index("image_id").afs
        b = res[res.model == "resnet50"].set_index("image_id").afs.reindex(a.index)
        from scipy.stats import wilcoxon
        summary["afs_proposed_vs_resnet50_wilcoxon_p"] = float(wilcoxon(a, b).pvalue) if (a != b).any() else 1.0

    # AE-MLP feature-level attribution: share of |grad x input| on the local vs global halves
    mdl = models["proposed"]
    ds = ROIDataset(te, roi_dirs["proposed"], cfg, False, mean, std, ("local", "global"))
    shares = []
    for i in range(len(ds)):
        vs, _, _ = collate_views([ds[i]])
        with torch.no_grad():
            z = mdl.deit.embed(vs[0].to(dev), vs[1].to(dev))
        z.requires_grad_(True)
        logit = mdl.aemlp(z)
        g, = torch.autograd.grad(logit[0, logit.argmax()], z)
        a = (g * z).abs()[0].detach().cpu().numpy()
        h = len(a) // 2
        shares.append(a[:h].sum() / a.sum())
    summary["aemlp_attribution_local_share_pct"] = 100 * float(np.mean(shares))
    save_json(summary, idir / "summary.json")
    log("AE-MLP attribution: local-view share", round(summary["aemlp_attribution_local_share_pct"], 2), "%")


if __name__ == "__main__":
    main()
