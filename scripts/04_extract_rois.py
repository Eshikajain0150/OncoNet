"""Step 4 - run a trained segmentation model on all 500 images and write the multi-resolution
cellular regions used by the classifiers: global (256x256, background outside the predicted
cell set to zero) and local (128x128 centred on the cell centroid).

    python scripts/04_extract_rois.py --seg-variant unetpp_ag_msf   # proposed pipeline
    python scripts/04_extract_rois.py --seg-variant unet            # CNN / ViT baselines
"""
from __future__ import annotations

import torch

from _common import base_parser, image_dir, load_manifest, norm_stats, processed, save_json, setup
from onconet.engine import predict_segmentation
from onconet.models import SEG_VARIANTS, build_seg_model
from onconet.pipeline import write_rois
from onconet.utils import get_device


def main():
    p = base_parser(__doc__)
    p.add_argument("--seg-variant", choices=list(SEG_VARIANTS), default="unetpp_ag_msf")
    args = p.parse_args()
    cfg, out, log = setup(args, "04_extract_rois")
    df = load_manifest(cfg)
    mean, std = norm_stats(cfg)
    ck = torch.load(out / "segmentation" / args.seg_variant / "best.pt", map_location=get_device())
    model = build_seg_model(args.seg_variant, cfg["segmentation"]["channels"]).to(get_device())
    model.load_state_dict(ck["state_dict"])
    probs = predict_segmentation(model, df, image_dir(cfg), cfg, mean, std)
    roi_dir = processed(cfg) / "rois" / args.seg_variant
    meta = write_rois(probs, df, image_dir(cfg), roi_dir, cfg)
    n_empty = int(meta.empty_mask.sum())
    save_json({"seg_variant": args.seg_variant, "n_images": len(meta), "n_empty_masks": n_empty},
              out / "rois" / f"{args.seg_variant}_summary.json")
    log(f"ROIs written to {roi_dir}; images with no predicted cell (fallback to full image): {n_empty}")


if __name__ == "__main__":
    main()
