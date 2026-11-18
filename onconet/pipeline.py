"""Segmentation-to-classification glue: predicted-mask post-processing, multi-resolution
ROI extraction (Section 3.1.4 / 3.2.2) and an end-to-end inference wrapper used for the
robustness study and runtime benchmark."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from .data import ReinhardNormalizer, imread_rgb, imwrite_gray, imwrite_rgb, resize_rgb
from .utils import get_device


def select_cell_component(binary: np.ndarray) -> tuple[np.ndarray, bool]:
    """Keep the connected component containing the image centre (images are centred on the
    target cell); otherwise the largest component. Returns (mask, empty_flag)."""
    binary = binary.astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return np.zeros_like(binary, bool), True
    h, w = binary.shape
    c = lab[h // 2, w // 2]
    if c == 0:
        c = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    return lab == c, False


def postprocess_prob(prob: np.ndarray, thr=0.5, dilation=3) -> tuple[np.ndarray, bool]:
    mask, empty = select_cell_component(prob >= thr)
    if not empty and dilation > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilation + 1, 2 * dilation + 1))
        mask = cv2.dilate(mask.astype(np.uint8), k) > 0
    return mask, empty


def make_rois(img: np.ndarray, mask: np.ndarray, local_size=128):
    """Global view = the 256x256 image restricted to the segmented cell (background set to 0);
    local view = a local_size x local_size region centred on the cell centroid."""
    H, W = mask.shape
    if mask.any():
        glob = img * mask[..., None].astype(img.dtype)
        ys, xs = np.nonzero(mask)
        cy, cx = int(round(ys.mean())), int(round(xs.mean()))
    else:  # no cell found: fall back to the unmasked image and a centred crop
        glob = img.copy()
        cy, cx = H // 2, W // 2
    y0 = int(np.clip(cy - local_size // 2, 0, H - local_size))
    x0 = int(np.clip(cx - local_size // 2, 0, W - local_size))
    local = glob[y0:y0 + local_size, x0:x0 + local_size]
    return glob, local, (y0, x0)


def write_rois(probs: dict, df, img_dir, roi_dir, cfg) -> pd.DataFrame:
    roi_dir = Path(roi_dir)
    thr = cfg["segmentation"]["threshold"]
    dil = cfg["preprocess"]["roi_mask_dilation"]
    rows = []
    for iid in df.image_id:
        img = imread_rgb(Path(img_dir) / f"{iid}.png")
        mask, empty = postprocess_prob(probs[iid], thr, dil)
        glob, local, (y0, x0) = make_rois(img, mask, cfg["preprocess"]["local_size"])
        imwrite_rgb(roi_dir / "global" / f"{iid}.png", glob)
        imwrite_rgb(roi_dir / "local" / f"{iid}.png", local)
        imwrite_gray(roi_dir / "pred_masks" / f"{iid}.png", mask.astype(np.uint8) * 255)
        rows.append({"image_id": iid, "y0": y0, "x0": x0, "empty_mask": empty, "mask_area": int(mask.sum())})
    meta = pd.DataFrame(rows)
    meta.to_csv(roi_dir / "roi_meta.csv", index=False)
    return meta


# =========================================================================== end-to-end
def _to_tensor(img_uint8: np.ndarray, mean, std, size=None) -> torch.Tensor:
    if size is not None and img_uint8.shape[0] != size:
        img_uint8 = cv2.resize(img_uint8, (size, size), interpolation=cv2.INTER_LINEAR)
    t = torch.from_numpy(img_uint8).permute(2, 0, 1).float() / 255.0
    m = torch.tensor(mean).view(3, 1, 1)
    s = torch.tensor(std).view(3, 1, 1)
    return (t - m) / s


class EndToEnd:
    """raw RGB image -> resize 256 -> (Reinhard) -> segmentation -> ROIs -> classifier.
    ``classify`` is a callable taking (local_batch, global_batch) tensors -> logits."""

    def __init__(self, cfg, seg_model, classify, mean, std, reinhard: ReinhardNormalizer | None,
                 two_view=True):
        self.cfg, self.seg, self.classify = cfg, seg_model.eval(), classify
        self.mean, self.std, self.reinhard, self.two_view = mean, std, reinhard, two_view
        self.dev = get_device()

    def preprocess(self, raw: np.ndarray) -> np.ndarray:
        img = resize_rgb(raw, self.cfg["preprocess"]["seg_size"]) if raw.shape[0] != self.cfg["preprocess"]["seg_size"] or raw.shape[1] != self.cfg["preprocess"]["seg_size"] else raw
        return self.reinhard.transform(img) if self.reinhard is not None else img

    @torch.no_grad()
    def segment(self, imgs: list[np.ndarray]):
        x = torch.stack([_to_tensor(i, self.mean, self.std) for i in imgs]).to(self.dev)
        probs = torch.sigmoid(self.seg(x))[:, 0].float().cpu().numpy()
        out = [postprocess_prob(p, self.cfg["segmentation"]["threshold"],
                                self.cfg["preprocess"]["roi_mask_dilation"]) for p in probs]
        return [m for m, _ in out], probs

    def roi_tensors(self, imgs, masks):
        vs = self.cfg["preprocess"]["vit_size"]
        L, G = [], []
        for img, m in zip(imgs, masks):
            glob, local, _ = make_rois(img, m, self.cfg["preprocess"]["local_size"])
            L.append(_to_tensor(local, self.mean, self.std, vs))
            G.append(_to_tensor(glob, self.mean, self.std, vs))
        return torch.stack(L).to(self.dev), torch.stack(G).to(self.dev)

    @torch.no_grad()
    def predict(self, raws: list[np.ndarray]):
        imgs = [self.preprocess(r) for r in raws]
        masks, probs = self.segment(imgs)
        xl, xg = self.roi_tensors(imgs, masks)
        logits = self.classify(xl, xg)
        return torch.softmax(logits, 1).float().cpu().numpy(), masks, probs
