"""Grad-CAM, Attention Focus Score (Eqs. 9a/9b) and regional (nucleus/cytoplasm) attention
(Section 3.2.4)."""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F


class _ActCatcher:
    def __init__(self, module):
        self.acts = []
        self.h = module.register_forward_hook(lambda m, i, o: self.acts.append(o))

    def remove(self):
        self.h.remove()


def _token_cam(A, G, n_prefix):
    A, G = A[:, n_prefix:, :], G[:, n_prefix:, :]
    w = G.mean(1, keepdim=True)
    cam = F.relu((A * w).sum(-1))
    side = int(round(cam.shape[1] ** 0.5))
    return cam.view(cam.shape[0], side, side)


def gradcam_proposed(pipe, xl, xg, target=None):
    """Grad-CAM for the full DeiT -> AE-MLP pipeline, taken at the final transformer block of
    the shared DeiT (token input of its attention, ``blocks[-1].norm1``; patch tokens after the
    last block no longer reach the class token, so their gradients are zero) for the local and
    global views (predicted class unless ``target``)."""
    bb = pipe.deit.backbone
    cat = _ActCatcher(bb.blocks[-1].norm1)
    pipe.eval()
    with torch.enable_grad():
        logits = pipe(xl, xg)
        cls = logits.argmax(1) if target is None else target
        score = logits.gather(1, cls.view(-1, 1)).sum()
        grads = torch.autograd.grad(score, cat.acts)
    cat.remove()
    n_prefix = getattr(bb, "num_prefix_tokens", 1)
    cam_local = _token_cam(cat.acts[0].detach(), grads[0], n_prefix)
    cam_global = _token_cam(cat.acts[1].detach(), grads[1], n_prefix)
    return torch.softmax(logits.detach(), 1), cam_local, cam_global


def gradcam_vit(vit, x, target=None):
    cat = _ActCatcher(vit.blocks[-1].norm1)
    vit.eval()
    with torch.enable_grad():
        logits = vit(x)
        cls = logits.argmax(1) if target is None else target
        grads = torch.autograd.grad(logits.gather(1, cls.view(-1, 1)).sum(), cat.acts)
    cat.remove()
    return torch.softmax(logits.detach(), 1), _token_cam(cat.acts[0].detach(), grads[0],
                                                          getattr(vit, "num_prefix_tokens", 1))


def gradcam_resnet(resnet, x, target=None):
    cat = _ActCatcher(resnet.layer4)
    resnet.eval()
    with torch.enable_grad():
        logits = resnet(x)
        cls = logits.argmax(1) if target is None else target
        grads = torch.autograd.grad(logits.gather(1, cls.view(-1, 1)).sum(), cat.acts)
    cat.remove()
    A, G = cat.acts[0].detach(), grads[0]
    cam = F.relu((A * G.mean((2, 3), keepdim=True)).sum(1))
    return torch.softmax(logits.detach(), 1), cam


def to_image_space(cam: np.ndarray, size=256, region=None, local_size=128) -> np.ndarray:
    """Rectify, bilinearly resize to image space and min-max normalise to [0, 1].
    ``region=(y0, x0)`` places a local-view map inside a size x size canvas."""
    cam = np.maximum(cam, 0).astype(np.float32)
    if region is None:
        out = cv2.resize(cam, (size, size), interpolation=cv2.INTER_LINEAR)
    else:
        y0, x0 = region
        out = np.zeros((size, size), np.float32)
        out[y0:y0 + local_size, x0:x0 + local_size] = cv2.resize(
            cam, (local_size, local_size), interpolation=cv2.INTER_LINEAR)
    out = np.maximum(out, 0)
    lo, hi = out.min(), out.max()
    return (out - lo) / (hi - lo) if hi > lo else np.zeros_like(out)


def attention_focus_score(A: np.ndarray, M: np.ndarray, eps=1e-8) -> float:
    """AFS_i = 100 * sum(A*M) / (sum(A) + eps)  (Eq. 9a)."""
    return float(100.0 * (A * M).sum() / (A.sum() + eps))


def regional_shares(A, nucleus, cell, eps=1e-8) -> dict:
    cyto = cell & ~nucleus
    n, c = float((A * nucleus).sum()), float((A * cyto).sum())
    peak = np.unravel_index(int(np.argmax(A)), A.shape)
    return {"nucleus_share": 100 * n / (n + c + eps), "cytoplasm_share": 100 * c / (n + c + eps),
            "peak_in_nucleus": bool(nucleus[peak]), "peak_in_cell": bool(cell[peak]),
            "background_fraction": 100 - attention_focus_score(A, cell)}


def overlay(img: np.ndarray, A: np.ndarray, alpha=0.45) -> np.ndarray:
    heat = cv2.applyColorMap((A * 255).astype(np.uint8), cv2.COLORMAP_JET)[..., ::-1]
    return (img * (1 - alpha) + heat * alpha).astype(np.uint8)
