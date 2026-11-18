"""Representation-learning and classification networks.

* ``MultiResDeiT`` - shared ImageNet-1K DeiT-Small/16 applied to the 128x128 (local) and
  256x256 (global) cellular regions (both resized to 224x224). The two 384-d class-token
  embeddings are scaled by learned resolution weights alpha and concatenated (Eq. 6) into
  a 768-d representation. An auxiliary linear head (weighted CE) and a projection head
  (supervised contrastive loss) are used only while fine-tuning (Eq. 17).
* ``AEMLP``        - Attention-Enhanced MLP 768-512-256-128-3 (Table 5). After each dense
  transformation (Linear -> ReLU -> LayerNorm) a scaled dot-product attention (Eq. 7) is
  computed with W_q, W_k, W_v in R^{d x d}, treating each of the d hidden units as a token;
  the attended vector is added to the dense output (hybrid representation) and dropout
  (p = 0.40) follows.
* Baselines       - ImageNet-1K ResNet-50 and ViT-Small/16 with three-unit heads.
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class MultiResDeiT(nn.Module):
    def __init__(self, backbone="deit_small_patch16_224.fb_in1k", pretrained=True,
                 num_classes=3, proj_dim=128):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        d = self.backbone.num_features
        self.embed_dim = 2 * d
        self.alpha_logits = nn.Parameter(torch.zeros(2))  # learned resolution weights
        self.head = nn.Linear(2 * d, num_classes)
        self.proj = nn.Sequential(nn.Linear(2 * d, 2 * d), nn.ReLU(inplace=True), nn.Linear(2 * d, proj_dim))

    @property
    def alphas(self):
        return 2.0 * torch.softmax(self.alpha_logits, dim=0)  # initialised at (1, 1)

    def embed(self, x_local, x_global):
        f_local = self.backbone(x_local)     # CLS token, 384-d
        f_global = self.backbone(x_global)
        a = self.alphas
        return torch.cat([a[0] * f_local, a[1] * f_global], dim=1)

    def forward(self, x_local, x_global):
        z = self.embed(x_local, x_global)
        return self.head(z), z


class FeatureAttention(nn.Module):
    """A = softmax(Q K^T / sqrt(d_k)) V with Q = W_q h, K = W_k h, V = W_v h (Eq. 7).
    The d hidden units act as tokens (scalar query/key/value per unit), so the attention
    matrix is d x d and models pairwise feature interdependencies; d_k = d."""

    def __init__(self, d):
        super().__init__()
        self.q, self.k, self.v = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
        self.scale = d ** -0.5

    def forward(self, h):
        q, k, v = self.q(h), self.k(h), self.v(h)
        attn = torch.softmax(q.unsqueeze(2) * k.unsqueeze(1) * self.scale, dim=-1)  # (B,d,d)
        return torch.bmm(attn, v.unsqueeze(2)).squeeze(2), attn


class AEMLP(nn.Module):
    def __init__(self, in_dim=768, hidden=(512, 256, 128), num_classes=3, dropout=0.40):
        super().__init__()
        self.dense, self.attn, self.drop = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        self.ln1, self.ln2 = nn.ModuleList(), nn.ModuleList()
        prev = in_dim
        for d in hidden:
            self.dense.append(nn.Linear(prev, d))
            self.ln1.append(nn.LayerNorm(d))
            self.attn.append(FeatureAttention(d))
            self.ln2.append(nn.LayerNorm(d))
            self.drop.append(nn.Dropout(dropout))
            prev = d
        self.out = nn.Linear(prev, num_classes)
        self.last_attn: list[torch.Tensor] = []

    def forward(self, x, keep_attn=False):
        h = x
        self.last_attn = []
        for dense, ln1, att, ln2, drop in zip(self.dense, self.ln1, self.attn, self.ln2, self.drop):
            h = ln1(F.relu(dense(h)))
            a, w = att(h)
            if keep_attn:
                self.last_attn.append(w.detach())
            h = drop(ln2(h + a))
        return self.out(h)


class Pipeline(nn.Module):
    """DeiT embedding + AE-MLP head; differentiable end-to-end (used for Grad-CAM)."""

    def __init__(self, deit: MultiResDeiT, aemlp: AEMLP):
        super().__init__()
        self.deit, self.aemlp = deit, aemlp

    def forward(self, x_local, x_global):
        return self.aemlp(self.deit.embed(x_local, x_global))


def build_resnet50(num_classes=3, pretrained=True):
    w = torchvision.models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
    m = torchvision.models.resnet50(weights=w)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def build_vit_small(backbone="vit_small_patch16_224.augreg_in1k", num_classes=3, pretrained=True):
    return timm.create_model(backbone, pretrained=pretrained, num_classes=num_classes)


def count_params(m: nn.Module, trainable_only=False) -> int:
    return sum(p.numel() for p in m.parameters() if (p.requires_grad or not trainable_only))
