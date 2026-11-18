"""Segmentation networks (manuscript Sections 3.2.1 and 3.3.4, Table 2).

* ``UNet``     - conventional five-level U-Net baseline (channels 32-512).
* ``UNetPP``   - five-level UNet++ (nested dense skip pathways, Zhou et al. 2018) with
                 optional Attention Gates on the skip pathways (Oktay et al. 2018): the
                 incoming same-level node X(i,j-1) is gated by the deeper node X(i+1,j-1)
                 before the dense concatenation (as in Attention UNet++, Li et al. 2020),
                 and optional multi-scale decoder fusion of X(0,4), X(1,3), X(2,2), X(3,1)
                 before the final 1x1 convolution.

Variants used in Table 2:
    unet            UNet
    unetpp          UNetPP(attention=False, msf=False)
    unetpp_ag       UNetPP(attention=True,  msf=False)
    unetpp_msf      UNetPP(attention=False, msf=True)
    unetpp_ag_msf   UNetPP(attention=True,  msf=True)   <- proposed
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Sequential):
    def __init__(self, cin, cout):
        super().__init__(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )


class AttentionGate(nn.Module):
    """Additive attention gate: alpha = sigmoid(psi(relu(W_x x + W_g g)))."""

    def __init__(self, f_x, f_g, f_int):
        super().__init__()
        self.wx = nn.Sequential(nn.Conv2d(f_x, f_int, 1, bias=False), nn.BatchNorm2d(f_int))
        self.wg = nn.Sequential(nn.Conv2d(f_g, f_int, 1, bias=True), nn.BatchNorm2d(f_int))
        self.psi = nn.Sequential(nn.Conv2d(f_int, 1, 1, bias=True), nn.BatchNorm2d(1), nn.Sigmoid())

    def forward(self, x, g):
        g = F.interpolate(g, size=x.shape[-2:], mode="bilinear", align_corners=False)
        alpha = self.psi(F.relu(self.wx(x) + self.wg(g), inplace=True))
        return x * alpha, alpha


def _up(x, ref):
    return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)


class UNetPP(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, channels=(32, 64, 128, 256, 512),
                 attention=True, msf=True):
        super().__init__()
        self.L = L = len(channels)
        self.ch = ch = list(channels)
        self.attention, self.msf = attention, msf
        self.pool = nn.MaxPool2d(2)
        self.nodes = nn.ModuleDict()
        self.gates = nn.ModuleDict()
        for i in range(L):
            self.nodes[f"{i}_0"] = ConvBlock(in_ch if i == 0 else ch[i - 1], ch[i])
        for j in range(1, L):
            for i in range(L - j):
                self.nodes[f"{i}_{j}"] = ConvBlock(ch[i] * j + ch[i + 1], ch[i])
                if attention:  # gate on the immediately preceding same-level node X(i, j-1)
                    self.gates[f"{i}_{j}"] = AttentionGate(ch[i], ch[i + 1], max(ch[i] // 2, 8))
        if msf:
            # decoder outputs at four scales: X(0,4) full, X(1,3) 1/2, X(2,2) 1/4, X(3,1) 1/8
            self.msf_src = [(k, L - 1 - k) for k in range(L - 1)]
            self.msf_reduce = nn.ModuleList(
                [nn.Sequential(nn.Conv2d(ch[i], ch[0], 1, bias=False), nn.BatchNorm2d(ch[0]),
                               nn.ReLU(inplace=True)) for i, _ in self.msf_src])
            self.msf_fuse = ConvBlock(ch[0] * len(self.msf_src), ch[0])
        self.head = nn.Conv2d(ch[0], out_ch, 1)
        self.last_alphas: dict[str, torch.Tensor] = {}

    def forward(self, x):
        X = {}
        for i in range(self.L):
            X[(i, 0)] = self.nodes[f"{i}_0"](x if i == 0 else self.pool(X[(i - 1, 0)]))
        self.last_alphas = {}
        for j in range(1, self.L):
            for i in range(self.L - j):
                g = X[(i + 1, j - 1)]
                prev = X[(i, j - 1)]
                if self.attention:
                    prev, a = self.gates[f"{i}_{j}"](prev, g)
                    self.last_alphas[f"{i}_{j}"] = a.detach()
                skips = [X[(i, k)] for k in range(j - 1)] + [prev]
                X[(i, j)] = self.nodes[f"{i}_{j}"](torch.cat(skips + [_up(g, prev)], dim=1))
        top = X[(0, self.L - 1)]
        if self.msf:
            feats = [_up(red(X[src]), top) for red, src in zip(self.msf_reduce, self.msf_src)]
            top = self.msf_fuse(torch.cat(feats, dim=1))
        return self.head(top)


class UNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, channels=(32, 64, 128, 256, 512)):
        super().__init__()
        ch = list(channels)
        self.enc = nn.ModuleList([ConvBlock(in_ch if i == 0 else ch[i - 1], ch[i]) for i in range(len(ch))])
        self.dec = nn.ModuleList([ConvBlock(ch[i] + ch[i + 1], ch[i]) for i in range(len(ch) - 1)])
        self.pool = nn.MaxPool2d(2)
        self.head = nn.Conv2d(ch[0], out_ch, 1)

    def forward(self, x):
        skips = []
        for i, blk in enumerate(self.enc):
            x = blk(x if i == 0 else self.pool(x))
            skips.append(x)
        x = skips[-1]
        for i in reversed(range(len(self.dec))):
            x = self.dec[i](torch.cat([skips[i], _up(x, skips[i])], dim=1))
        return self.head(x)


SEG_VARIANTS = {
    "unet": dict(arch="unet"),
    "unetpp": dict(arch="unetpp", attention=False, msf=False),
    "unetpp_ag": dict(arch="unetpp", attention=True, msf=False),
    "unetpp_msf": dict(arch="unetpp", attention=False, msf=True),
    "unetpp_ag_msf": dict(arch="unetpp", attention=True, msf=True),
}


def build_seg_model(variant: str, channels=(32, 64, 128, 256, 512)) -> nn.Module:
    spec = SEG_VARIANTS[variant]
    if spec["arch"] == "unet":
        return UNet(channels=channels)
    return UNetPP(channels=channels, attention=spec["attention"], msf=spec["msf"])
