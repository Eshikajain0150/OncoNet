"""Loss functions (Eqs. 10-12, 14-17)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss_from_logits(logits, target, eps=1.0):
    p = torch.sigmoid(logits).flatten(1)
    t = target.flatten(1)
    inter = (p * t).sum(1)
    return (1 - (2 * inter + eps) / (p.sum(1) + t.sum(1) + eps)).mean()


class SegLoss(nn.Module):
    """L_seg = w_dice * L_Dice + w_bce * L_BCE  (Eq. 16; 0.5/0.5)."""

    def __init__(self, w_dice=0.5, w_bce=0.5):
        super().__init__()
        self.w_dice, self.w_bce = w_dice, w_bce

    def forward(self, logits, target):
        return (self.w_dice * dice_loss_from_logits(logits, target)
                + self.w_bce * F.binary_cross_entropy_with_logits(logits, target))


def weighted_soft_ce(logits, soft_targets, class_weights, sample_weights=None):
    """L_WCE = - sum_c w_c y_c log(y'_c)  (Eq. 15), averaged over the batch.
    Works with one-hot or MixUp/CutMix soft targets."""
    logp = F.log_softmax(logits, dim=1)
    per = -(soft_targets * class_weights.view(1, -1) * logp).sum(1)
    if sample_weights is not None:
        return (per * sample_weights).sum() / sample_weights.sum()
    return per.mean()


class SupConLoss(nn.Module):
    """Supervised contrastive loss (Khosla et al., 2020), temperature tau = 0.07 (Eq. 14).
    Anchors without any same-class positive in the batch are skipped."""

    def __init__(self, temperature=0.07):
        super().__init__()
        self.t = temperature

    def forward(self, features, labels):
        z = F.normalize(features, dim=1)
        B = z.shape[0]
        sim = z @ z.t() / self.t
        self_mask = torch.eye(B, dtype=torch.bool, device=z.device)
        sim = sim.masked_fill(self_mask, -1e9)
        pos = (labels.view(-1, 1) == labels.view(1, -1)) & ~self_mask
        log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
        n_pos = pos.sum(1)
        valid = n_pos > 0
        if not valid.any():
            return sim.new_zeros(())
        mean_log_pos = (log_prob * pos).sum(1)[valid] / n_pos[valid]
        return -mean_log_pos.mean()
