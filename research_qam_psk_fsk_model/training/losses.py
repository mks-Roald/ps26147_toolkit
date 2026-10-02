"""Numerically stable focal loss and configurable low-SNR QAM sample weights."""
from __future__ import annotations

import warnings
import torch
from torch import nn
from torch.nn import functional as F

from ..labels import CLASS_TO_INDEX


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        if gamma < 0:
            raise ValueError("gamma must be nonnegative")
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError("reduction must be none, mean, or sum")
        self.gamma, self.reduction = float(gamma), reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                sample_weights: torch.Tensor | None = None,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        per_sample_ce = F.cross_entropy(logits.float(), targets.long(), reduction="none")
        pt = torch.exp(-per_sample_ce).clamp(0.0, 1.0)
        loss = (1.0 - pt).pow(self.gamma) * per_sample_ce
        weights = torch.ones_like(loss)
        if sample_weights is not None:
            sw = torch.as_tensor(sample_weights, device=loss.device, dtype=loss.dtype).reshape(-1)
            if sw.numel() != loss.numel():
                raise ValueError("sample_weights must contain one value per sample")
            weights = weights * torch.nan_to_num(sw, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        if mask is not None:
            m = torch.as_tensor(mask, device=loss.device).reshape(-1).bool()
            if m.numel() != loss.numel():
                raise ValueError("mask must contain one value per sample")
            weights = weights * m.to(loss.dtype)
        weighted = torch.nan_to_num(loss * weights, nan=0.0, posinf=0.0, neginf=0.0)
        if self.reduction == "none":
            return weighted
        if self.reduction == "sum":
            return weighted.sum()
        # Divide by item count, not by sum(weights): sample weights represent
        # fractional loss contributions, including a true 0.0 hard mask.
        return weighted.sum() / max(weighted.numel(), 1)


def low_snr_qam_sample_weights(targets: torch.Tensor, snr_db: torch.Tensor,
                               low_snr_qam_weight: float = 0.25,
                               *, warn_missing: bool = True) -> torch.Tensor:
    if not 0.0 <= low_snr_qam_weight <= 1.0:
        raise ValueError("low_snr_qam_weight must be between 0 and 1")
    targets, snr = targets.long(), snr_db.to(targets.device, dtype=torch.float32)
    if targets.numel() != snr.numel():
        raise ValueError("targets and snr_db must have matching lengths")
    missing = ~torch.isfinite(snr)
    if warn_missing and bool(missing.any()):
        warnings.warn("SNR metadata missing for one or more samples; using loss weight 1.0 for them.",
                      RuntimeWarning, stacklevel=2)
    qam_ids = torch.tensor((CLASS_TO_INDEX["16QAM"], CLASS_TO_INDEX["64QAM"]), device=targets.device)
    is_qam = (targets[:, None] == qam_ids[None, :]).any(dim=1)
    low = torch.isfinite(snr) & (snr < 0.0) & is_qam
    return torch.where(low, torch.full_like(snr, low_snr_qam_weight), torch.ones_like(snr))
