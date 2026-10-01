"""Research-only six-branch, seven-class AMR model."""
from __future__ import annotations

import torch
from torch import nn

from .encoders import IQEncoder, PolarEncoder, PSKEncoder, IFEncoder, PSDSpectralEncoder, FeatureEncoder
from .fusion import WeightedMultimodalFusion


class MultimodalAMR(nn.Module):
    def __init__(self, feature_count: int = 22, num_classes: int = 7,
                 iq_dim: int = 128, aux_dim: int = 64, fused_dim: int = 128,
                 dropout: float = 0.25):
        super().__init__()
        if (num_classes, feature_count, iq_dim, aux_dim, fused_dim) != (7, 22, 128, 64, 128):
            raise ValueError("Architecture contract is fixed to 7 classes, 22 features, 128/64-D branches, and 128-D fusion")
        self.iq_encoder = IQEncoder(iq_dim)
        self.polar_encoder = PolarEncoder(aux_dim)
        self.psk_encoder = PSKEncoder(aux_dim)
        self.if_encoder = IFEncoder(aux_dim)
        self.psd_encoder = PSDSpectralEncoder(aux_dim)
        self.feature_encoder = FeatureEncoder(feature_count, iq_dim)
        self.to_fused = nn.ModuleDict({
            "iq": nn.Identity(), "polar": nn.Linear(aux_dim, fused_dim),
            "psk": nn.Linear(aux_dim, fused_dim), "if": nn.Linear(aux_dim, fused_dim),
            "psd": nn.Linear(aux_dim, fused_dim), "features": nn.Identity(),
        })
        self.fusion = WeightedMultimodalFusion(fused_dim)
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(64, num_classes),
        )

    def encode(self, iq: torch.Tensor, polar: torch.Tensor, psk: torch.Tensor,
               ifreq: torch.Tensor, psd: torch.Tensor, features: torch.Tensor):
        branches = {
            "iq": self.to_fused["iq"](self.iq_encoder(iq)),
            "polar": self.to_fused["polar"](self.polar_encoder(polar)),
            "psk": self.to_fused["psk"](self.psk_encoder(psk)),
            "if": self.to_fused["if"](self.if_encoder(ifreq)),
            "psd": self.to_fused["psd"](self.psd_encoder(psd)),
            "features": self.to_fused["features"](self.feature_encoder(features)),
        }
        return branches

    def forward(self, iq: torch.Tensor, polar: torch.Tensor, psk: torch.Tensor,
                ifreq: torch.Tensor, psd: torch.Tensor, features: torch.Tensor,
                *, return_aux: bool = False):
        expected = ((iq, 2), (polar, 4), (psk, 4), (ifreq, 1), (psd, 1))
        for tensor, channels in expected:
            if tensor.ndim != 3 or tensor.shape[1] != channels:
                raise ValueError(f"Expected [B,{channels},L] input, got {tuple(tensor.shape)}")
            expected_length = 2048 if tensor is psd else 4096
            if tensor.shape[2] != expected_length:
                raise ValueError(f"Expected sequence length {expected_length}, got {tensor.shape[2]}")
        if features.ndim != 2 or features.shape[1] != 22:
            raise ValueError(f"Expected [B,22] features, got {tuple(features.shape)}")
        branches = self.encode(iq, polar, psk, ifreq, psd, features)
        fused, fusion_aux = self.fusion(branches)
        logits = self.classifier(fused)
        if not return_aux:
            return logits
        return logits, {"branches": branches, "fused": fused, **fusion_aux}


def count_parameters(model: nn.Module) -> dict[str, int]:
    groups = {
        "iq": model.iq_encoder,
        "polar": model.polar_encoder,
        "psk": model.psk_encoder,
        "if": model.if_encoder,
        "psd": nn.ModuleList([model.psd_encoder, model.to_fused["psd"]]),
        "features": nn.ModuleList([model.feature_encoder, model.to_fused["features"]]),
        "branch_projections": nn.ModuleList([model.to_fused[k] for k in ("polar", "psk", "if")]),
        "fusion": model.fusion,
        "classifier": model.classifier,
    }
    result = {name: sum(p.numel() for p in module.parameters() if p.requires_grad)
              for name, module in groups.items()}
    result["total"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return result
