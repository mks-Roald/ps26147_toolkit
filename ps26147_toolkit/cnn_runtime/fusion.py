"""Softmax-normalized scalar modality weighting for 128-D branch embeddings."""
from __future__ import annotations

import torch
from torch import nn


class WeightedMultimodalFusion(nn.Module):
    NAMES = ("iq", "polar", "psk", "if", "psd", "features")

    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(len(self.NAMES)))
        self.norms = nn.ModuleDict({name: nn.LayerNorm(embedding_dim) for name in self.NAMES})

    def forward(self, branches: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if tuple(branches) != self.NAMES:
            raise ValueError(f"Expected branch order {self.NAMES}, got {tuple(branches)}")
        weights = torch.softmax(self.logits, dim=0)
        normalized = {name: self.norms[name](branches[name]) for name in self.NAMES}
        contributions = {name: weights[i] * normalized[name]
                         for i, name in enumerate(self.NAMES)}
        fused = sum(contributions.values())
        return fused, {"weights": weights, "contributions": contributions}
