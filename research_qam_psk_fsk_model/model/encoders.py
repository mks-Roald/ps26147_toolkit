"""Compact independent temporal/spectral encoders with global average pooling."""
from __future__ import annotations

import torch
from torch import nn


class ConvBlock1d(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1):
        super().__init__(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride,
                      padding=kernel_size // 2, bias=False),
            nn.BatchNorm1d(out_channels), nn.GELU(),
        )


class TemporalEncoder(nn.Module):
    """Three shallow convolutional blocks followed by adaptive average pooling."""
    def __init__(self, in_channels: int, embedding_dim: int):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock1d(in_channels, 32, 7), nn.MaxPool1d(2),
            ConvBlock1d(32, 64, 5), nn.MaxPool1d(2),
            ConvBlock1d(64, 96, 3, stride=2),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
        )
        self.projection = nn.Sequential(nn.Linear(96, embedding_dim), nn.LayerNorm(embedding_dim), nn.GELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(self.features(x))


class IQEncoder(TemporalEncoder):
    def __init__(self, embedding_dim: int = 128):
        nn.Module.__init__(self)
        # Preserve the current IQ branch's 2→32→64→128 convolution widths,
        # kernels, one max-pool, and global average pooling.
        self.features = nn.Sequential(
            nn.Conv1d(2,32,7,padding=3,bias=False), nn.BatchNorm1d(32), nn.ReLU(inplace=True),
            nn.Conv1d(32,64,5,padding=2,bias=False), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(64,128,3,padding=1,bias=False), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
        )
        self.projection = nn.Identity() if embedding_dim == 128 else nn.Sequential(
            nn.Linear(128, embedding_dim), nn.LayerNorm(embedding_dim), nn.GELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(self.features(x))


class PolarEncoder(TemporalEncoder):
    def __init__(self, embedding_dim: int = 64):
        super().__init__(4, embedding_dim)


class PSKEncoder(TemporalEncoder):
    def __init__(self, embedding_dim: int = 64):
        super().__init__(4, embedding_dim)


class IFEncoder(TemporalEncoder):
    def __init__(self, embedding_dim: int = 64):
        super().__init__(1, embedding_dim)


class PSDSpectralEncoder(nn.Module):
    """Small one-dimensional encoder for a normalized, shifted log-Welch PSD."""
    def __init__(self, embedding_dim: int = 64):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock1d(1, 16, 9), nn.MaxPool1d(4),
            ConvBlock1d(16, 32, 7), nn.MaxPool1d(4),
            ConvBlock1d(32, 64, 5),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
        )
        self.projection = nn.Sequential(nn.Linear(64, embedding_dim), nn.LayerNorm(embedding_dim), nn.GELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(self.features(x))


class FeatureEncoder(nn.Module):
    """Same 22→32→32 feature MLP width, followed by 32→128 projection."""
    def __init__(self, feature_count: int = 22, embedding_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(feature_count, 32), nn.ReLU(inplace=True),
                                 nn.Linear(32, 32), nn.ReLU(inplace=True))
        self.projection = nn.Sequential(nn.Linear(32, embedding_dim), nn.LayerNorm(embedding_dim), nn.GELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(self.mlp(x))
