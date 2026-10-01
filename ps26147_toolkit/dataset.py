"""Lazy IQ/WAV dataset and signal preparation for hybrid AMR."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import re

import numpy as np
import torch
from torch.utils.data import Dataset

from .feature_extractor import extract_features
from .preprocess import load_iq_with_sigmf, load_wav

SAMPLE_LENGTH = 4096
CLASSES = ("BPSK", "QPSK", "8PSK", "16QAM", "64QAM", "2FSK", "4FSK", "AM", "FM")
_LABEL_ALIASES = {c.lower(): c for c in CLASSES}


def label_from_path(path: str | Path) -> str | None:
    """Infer modulation labels from common filename or directory conventions."""
    p = Path(path)
    # Check only the filename and directory tokens, using whole-token/prefix
    # matches so a label such as AM cannot be inferred from a user's folder
    # name (for example ``C:\\Users\\amity``).
    for token in (p.stem, *reversed(p.parts[:-1])):
        normalized = re.sub(r"[^a-z0-9]", "", token.lower())
        for key, label in _LABEL_ALIASES.items():
            key_normalized = re.sub(r"[^a-z0-9]", "", key)
            if normalized == key_normalized or (token == p.stem and normalized.startswith(key_normalized)):
                return label
    return None


def load_signal(path: str | Path) -> tuple[np.ndarray, float]:
    """Load one supported IQ or WAV capture and return samples and sample rate."""
    p = Path(path)
    if p.suffix.lower() == ".wav":
        signal, meta = load_wav(p)
        return np.asarray(signal), float(meta.fs)
    # Generated captures carry the actual rate in SigMF. Keep the legacy rate
    # only for older files that genuinely have no companion metadata.
    signal, meta = load_iq_with_sigmf(p, fallback_fs=2_048_000.0)
    return np.asarray(signal), float(meta.fs)


def prepare_iq(signal: np.ndarray, length: int = SAMPLE_LENGTH) -> np.ndarray:
    """Convert complex samples to normalized float32 I/Q with shape (2, length)."""
    if length <= 0:
        raise ValueError("length must be positive")
    sig = np.asarray(signal)
    if sig.ndim != 1:
        raise ValueError("signal must be a one-dimensional real or complex array")
    if not np.iscomplexobj(sig):
        sig = np.asarray(sig, dtype=np.float32).astype(np.complex64)
    sig = np.nan_to_num(sig[:length], copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    channels = np.zeros((2, length), dtype=np.float32)
    n = min(len(sig), length)
    if n:
        channels[0, :n] = sig.real[:n]
        channels[1, :n] = sig.imag[:n]
        scale = float(np.sqrt(np.mean(channels[:, :n] ** 2)))
        if np.isfinite(scale) and scale > 1e-8:
            channels[:, :n] /= scale
    return channels


class IQModulationDataset(Dataset):
    """Lazy dataset yielding ``(iq_tensor, feature_tensor, label_index)``.

    Records stay on disk; only one capture is decoded and featurized per item,
    so large collections do not need to fit in host memory.
    """
    def __init__(self, records: list[dict[str, Any]], class_names: tuple[str, ...] = CLASSES,
                 sample_length: int = SAMPLE_LENGTH, strict: bool = True,
                 mode: str = "train", feature_scaler: Any | None = None):
        if mode not in {"train", "val", "test", "fixed"}:
            raise ValueError("mode must be one of: 'train', 'val', 'test', 'fixed'")
        if sample_length <= 0:
            raise ValueError("sample_length must be positive")
        self.records = records
        self.class_names = tuple(class_names)
        self.class_to_index = {name: i for i, name in enumerate(self.class_names)}
        self.sample_length = sample_length
        self.strict = strict
        self.mode = mode
        self.feature_scaler = feature_scaler
        # Exposed for diagnostics/sanity checks; DataLoader workers each own
        # their dataset copy, and crop selection itself uses worker-seeded RNG.
        self.last_crop_start: int | None = None
        for record in records:
            if record["label"] not in self.class_to_index:
                raise ValueError(f"Unknown label {record['label']!r}")

    def _crop(self, signal: np.ndarray) -> np.ndarray:
        """Select the mode-appropriate window and right-pad short captures."""
        n = len(signal)
        if n > self.sample_length:
            max_start = n - self.sample_length
            if self.mode == "train":
                start = int(torch.randint(max_start + 1, (1,)).item())
            else:
                start = max(0, (n - self.sample_length) // 2)
        else:
            start = 0
        self.last_crop_start = start
        window = signal[start:start + self.sample_length]
        if len(window) < self.sample_length:
            window = np.pad(window, (0, self.sample_length - len(window)))
        return np.asarray(window)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        record = self.records[index]
        signal, fs = load_signal(record["path"])
        if not np.iscomplexobj(signal):
            from scipy.signal import hilbert
            signal = hilbert(signal).astype(np.complex64)
        signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
        # Both model branches must observe the exact same capture window.
        window = self._crop(signal)
        features = np.asarray(extract_features(window, fs=fs), dtype=np.float32)
        if features.shape != (16,):
            raise ValueError(f"Expected 16 handcrafted features, received {features.shape}")
        features = np.nan_to_num(features, nan=0.0, posinf=100.0, neginf=-100.0)
        if self.feature_scaler is not None:
            features = np.asarray(self.feature_scaler.transform(features.reshape(1, -1))[0],
                                  dtype=np.float32)
        iq = prepare_iq(window, self.sample_length)
        return (torch.from_numpy(iq), torch.from_numpy(features),
                torch.tensor(self.class_to_index[record["label"]], dtype=torch.long))


def fit_feature_scaler(records: list[dict[str, Any]], sample_length: int = SAMPLE_LENGTH) -> Any:
    """Fit StandardScaler on deterministic center crops from training records.

    One crop per original training capture is a reproducible calibration set;
    validation/test records must never be passed here. Training Dataset items
    remain lazy and random-cropped after this scaler is frozen.
    """
    from sklearn.preprocessing import StandardScaler

    calibration = IQModulationDataset(records, sample_length=sample_length, mode="val")
    matrix = np.empty((len(calibration), 16), dtype=np.float32)
    for index in range(len(calibration)):
        _, features, _ = calibration[index]
        matrix[index] = features.numpy()
    return StandardScaler().fit(matrix)
