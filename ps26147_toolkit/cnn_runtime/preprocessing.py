"""Exact preprocessing and multi-view representation generation for CNN inference."""
from __future__ import annotations

from typing import Any
import numpy as np
from scipy.signal import hilbert, welch
from sklearn.preprocessing import StandardScaler

from ps26147_toolkit.feature_extractor import extract_features
from ps26147_toolkit.qam_candidate_features import extract_qam_candidate_features
from .labels import FEATURE_NAMES, INPUT_LENGTH

PSD_NPERSEG = 1024
PSD_NOVERLAP = 512
PSD_NFFT = 2048
PSD_EPSILON = 1e-30


def restore_feature_scaler(state: dict[str, Any]) -> StandardScaler:
    """Restore StandardScaler from checkpoint state dict without importing training code."""
    scaler = StandardScaler()
    scaler.mean_ = np.asarray(state["mean"], dtype=np.float64)
    scaler.scale_ = np.asarray(state["scale"], dtype=np.float64)
    scaler.var_ = np.asarray(state["var"], dtype=np.float64)
    scaler.n_samples_seen_ = np.asarray(state["n_samples_seen"])
    if scaler.n_samples_seen_.ndim == 0:
        scaler.n_samples_seen_ = int(scaler.n_samples_seen_)
    scaler.n_features_in_ = int(state["n_features_in"])
    return scaler


class PSDStandardizer:
    """Per-bin standardizer fitted only with training-set PSD rows."""
    def __init__(self, mean: np.ndarray, scale: np.ndarray):
        self.mean = np.asarray(mean, dtype=np.float64)
        self.scale = np.asarray(scale, dtype=np.float64)
        if self.mean.shape != (PSD_NFFT,) or self.scale.shape != (PSD_NFFT,):
            raise ValueError("PSD scaler must contain one value for each of 2048 bins")
        self.scale = np.where(self.scale < 1e-8, 1.0, self.scale)

    def transform(self, log_psd: np.ndarray) -> np.ndarray:
        value = (np.asarray(log_psd, dtype=np.float64) - self.mean) / self.scale
        return np.nan_to_num(value, nan=0.0, posinf=1e4, neginf=-1e4).astype(np.float32)

    @classmethod
    def from_state_dict(cls, state: dict) -> "PSDStandardizer":
        return cls(np.asarray(state["mean"]), np.asarray(state["scale"]))


def log_welch_psd(iq: np.ndarray, sample_rate_hz: float) -> np.ndarray:
    """Compute two-sided, Hann-window, fft-shifted log-Welch PSD."""
    if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
        raise ValueError(f"Invalid sample rate: {sample_rate_hz}")
    _, power = welch(np.asarray(iq, dtype=np.complex128), fs=float(sample_rate_hz),
                     window="hann", nperseg=PSD_NPERSEG, noverlap=PSD_NOVERLAP,
                     nfft=PSD_NFFT, detrend=False, return_onesided=False, scaling="density")
    shifted = np.fft.fftshift(power)
    return np.log(np.maximum(shifted, PSD_EPSILON)).astype(np.float32)


def crop_window_center(signal: np.ndarray, length: int = INPUT_LENGTH) -> np.ndarray:
    """Deterministic center-crop (matching validation/inference policy) or zero-pad to length."""
    x = np.asarray(signal)
    if not np.iscomplexobj(x):
        x = hilbert(x).astype(np.complex64)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.complex64, copy=False)
    if len(x) > length:
        max_start = len(x) - length
        start = max_start // 2  # deterministic center crop
    else:
        start = 0
    crop = x[start:start + length]
    if len(crop) < length:
        crop = np.pad(crop, (0, length - len(crop)))
    return crop.astype(np.complex64, copy=False)


def extract_22_features(crop: np.ndarray, sample_rate_hz: float) -> np.ndarray:
    """Extract repository 16-feature vector + exact six radial extras."""
    base = np.asarray(extract_features(crop, fs=float(sample_rate_hz)), dtype=np.float32)
    radial = extract_qam_candidate_features(crop)
    result = np.nan_to_num(np.concatenate((base, radial)).astype(np.float32),
                           nan=0.0, posinf=1e6, neginf=-1e6)
    if result.shape != (len(FEATURE_NAMES),):
        raise ValueError(f"Feature stack returned {result.shape}, expected ({len(FEATURE_NAMES)},)")
    return result


def joint_rms_normalize(crop: np.ndarray, length: int = INPUT_LENGTH) -> tuple[np.ndarray, np.ndarray]:
    """Pad/truncate then scale I and Q jointly using valid complex samples."""
    x = np.asarray(crop)
    if x.ndim != 1:
        raise ValueError("IQ crop must be one-dimensional")
    x = np.nan_to_num(x[:length], nan=0.0, posinf=0.0, neginf=0.0).astype(np.complex64, copy=False)
    valid = len(x)
    if valid < length:
        x = np.pad(x, (0, length - valid))
    channels = np.stack((x.real, x.imag)).astype(np.float32)
    rms = float(np.sqrt(np.mean(channels[:, :valid] ** 2))) if valid else 0.0
    if np.isfinite(rms) and rms > 1e-8:
        channels /= rms
        x = (x / rms).astype(np.complex64)
    return channels, x


def differential_phase(x: np.ndarray) -> np.ndarray:
    """Signed angle(x[n] conj(x[n-1])) padded by repeating first valid value."""
    z = np.asarray(x, dtype=np.complex64)
    if z.size <= 1:
        return np.zeros(z.size, dtype=np.float32)
    dphi = np.angle(z[1:].astype(np.complex128) * np.conj(z[:-1].astype(np.complex128)))
    return np.pad(dphi.astype(np.float32), (1, 0), mode="edge")


def build_representations(crop: np.ndarray, sample_rate_hz: float,
                          feature_scaler: StandardScaler,
                          psd_scaler: PSDStandardizer | None,
                          length: int = INPUT_LENGTH) -> dict[str, np.ndarray]:
    """Derive every branch from the same crop; exact match to training."""
    iq, normalized_x = joint_rms_normalize(crop, length)
    phase = np.angle(normalized_x)
    dphi = differential_phase(normalized_x)
    amplitude = np.clip(np.abs(normalized_x) / np.sqrt(2.0), 0.0, 6.0)
    polar = np.stack((amplitude, np.sin(phase), np.cos(phase), dphi / np.pi)).astype(np.float32)
    psk = np.stack((np.cos(2 * phase), np.sin(2 * phase),
                    np.cos(4 * phase), np.sin(4 * phase))).astype(np.float32)
    ifreq = (dphi / (2.0 * np.pi))[None, :].astype(np.float32)
    raw_features = extract_22_features(crop, sample_rate_hz)
    standardized = feature_scaler.transform(raw_features.reshape(1, -1))[0].astype(np.float32)
    log_psd = log_welch_psd(normalized_x, sample_rate_hz)
    psd = psd_scaler.transform(log_psd) if psd_scaler is not None else log_psd
    arrays = {
        "iq": iq, "polar": polar, "psk": psk, "ifreq": ifreq,
        "psd": psd[None, :].astype(np.float32), "features": standardized
    }
    for name, value in arrays.items():
        if not np.isfinite(value).all():
            raise ValueError(f"Non-finite values in {name} representation")
    return arrays
