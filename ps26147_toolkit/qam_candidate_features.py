"""Compact gain-invariant radial features for QAM-order experiments.

These functions are deliberately separate from ``feature_extractor`` so the
existing 16-feature API and all legacy model paths remain unchanged.
"""
from __future__ import annotations

import numpy as np


QAM_CANDIDATE_FEATURE_NAMES = (
    "radial_q90_over_q50",
    "radial_q50_over_q10",
    "radial_moment4_rmsnorm",
    "radial_moment6_rmsnorm",
    "radial_hist_entropy_8",
    "radial_midband_fraction_075_125",
)


def extract_qam_candidate_features(signal: np.ndarray) -> np.ndarray:
    """Return six compact gain-invariant radial distribution descriptors.

    The complex envelope is normalized by its RMS, so a constant receive gain
    does not alter the features. Radial statistics are invariant to constant
    phase rotation/CFO invariant because multiplying each sample by a unit
    phasor leaves its radius unchanged. AWGN, fading, pulse shaping, and crop
    composition can still change the radial distribution.
    """
    samples = np.asarray(signal)
    if samples.ndim != 1:
        raise ValueError("signal must be one-dimensional")
    if not np.iscomplexobj(samples):
        samples = samples.astype(np.float64).astype(np.complex128)
    radius = np.abs(samples).astype(np.float64, copy=False)
    if radius.size == 0:
        return np.zeros(len(QAM_CANDIDATE_FEATURE_NAMES), dtype=np.float32)
    rms = float(np.sqrt(np.mean(radius * radius)))
    if not np.isfinite(rms) or rms <= 1e-12:
        return np.zeros(len(QAM_CANDIDATE_FEATURE_NAMES), dtype=np.float32)
    normalized = radius / rms
    q10, q50, q90 = np.quantile(normalized, (0.10, 0.50, 0.90))
    moment4 = float(np.mean(normalized ** 4))
    moment6 = float(np.mean(normalized ** 6))

    # Fixed bins preserve a comparable descriptor across captures; entropy is
    # normalized to [0, 1]. This captures radial occupancy without symbol
    # recovery, at the cost of sensitivity to SNR/fading and pulse shaping.
    edges = np.asarray((0.0, 0.50, 0.75, 1.0, 1.25, 1.50, 1.75, 2.25, np.inf))
    counts, _ = np.histogram(normalized, bins=edges)
    probabilities = counts.astype(np.float64) / max(float(counts.sum()), 1.0)
    nonzero = probabilities[probabilities > 0]
    entropy = float(-np.sum(nonzero * np.log2(nonzero)) / np.log2(len(edges) - 1))
    midband = float(np.mean((normalized >= 0.75) & (normalized < 1.25)))
    values = np.asarray((
        q90 / max(float(q50), 1e-12),
        q50 / max(float(q10), 1e-12),
        moment4,
        moment6,
        entropy,
        midband,
    ), dtype=np.float64)
    return np.nan_to_num(values, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)
