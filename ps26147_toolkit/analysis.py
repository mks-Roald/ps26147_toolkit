"""Canonical per-signal analysis shared by API endpoints."""
from dataclasses import dataclass, field
from typing import Any
import numpy as np
from scipy.signal import hilbert
from . import classifier, parameter_extractor
from .preprocess import trim_active_burst

# Model loading includes disk I/O and happens once per process. Reuse the same
# fitted detector for uploaded and streaming signal analysis.
_CLASSIFIER = classifier.ModulationClassifier()


@dataclass
class SignalAnalysis:
    raw_signal: np.ndarray
    analytic_signal: np.ndarray
    baseband_signal: np.ndarray
    sample_rate: float
    parameters: dict[str, Any]
    classification: dict[str, Any]
    preprocessing: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    demod_baseband: np.ndarray | None = None


def analyze_signal(signal: np.ndarray, fs: float, modulation: str | None = None,
                   include_diagnostics: bool = True, center_freq_hint: float | None = None,
                   trim_burst: bool = False) -> SignalAnalysis:
    raw = np.asarray(signal).copy()
    analysis_signal = raw
    burst_info = {"trimmed": False, "start": 0, "stop": len(raw), "original_length": len(raw), "contrast_db": 0.0}
    if trim_burst:
        analysis_signal, burst_info = trim_active_burst(raw)
        analysis_signal = np.asarray(analysis_signal).copy()
    analytic = analysis_signal.astype(np.complex64) if np.iscomplexobj(analysis_signal) else hilbert(analysis_signal).astype(np.complex64)
    dc_removed = analytic - np.mean(analytic) if analytic.size else analytic
    # Estimate physical parameters once, then use its carrier for the sole
    # conversion. The converter's fine correction is part of that operation.
    # If center_freq_hint is provided (e.g., from simulation config), use it
    # for downconversion instead of estimating from PSD, which has bias for
    # RRC-shaped signals at baseband.
    params = parameter_extractor.extract_signal_parameters(dc_removed, fs=fs, modulation=modulation)
    fc = float(center_freq_hint) if center_freq_hint is not None else float(params["center_frequency_hz"])
    bb = classifier.downconvert_baseband(dc_removed, fs=fs, fc=fc)
    demod_bb = classifier.downconvert_baseband(analytic, fs=fs, fc=fc) if include_diagnostics else None
    decision = _CLASSIFIER.predict_with_confidence(
        bb, fs=fs, fc=0.0, include_diagnostics=include_diagnostics
    )
    # Reject obvious out-of-domain captures (noise-like full-band spectra or
    # non-finite parameter estimates) instead of presenting a forced RF label
    # as a confident modulation decision. Explicit caller hints remain intact.
    if modulation is None and include_diagnostics:
        snr = float(params.get("snr_db", 0.0))
        bandwidth = float(params.get("bandwidth_hz", 0.0))
        if (len(analysis_signal) < 32 or not np.isfinite(snr) or not np.isfinite(bandwidth)
                or snr < 0.0 or (bandwidth >= 0.9 * float(fs) and snr < 3.0)):
            decision = dict(decision, modulation="UNKNOWN", confidence=0.0)
            params = dict(params, baud_rate=0.0, baud_rate_valid=False)
    chosen = modulation or decision["modulation"]
    if modulation:
        decision = dict(decision, modulation=modulation, confidence=1.0)
    return SignalAnalysis(raw, analytic, bb, float(fs), params, decision,
                          {"dc_removed": True, "baseband_conversion_count": 2 if demod_bb is not None else 1,
                           "burst": burst_info},
                          {"classifier": decision.get("diagnostics", {})},
                          demod_baseband=demod_bb)
