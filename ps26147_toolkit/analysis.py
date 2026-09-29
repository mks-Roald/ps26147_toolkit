"""Canonical per-signal analysis shared by API endpoints."""
from dataclasses import dataclass, field
from typing import Any
import numpy as np
from scipy.signal import hilbert
from . import classifier, parameter_extractor


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


def analyze_signal(signal: np.ndarray, fs: float, modulation: str | None = None) -> SignalAnalysis:
    raw = np.asarray(signal).copy()
    analytic = raw.astype(np.complex64) if np.iscomplexobj(raw) else hilbert(raw).astype(np.complex64)
    dc_removed = analytic - np.mean(analytic) if analytic.size else analytic
    # Estimate physical parameters once, then use its carrier for the sole
    # conversion. The converter's fine correction is part of that operation.
    params = parameter_extractor.extract_signal_parameters(dc_removed, fs=fs, modulation=modulation)
    fc = float(params["center_frequency_hz"])
    bb = classifier.downconvert_baseband(dc_removed, fs=fs, fc=fc)
    clf = classifier.ModulationClassifier()
    decision = clf.predict_with_confidence(bb, fs=fs, fc=0.0)
    chosen = modulation or decision["modulation"]
    if modulation:
        decision = dict(decision, modulation=modulation, confidence=1.0)
    return SignalAnalysis(raw, analytic, bb, float(fs), params, decision,
                          {"dc_removed": True, "baseband_conversion_count": 1},
                          {"classifier": decision.get("diagnostics", {})})
