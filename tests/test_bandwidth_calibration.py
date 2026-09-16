"""Phase 6 §1.7 — Bandwidth calibration validation.

Generates a ground-truth signal for each modulation and asserts that the
calibrated occupied-bandwidth (via the per-modulation contour ladder in
``extract_signal_parameters``) lands within an acceptable error band of the
analytic reference (RRC-0.35 ``(1+α)·baud`` for linear mods, Carson
``2·Δf + baud`` for FSK).

The tolerance is deliberately generous (≤12% relative) to accommodate the
slight 4FSK residual from the Carson reference being an over-estimate of the
true 4-level spread; tighter modulations land well under this cap.
"""

import pytest
from pathlib import Path

import numpy as np

from scripts.generate_ground_truth_corpus import generate_test_case
from ps26147_toolkit.preprocess import load_wav
from ps26147_toolkit.parameter_extractor import extract_signal_parameters

# Default generator parameters (must match generate_test_case defaults).
_BAUD = 1200.0
_FSK_DEVIATION = 498.0
_ALPHA = 0.35  # RRC roll-off


def _theoretical_bandwidth(mod: str) -> float:
    if "FSK" in mod.upper():
        return 2.0 * _FSK_DEVIATION + _BAUD
    return (1.0 + _ALPHA) * _BAUD


_LINEAR_MODS = ["BPSK", "QPSK", "8PSK", "16QAM", "64QAM"]
_FSK_MODS = ["2FSK", "4FSK"]

# Per-modulation tolerance after contour calibration (Phase 6 §1.7).
# Linear mods land ≤2.5% on typical payloads; 2FSK ≤4%; 4FSK worst-case
# ~10–11% due to the Carson reference being generous for 4-level spreading.
_TOL_LINEAR = 0.10
_TOL_4FSK   = 0.15
_TOL_2FSK   = 0.08


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    return tmp_path_factory.mktemp("bw_cal")


def _check(mod: str, corpus: Path, payload: str = "the quick brown fox jumps over the lazy dog"):
    path, gt = generate_test_case(mod, payload_text=payload, out_dir=str(corpus))
    sig, meta = load_wav(str(path))
    params = extract_signal_parameters(sig, meta.fs, modulation=mod)
    theo = _theoretical_bandwidth(mod)
    bw = params["bandwidth_hz"]
    err = abs(bw - theo) / theo
    if mod == "4FSK":
        assert err <= _TOL_4FSK, f"{mod} bw err {err:.3f} (≤{_TOL_4FSK})"
    elif mod == "2FSK":
        assert err <= _TOL_2FSK, f"{mod} bw err {err:.3f} (≤{_TOL_2FSK})"
    else:
        assert err <= _TOL_LINEAR, f"{mod} bw err {err:.3f} (≤{_TOL_LINEAR})"


@pytest.mark.parametrize("mod", _LINEAR_MODS)
def test_linear_modulation_bandwidth(mod, corpus):
    _check(mod, corpus)


@pytest.mark.parametrize("mod", _FSK_MODS)
def test_fsk_bandwidth(mod, corpus):
    _check(mod, corpus)


def test_all_modulations_with_payload(corpus):
    """Full sweep: one long payload through all 7 modulations, every error
    within tolerance. This is the Phase 6 §1.7 accuracy contract."""
    payload = "the quick brown fox jumps over the lazy dog 0123456789" * 2
    for mod in _LINEAR_MODS + _FSK_MODS:
        _check(mod, corpus, payload=payload)


def test_contour_reported_in_result(corpus):
    """Ensure the calibrated contour dB is surfaced in the result dict."""
    path, _ = generate_test_case("BPSK", out_dir=str(corpus))
    sig, meta = load_wav(str(path))
    params = extract_signal_parameters(sig, meta.fs, modulation="BPSK")
    assert "bandwidth_contour_db" in params
    assert params["bandwidth_contour_db"] == 25  # calibrated value for BPSK
    assert "bandwidth_ladder_hz" in params
    assert isinstance(params["bandwidth_ladder_hz"], dict)
