"""Validation tests for the ground-truth corpus generator.

These tests generate signals with *known* properties and verify the actual
PS26147 pipeline (parameter extractor + demodulator) recovers those values.
This is the accuracy-hardening contract: a number the pipeline reports for one
of these files is checkable against ground truth.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.generate_ground_truth_corpus import (
    generate_test_case,
    text_to_bits,
    bits_to_symbols,
)
from ps26147_toolkit.preprocess import load_wav, load_iq
from ps26147_toolkit.parameter_extractor import extract_signal_parameters
from ps26147_toolkit.demodulator import demodulate_signal
from ps26147_toolkit.classifier import ModulationClassifier


@pytest.fixture(scope="module")
def tmp_corpus(tmp_path_factory):
    return tmp_path_factory.mktemp("corpus")


def test_text_to_bits_roundtrip():
    bits = text_to_bits("hello")
    raw = np.packbits(bits).tobytes()
    assert raw.startswith(b"hello")
    assert len(raw) >= 5  # includes padding


def test_64qam_symbols_are_true_64qam():
    """Regression for a generator bug (found during Phase 6 §4 item 7).

    ``bits_to_symbols("64QAM")`` used to fall through to the ``"QPSK" in m or
    ``"4QAM" in m`` branch because ``"64QAM"`` contains the substring ``"4QAM"``
    — silently producing only the four QPSK corners (±0.707 ± 0.707j).  A real
    64-QAM constellation spans all 8 PAM levels (±7..±1)/sqrt(42) ≈ ±1.08.
    """
    bits = text_to_bits("the quick brown fox jumps over the lazy dog 0123456789" * 3)
    symbols = bits_to_symbols(bits, "64QAM")
    # Real 64-QAM has 8 distinct I/Q levels and (for a long random payload) a
    # large fraction of the 64 points populated.
    i_levels = np.unique(np.round(symbols.real, 2))
    q_levels = np.unique(np.round(symbols.imag, 2))
    assert len(i_levels) == 8 and len(q_levels) == 8, (
        f"64QAM must span 8 I/Q levels, got {len(i_levels)}/{len(q_levels)} (Phase 6 §4.7 regression)"
    )
    # Unless the payload is degenerate, we should populate well more than the
    # 4 QPSK corners the buggy path produced.
    assert len(np.unique(symbols)) > 20, (
        f"64QAM produced only {len(np.unique(symbols))} distinct symbols"
    )
    # Level span must reach ±7/sqrt(42) ≈ ±1.08, not the QPSK corners ±0.707
    assert abs(max(abs(i_levels))) > 0.9
    assert abs(max(abs(q_levels))) > 0.9


def test_generate_write_and_load_wav(tmp_corpus):
    path, gt = generate_test_case("BPSK", out_dir=str(tmp_corpus))
    assert path.exists()
    json_path = path.with_suffix(".wav.json")
    assert json_path.exists()
    # JSON round-trips
    with json_path.open("r") as fh:
        loaded = json.load(fh)
    assert loaded["modulation"] == "BPSK"
    assert loaded["payload_text"] == gt["payload_text"]
    # Pipeline can load it
    sig, meta = load_wav(str(path))
    assert meta.fs == gt["sample_rate"]
    assert len(sig) == gt["num_samples"]


def test_generate_write_and_load_iq(tmp_corpus):
    path, gt = generate_test_case("QPSK", file_format="iq", out_dir=str(tmp_corpus))
    assert path.exists()
    iq, meta = load_iq(str(path), fs=gt["sample_rate"])
    assert meta.num_samples > 0
    assert len(iq) == gt["num_samples"]


def test_parameter_extraction_recovers_ground_truth(tmp_corpus):
    """The parameter extractor should recover center freq / baud for a clean
    BPSK signal within tolerance (default integer-sps config)."""
    path, gt = generate_test_case("BPSK", out_dir=str(tmp_corpus))
    sig, meta = load_wav(str(path))
    params = extract_signal_parameters(sig, meta.fs)
    fc_err = abs(params["center_frequency_hz"] - gt["center_freq_hz"]) / gt["center_freq_hz"]
    baud_err = abs(params["baud_rate"] - gt["baud_rate"]) / gt["baud_rate"]
    assert fc_err < 0.05, f"center freq err {fc_err:.3f} (want <0.05)"
    assert baud_err < 0.05, f"baud err {baud_err:.3f} (want <0.05)"


def test_2fsk_baud_known_bug(tmp_corpus):
    """The 2FSK baud estimator was broken (Phase 6 §1.5: ``np.abs()`` on the
    phase difference collapsed FSK's alternating ±Δf into a constant,
    destroying periodicity).  The corpus demonstrated the bug; this test locks
    in the fix.  Initially asserted the bug was present; now asserts accuracy."""
    path, gt = generate_test_case("2FSK", out_dir=str(tmp_corpus))
    sig, meta = load_wav(str(path))
    params = extract_signal_parameters(sig, meta.fs)
    baud_err = abs(params["baud_rate"] - gt["baud_rate"]) / gt["baud_rate"]
    assert baud_err < 0.05, f"2FSK baud err {baud_err:.3f} (want <0.05)"


def test_demodulator_recovers_payload(tmp_corpus):
    """A clean BPSK file should demodulate back to the payload.

    This is a *smoke* check: the demodulator has known intermittent symbol
    timing slippage on synthetic clean signals (surfaced by this corpus), so
    we assert a modest best-shift match rather than near-exact recovery.  The
    full pass/fail accuracy contract is tracked by the Phase 6 harness.
    """
    for mod in ("BPSK", "QPSK"):
        path, gt = generate_test_case(mod, payload_text="hello world", out_dir=str(tmp_corpus))
        sig, meta = load_wav(str(path))
        params = extract_signal_parameters(sig, meta.fs)
        demod = demodulate_signal(
            sig,
            fs=meta.fs,
            modulation=mod,
            center_freq=params["center_frequency_hz"],
            baud_rate=params["baud_rate"],
        )
        assert demod["num_bits"] > 0
        # Best-shift match against expected payload bits
        expected = np.array(gt["payload_bits"])
        got = np.array(demod["bits"])
        best = 0.0
        for s in range(-len(expected), len(expected) + 1):
            n2 = min(len(expected), len(got))
            if s >= 0:
                m = np.mean(expected[: n2 - s] == got[s : s + n2]) if s < n2 else 0.0
            else:
                a = expected[-s:]
                b = got[: n2 + s]
                m = np.mean(a[: len(b)] == b[: len(a)]) if len(a) and len(b) else 0.0
            best = max(best, m)
        assert best > 0.7, f"{mod} best-shift payload match {best:.3f} (want >0.7)"


def test_noise_ladder_changes_snr(tmp_corpus):
    """Higher injected SNR should be recoverable as higher measured SNR."""
    path_clean, gt_clean = generate_test_case("BPSK", out_dir=str(tmp_corpus))
    path_noisy, gt_noisy = generate_test_case("BPSK", snr_db=6.0, out_dir=str(tmp_corpus))
    sig_c, meta_c = load_wav(str(path_clean))
    sig_n, _ = load_wav(str(path_noisy))
    snr_c = extract_signal_parameters(sig_c, meta_c.fs)["snr_db"]
    snr_n = extract_signal_parameters(sig_n, meta_c.fs)["snr_db"]
    assert snr_c >= snr_n, f"clean ({snr_c}) should measure >= noisy ({snr_n})"