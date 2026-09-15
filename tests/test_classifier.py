"""Unit tests for Automatic Modulation Recognition (AMR) classifier."""

import numpy as np
import pytest
from pathlib import Path

from ps26147_toolkit.classifier import (
    ModulationClassifier,
    rule_based_classify,
    generate_synthetic_dataset,
    MODULATION_CLASSES,
)
from ps26147_toolkit.feature_extractor import rrc_filter


def generate_test_signal(mod: str, num_symbols: int = 512, sps: int = 8, fs: float = 1e6, snr_db: float = 20.0):
    """Generate a clean test signal with specified modulation."""
    n_samples = num_symbols * sps
    rrc = rrc_filter(num_taps=49, alpha=0.35, sps=sps)

    if mod == "BPSK":
        syms = np.random.choice([-1.0, 1.0], size=num_symbols)
        upsampled = np.zeros(n_samples, dtype=np.complex64)
        upsampled[::sps] = syms
        sig = np.convolve(upsampled, rrc, mode="same")
    elif mod == "QPSK":
        syms = (
            np.random.choice([-1.0, 1.0], size=num_symbols)
            + 1j * np.random.choice([-1.0, 1.0], size=num_symbols)
        ) / np.sqrt(2)
        upsampled = np.zeros(n_samples, dtype=np.complex64)
        upsampled[::sps] = syms
        sig = np.convolve(upsampled, rrc, mode="same")
    elif mod == "8PSK":
        phases = np.random.choice(np.arange(8) * (2.0 * np.pi / 8.0), size=num_symbols)
        syms = np.exp(1j * phases)
        upsampled = np.zeros(n_samples, dtype=np.complex64)
        upsampled[::sps] = syms
        sig = np.convolve(upsampled, rrc, mode="same")
    elif mod == "16QAM":
        grid = np.array([-3, -1, 1, 3])
        syms = (
            np.random.choice(grid, size=num_symbols)
            + 1j * np.random.choice(grid, size=num_symbols)
        ) / np.sqrt(10)
        upsampled = np.zeros(n_samples, dtype=np.complex64)
        upsampled[::sps] = syms
        sig = np.convolve(upsampled, rrc, mode="same")
    elif mod == "64QAM":
        grid = np.array([-7, -5, -3, -1, 1, 3, 5, 7])
        syms = (
            np.random.choice(grid, size=num_symbols)
            + 1j * np.random.choice(grid, size=num_symbols)
        ) / np.sqrt(42)
        upsampled = np.zeros(n_samples, dtype=np.complex64)
        upsampled[::sps] = syms
        sig = np.convolve(upsampled, rrc, mode="same")
    elif mod == "2FSK":
        bits = np.random.choice([0, 1], size=num_symbols)
        f_dev = 30000.0
        freqs = np.where(np.repeat(bits, sps) == 1, f_dev, -f_dev)
        phase = 2.0 * np.pi * np.cumsum(freqs) / fs
        sig = np.exp(1j * phase).astype(np.complex64)
    elif mod == "4FSK":
        symbols = np.random.choice([-3, -1, 1, 3], size=num_symbols)
        f_dev = 20000.0
        freqs = np.repeat(symbols, sps) * f_dev
        phase = 2.0 * np.pi * np.cumsum(freqs) / fs
        sig = np.exp(1j * phase).astype(np.complex64)
    elif mod == "AM":
        t = np.arange(n_samples) / fs
        mod_sig = 0.5 * np.cos(2 * np.pi * 1000 * t) + 0.3 * np.sin(2 * np.pi * 3000 * t)
        sig = (1.0 + 0.7 * mod_sig).astype(np.complex64)
    else:
        raise ValueError(f"Unknown modulation: {mod}")

    # Add AWGN
    sig_pwr = np.mean(np.abs(sig) ** 2)
    noise_pwr = sig_pwr / (10.0 ** (snr_db / 10.0))
    noise = np.sqrt(noise_pwr / 2.0) * (np.random.randn(len(sig)) + 1j * np.random.randn(len(sig)))
    return sig + noise


def test_rule_based_classifier_no_fsk_bug():
    """Verify that BPSK and QPSK signals are never falsely classified as FSK."""
    np.random.seed(42)
    fs = 1e6
    # Test multiple BPSK instances
    for _ in range(5):
        sig_bpsk = generate_test_signal("BPSK", fs=fs, snr_db=25.0)
        pred_bpsk = rule_based_classify(sig_bpsk, fs=fs)
        assert pred_bpsk != "2FSK" and pred_bpsk != "4FSK"
        assert pred_bpsk in ["BPSK", "QPSK"]

    # Test multiple QPSK instances
    for _ in range(5):
        sig_qpsk = generate_test_signal("QPSK", fs=fs, snr_db=25.0)
        pred_qpsk = rule_based_classify(sig_qpsk, fs=fs)
        assert pred_qpsk != "2FSK" and pred_qpsk != "4FSK"
        assert pred_qpsk in ["QPSK", "8PSK"]


def test_rule_based_fsk_classification():
    """Verify that 2FSK and 4FSK are detected by the rule-based classifier."""
    np.random.seed(42)
    fs = 1e6
    sig_2fsk = generate_test_signal("2FSK", fs=fs, snr_db=25.0)
    pred_2fsk = rule_based_classify(sig_2fsk, fs=fs)
    assert pred_2fsk in ["2FSK", "4FSK"]

    sig_4fsk = generate_test_signal("4FSK", fs=fs, snr_db=25.0)
    pred_4fsk = rule_based_classify(sig_4fsk, fs=fs)
    assert pred_4fsk in ["4FSK", "2FSK"]


def test_rule_based_am_classification():
    """Verify that AM signals are classified as AM."""
    np.random.seed(42)
    fs = 1e6
    sig_am = generate_test_signal("AM", fs=fs, snr_db=25.0)
    pred_am = rule_based_classify(sig_am, fs=fs)
    assert pred_am == "AM"


def test_synthetic_dataset_generator():
    """Test synthetic dataset generation function."""
    X, y = generate_synthetic_dataset(n_samples_per_class=10, fs=1e6, random_state=42)
    assert X.shape[0] == 80
    assert X.shape[1] == 15
    assert len(y) == 80
    assert set(np.unique(y)) == set(MODULATION_CLASSES)


def test_modulation_classifier_predict_and_confidence():
    """Test ModulationClassifier inference, probability output, and confidence score."""
    np.random.seed(42)
    clf = ModulationClassifier()
    assert clf.is_fitted

    fs = 1e6
    sig_bpsk = generate_test_signal("BPSK", fs=fs, snr_db=20.0)

    # 1. predict
    pred = clf.predict(sig_bpsk, fs=fs)
    assert isinstance(pred, str)
    assert pred in MODULATION_CLASSES

    # 2. predict_proba
    probs = clf.predict_proba(sig_bpsk, fs=fs)
    assert isinstance(probs, dict)
    assert len(probs) == len(MODULATION_CLASSES)
    assert np.isclose(sum(probs.values()), 1.0, atol=1e-3)

    # 3. predict_with_confidence
    diag = clf.predict_with_confidence(sig_bpsk, fs=fs)
    assert "modulation" in diag
    assert "confidence" in diag
    assert "probabilities" in diag
    assert "cumulants" in diag
    assert "features" in diag
    assert 0.0 <= diag["confidence"] <= 1.0


def test_modulation_classifier_serialization(tmp_path):
    """Test saving and loading trained ModulationClassifier."""
    clf = ModulationClassifier()
    model_file = tmp_path / "test_model.pkl"
    clf.save(str(model_file))
    assert model_file.exists()

    clf_loaded = ModulationClassifier(model_path=str(model_file))
    assert clf_loaded.is_fitted
    sig = generate_test_signal("QPSK", fs=1e6)
    pred = clf_loaded.predict(sig, fs=1e6)
    assert pred in MODULATION_CLASSES
