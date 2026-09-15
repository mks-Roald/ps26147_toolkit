"""Unit tests for feature extractor and Higher-Order Cumulants (HOC) computation."""

import numpy as np
import pytest
import matplotlib.pyplot as plt

from ps26147_toolkit.feature_extractor import (
    compute_cumulants,
    extract_features,
    extract_instantaneous_features,
    extract_spectral_features,
    rrc_filter,
    compute_psd,
    compute_spectrogram,
    plot_spectrogram,
    plot_constellation,
)


def test_rrc_filter():
    """Test Root-Raised Cosine (RRC) pulse filter properties."""
    num_taps = 49
    alpha = 0.35
    sps = 8
    h = rrc_filter(num_taps=num_taps, alpha=alpha, sps=sps)

    assert len(h) == num_taps
    # Energy should be normalized to 1.0
    assert np.isclose(np.sum(h ** 2), 1.0, atol=1e-4)
    # Filter should be symmetric around the center tap
    center = num_taps // 2
    assert np.allclose(h[:center], h[center + 1:][::-1], atol=1e-5)


def test_cumulants_ground_truth_bpsk():
    """Verify BPSK theoretical cumulants:
    C20 = 1.0, C40 = -2.0, C42 = -2.0, C60 = 16.0, C63 = 16.0.
    """
    np.random.seed(42)
    syms = np.random.choice([-1.0, 1.0], size=50000).astype(np.complex64)
    c = compute_cumulants(syms)

    assert np.isclose(np.abs(c["c20"]), 1.0, atol=0.05)
    assert np.isclose(c["c21"], 1.0, atol=0.05)
    assert np.isclose(c["c40"].real, -2.0, atol=0.05)
    assert np.isclose(c["c42"], -2.0, atol=0.05)
    assert np.isclose(c["c60"].real, 16.0, atol=0.5)
    assert np.isclose(c["c63"], 16.0, atol=0.5)


def test_cumulants_ground_truth_qpsk():
    """Verify QPSK theoretical cumulants:
    C20 = 0.0, |C40| = 1.0, C42 = -1.0, C60 = 0.0, C63 = 4.0.
    """
    np.random.seed(42)
    syms = (
        np.random.choice([-1.0, 1.0], size=50000)
        + 1j * np.random.choice([-1.0, 1.0], size=50000)
    ) / np.sqrt(2)
    c = compute_cumulants(syms)

    assert np.isclose(np.abs(c["c20"]), 0.0, atol=0.05)
    assert np.isclose(c["c21"], 1.0, atol=0.05)
    assert np.isclose(np.abs(c["c40"]), 1.0, atol=0.05)
    assert np.isclose(c["c42"], -1.0, atol=0.05)
    assert np.isclose(np.abs(c["c60"]), 0.0, atol=0.2)
    assert np.isclose(c["c63"], 4.0, atol=0.2)


def test_cumulants_ground_truth_8psk():
    """Verify 8PSK theoretical cumulants:
    C20 = 0.0, C40 = 0.0, C42 = -1.0, C60 = 0.0, C63 = 4.0.
    """
    np.random.seed(42)
    phases = np.random.choice(np.arange(8) * (2.0 * np.pi / 8.0), size=50000)
    syms = np.exp(1j * phases).astype(np.complex64)
    c = compute_cumulants(syms)

    assert np.isclose(np.abs(c["c20"]), 0.0, atol=0.05)
    assert np.isclose(c["c21"], 1.0, atol=0.05)
    assert np.isclose(np.abs(c["c40"]), 0.0, atol=0.05)
    assert np.isclose(c["c42"], -1.0, atol=0.05)
    assert np.isclose(np.abs(c["c60"]), 0.0, atol=0.2)
    assert np.isclose(c["c63"], 4.0, atol=0.2)


def test_cumulants_ground_truth_16qam():
    """Verify 16QAM theoretical cumulants:
    C20 = 0.0, C40 = -0.68, C42 = -0.68, C60 = 0.0, C63 = 2.08.
    """
    np.random.seed(42)
    grid = np.array([-3, -1, 1, 3])
    syms = (
        np.random.choice(grid, size=50000)
        + 1j * np.random.choice(grid, size=50000)
    ) / np.sqrt(10)
    c = compute_cumulants(syms)

    assert np.isclose(np.abs(c["c20"]), 0.0, atol=0.05)
    assert np.isclose(c["c21"], 1.0, atol=0.05)
    assert np.isclose(np.abs(c["c40"]), 0.68, atol=0.05)
    assert np.isclose(c["c42"], -0.68, atol=0.05)
    assert np.isclose(np.abs(c["c60"]), 0.0, atol=0.2)
    assert np.isclose(c["c63"], 2.08, atol=0.2)


def test_cumulants_ground_truth_64qam():
    """Verify 64QAM theoretical cumulants:
    C20 = 0.0, C40 = -0.62, C42 = -0.62, C60 = 0.0, C63 = 1.80.
    """
    np.random.seed(42)
    grid = np.array([-7, -5, -3, -1, 1, 3, 5, 7])
    syms = (
        np.random.choice(grid, size=50000)
        + 1j * np.random.choice(grid, size=50000)
    ) / np.sqrt(42)
    c = compute_cumulants(syms)

    assert np.isclose(np.abs(c["c20"]), 0.0, atol=0.05)
    assert np.isclose(c["c21"], 1.0, atol=0.05)
    assert np.isclose(np.abs(c["c40"]), 0.62, atol=0.05)
    assert np.isclose(c["c42"], -0.62, atol=0.05)
    assert np.isclose(np.abs(c["c60"]), 0.0, atol=0.2)
    assert np.isclose(c["c63"], 1.80, atol=0.2)


def test_cumulants_with_downconversion():
    """Test compute_cumulants with carrier frequency downconversion."""
    fs = 1e6
    fc = 100e3
    t = np.arange(10000) / fs
    syms = np.random.choice([-1.0, 1.0], size=10000).astype(np.complex64)
    # Modulate onto carrier fc
    sig_rf = syms * np.exp(2j * np.pi * fc * t)

    # Without fc, C20 should be near 0 due to rotation
    c_raw = compute_cumulants(sig_rf, fs=fs, fc=0.0)
    assert np.abs(c_raw["c20"]) < 0.2

    # With fc provided, C20 should be recovered close to 1.0
    c_baseband = compute_cumulants(sig_rf, fs=fs, fc=fc)
    assert np.isclose(np.abs(c_baseband["c20"]), 1.0, atol=0.1)


def test_extract_instantaneous_features():
    """Test instantaneous amplitude, phase, and frequency features."""
    fs = 1e6
    t = np.arange(10000) / fs
    # Pure tone
    sig = np.exp(2j * np.pi * 50e3 * t)
    feats = extract_instantaneous_features(sig, fs=fs)

    assert "gamma_max" in feats
    assert "sigma_aa" in feats
    assert "sigma_dp" in feats
    assert "sigma_af" in feats
    assert "fsk_persistence" in feats
    assert feats["sigma_aa"] < 0.05  # Constant envelope


def test_extract_spectral_features():
    """Test spectral entropy calculation."""
    fs = 1e6
    sig = np.random.randn(5000) + 1j * np.random.randn(5000)
    feats = extract_spectral_features(sig, fs=fs)

    assert "spec_entropy" in feats
    assert 0.0 <= feats["spec_entropy"] <= 1.0


def test_extract_features_shape_and_finite():
    """Test 15-dimensional feature vector extraction."""
    fs = 1e6
    sig = np.random.randn(2048) + 1j * np.random.randn(2048)
    feats = extract_features(sig, fs=fs)

    assert isinstance(feats, np.ndarray)
    assert feats.shape == (15,)
    assert np.all(np.isfinite(feats))
    assert not np.any(np.isnan(feats))


def test_psd_and_spectrogram():
    """Test PSD and spectrogram computation and plotting."""
    fs = 1e6
    t = np.arange(2048) / fs
    sig = np.sin(2 * np.pi * 50e3 * t)

    freqs, psd = compute_psd(sig, fs=fs, nperseg=256)
    assert len(freqs) == len(psd)
    assert len(freqs) > 0

    t_spec, f_spec, sxx_db = compute_spectrogram(sig, fs=fs, nperseg=128, noverlap=64)
    assert sxx_db.shape == (len(f_spec), len(t_spec))

    fig1 = plot_spectrogram(t_spec, f_spec, sxx_db)
    assert isinstance(fig1, plt.Figure)
    plt.close(fig1)

    fig2 = plot_constellation(sig[:100])
    assert isinstance(fig2, plt.Figure)
    plt.close(fig2)
