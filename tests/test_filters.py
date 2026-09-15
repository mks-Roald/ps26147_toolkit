"""
tests/test_filters.py
Unit tests for filters and signal conditioning (SOP Phase 1.3).
"""

import numpy as np
import pytest

from ps26147_toolkit.filters import (
    remove_dc_offset,
    correct_iq_imbalance,
    bandpass_filter,
    spectral_denoise,
    median_filter_signal,
    clean_signal,
)


class TestDCOffsetRemoval:
    def test_complex_dc_offset(self):
        rng = np.random.default_rng(42)
        n = 1000
        clean_sig = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        dc_bias = 3.5 + 2.1j
        noisy_sig = clean_sig + dc_bias

        corrected = remove_dc_offset(noisy_sig)
        assert abs(np.mean(corrected.real)) < 1e-6
        assert abs(np.mean(corrected.imag)) < 1e-6

    def test_real_dc_offset(self):
        rng = np.random.default_rng(42)
        n = 1000
        clean_sig = rng.standard_normal(n)
        noisy_sig = clean_sig + 5.0

        corrected = remove_dc_offset(noisy_sig)
        assert abs(np.mean(corrected)) < 1e-6


class TestIQImbalanceCorrection:
    def test_amplitude_and_phase_imbalance_recovery(self):
        rng = np.random.default_rng(42)
        n = 10000
        t = np.arange(n) / 1000.0

        # Ideal QPSK-like complex signal
        i_ideal = rng.choice([-1.0, 1.0], size=n)
        q_ideal = rng.choice([-1.0, 1.0], size=n)

        # Introduce IQ imbalance: Amplitude mismatch (gain = 1.3) and Phase mismatch (phi = 15 deg)
        gain_imbalance = 1.3
        phi_rad = np.radians(15.0)

        i_distorted = i_ideal
        q_distorted = gain_imbalance * (np.sin(phi_rad) * i_ideal + np.cos(phi_rad) * q_ideal)
        rx_signal = i_distorted + 1j * q_distorted

        # Apply correction
        corrected = correct_iq_imbalance(rx_signal)

        # Verify balanced power between I and Q
        power_i = np.mean(corrected.real ** 2)
        power_q = np.mean(corrected.imag ** 2)
        assert abs(power_i - power_q) / power_i < 0.05

        # Verify phase orthogonality: correlation between I and Q should be near 0
        corr_iq = np.mean(corrected.real * corrected.imag) / np.sqrt(power_i * power_q)
        assert abs(corr_iq) < 0.05

    def test_short_or_real_signal_passthrough(self):
        sig = np.array([1.0, 2.0, 3.0])
        assert np.array_equal(correct_iq_imbalance(sig), sig)


class TestFiltersAndPipeline:
    def test_bandpass_filter_complex(self):
        fs = 100_000.0
        t = np.arange(2000) / fs
        # 10 kHz signal + 40 kHz interference
        sig = np.exp(1j * 2 * np.pi * 10_000 * t) + 0.5 * np.exp(1j * 2 * np.pi * 40_000 * t)

        filtered = bandpass_filter(sig, fs=fs, center_freq=10_000, bandwidth=5_000)
        assert len(filtered) == len(sig)
        assert filtered.dtype == sig.dtype

    def test_median_filter(self):
        sig = np.ones(100, dtype=np.complex64)
        sig[50] = 100.0 + 100.0j  # Impulsive spike
        filtered = median_filter_signal(sig, kernel_size=5)
        assert abs(filtered[50]) < 5.0

    def test_clean_signal_pipeline(self):
        fs = 100_000.0
        t = np.arange(2000) / fs
        sig = (np.exp(1j * 2 * np.pi * 10_000 * t) + 2.0 + 3.0j).astype(np.complex64)

        cleaned = clean_signal(
            sig,
            fs=fs,
            center_freq=10_000,
            bandwidth=5_000,
            enable_dc_removal=True,
            enable_iq_imbalance_correction=True,
        )
        assert len(cleaned) == len(sig)
        assert abs(np.mean(cleaned.real)) < 1e-4
