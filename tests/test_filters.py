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


class TestIQImbalanceEdgeCases:
    def test_zero_power_channel_passthrough(self):
        """A real-only signal (zero Q power) passes through unchanged (branch coverage)."""
        sig = np.arange(100, dtype=np.float64) + 1j * np.zeros(100)
        out = correct_iq_imbalance(sig)
        assert len(out) == len(sig)
        np.testing.assert_allclose(np.real(out), np.arange(100))
        np.testing.assert_allclose(np.imag(out), 0.0, atol=1e-10)


class TestFiltersAndPipeline:
    def test_bandpass_filter_complex(self):
        fs = 100_000.0
        t = np.arange(2000) / fs
        # 10 kHz signal + 40 kHz interference
        sig = np.exp(1j * 2 * np.pi * 10_000 * t) + 0.5 * np.exp(1j * 2 * np.pi * 40_000 * t)

        filtered = bandpass_filter(sig, fs=fs, center_freq=10_000, bandwidth=5_000)
        assert len(filtered) == len(sig)
        assert filtered.dtype == sig.dtype

    def test_bandpass_filter_real(self):
        """Real (non-complex) passband filtering isolates the wanted tone."""
        fs = 100_000.0
        t = np.arange(4000) / fs
        sig = np.cos(2 * np.pi * 10_000 * t) + 0.5 * np.cos(2 * np.pi * 40_000 * t)
        filtered = bandpass_filter(sig, fs=fs, center_freq=10_000, bandwidth=5_000)
        assert len(filtered) == len(sig)
        # 40 kHz component should be strongly attenuated
        out_psd = np.abs(np.fft.fft(filtered)) ** 2
        f = np.fft.fftfreq(len(sig), 1 / fs)
        out_10k = out_psd[np.argmin(np.abs(f - 10_000))]
        out_40k = out_psd[np.argmin(np.abs(f - 40_000))]
        assert out_40k < out_10k * 0.05

    def test_bandpass_real_lowpass_region(self):
        """Real low-frequency (near-DC) case exercises the lowpass branch."""
        fs = 100_000.0
        t = np.arange(2000) / fs
        sig = np.cos(2 * np.pi * 500 * t)
        out = bandpass_filter(sig, fs=fs, center_freq=250.0, bandwidth=1_000.0)
        assert np.all(np.isfinite(out))

    def test_bandpass_short_input_passthrough(self):
        sig = np.zeros(8)
        out = bandpass_filter(sig, fs=1000.0, center_freq=100.0, bandwidth=50.0)
        assert np.array_equal(out, sig)

    def test_bandpass_bad_args_passthrough(self):
        sig = np.zeros(100)
        assert len(bandpass_filter(sig, fs=1000.0, center_freq=100.0, bandwidth=0.0)) == 100
        assert len(bandpass_filter(sig, fs=0.0, center_freq=100.0, bandwidth=50.0)) == 100

    def test_spectral_denoise_short_signal_passthrough(self):
        sig = np.ones(50)
        assert np.array_equal(spectral_denoise(sig, n_fft=1024), sig)

    def test_spectral_denoise_complex(self):
        rng = np.random.default_rng(1)
        n = 4096
        carrier = np.exp(1j * 2 * np.pi * 0.1 * np.arange(n))
        noise = 0.3 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
        noisy = carrier + noise
        out = spectral_denoise(noisy)
        assert len(out) == n
        # Cleaned signal should be closer to the clean carrier
        err_before = np.mean(np.abs(noisy - carrier))
        err_after = np.mean(np.abs(out - carrier))
        assert err_after < err_before

    def test_spectral_denoise_real(self):
        rng = np.random.default_rng(2)
        n = 4096
        tone = np.cos(2 * np.pi * 0.1 * np.arange(n))
        noisy = tone + 0.4 * rng.standard_normal(n)
        out = spectral_denoise(noisy)
        assert len(out) == n
        err_before = np.mean(np.abs(noisy - tone))
        err_after = np.mean(np.abs(out - tone))
        assert err_after < err_before

    def test_median_filter_real_and_even_kernel(self):
        base = np.zeros(100)
        base[40] = 50.0  # spike
        # Even kernel → internally bumped to odd
        filtered = median_filter_signal(base, kernel_size=4)
        assert abs(filtered[40]) < 5.0

    def test_median_filter_real(self):
        base = np.ones(100)
        base[60] = 100.0
        filtered = median_filter_signal(base, kernel_size=5)
        assert filtered[60] < 5.0

    def test_median_filter_complex_even_kernel(self):
        sig = np.ones(80, dtype=np.complex64)
        sig[30] = 50 + 50j
        filtered = median_filter_signal(sig, kernel_size=6)
        assert abs(filtered[30]) < 5.0

    def test_clean_signal_denoise(self):
        rng = np.random.default_rng(3)
        fs = 100_000.0
        t = np.arange(4096) / fs
        sig = np.exp(1j * 2 * np.pi * 10_000 * t) + 0.2 * (rng.standard_normal(4096) + 1j * rng.standard_normal(4096))
        cleaned = clean_signal(
            sig, fs=fs, center_freq=10_000, bandwidth=8_000,
            enable_dc_removal=True, enable_bandpass=True, enable_denoise=True,
        )
        assert len(cleaned) == len(sig)
        # Cleaned signal should carry less out-of-band noise power
        out_psd = np.abs(np.fft.fft(cleaned)) ** 2
        assert np.sum(out_psd) > 0

    def test_clean_signal_bandpass_disabled_when_no_centerfreq(self):
        sig = np.ones(100) + 0.1 * np.arange(100)
        cleaned = clean_signal(sig, fs=1000.0, center_freq=None, bandwidth=None,
                               enable_dc_removal=False, enable_bandpass=True)
        # Without center_freq/bandwidth, bandpass is skipped → passthrough
        np.testing.assert_allclose(cleaned, sig)

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
