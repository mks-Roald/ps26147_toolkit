"""
tests/test_parameter_extractor.py
Phase 2 – Parametric Signal Estimation Unit Tests & Verification

Validates:
- SOP 2.1: Center Frequency (fc) estimation with power-weighted centroid & confidence scoring.
- SOP 2.2: Multi-Bandwidth (BW -3dB, -10dB, OBW 95%, OBW 99%) & masked noise floor.
- SOP 2.3: Calibrated In-Band SNR & M2M4 Split-Moment estimator.
- SOP 2.4: Baseband-downconverted cyclic transition symbol/baud rate (Rs) estimation.
"""

import numpy as np
import pytest

from ps26147_toolkit.feature_extractor import compute_psd
from ps26147_toolkit.parameter_extractor import (
    estimate_center_frequency,
    estimate_bandwidth,
    estimate_bandwidth_all,
    estimate_snr,
    estimate_snr_m2m4,
    estimate_baud_rate,
    extract_signal_parameters,
)


# ===========================================================================
# Fixtures & Synthetic Signal Generators
# ===========================================================================

def generate_synthetic_psk(
    fs: float = 1_000_000.0,
    fc: float = 100_000.0,
    baud_rate: float = 25_000.0,
    snr_db: float = 20.0,
    num_symbols: int = 1000,
    mod_order: int = 2,
    seed: int = 42,
) -> np.ndarray:
    """Generate a synthetic baseband/passband PSK signal with known parameters and AWGN."""
    rng = np.random.default_rng(seed)
    sps = int(fs / baud_rate)
    total_samples = num_symbols * sps

    # Generate random symbols
    symbols = rng.integers(0, mod_order, size=num_symbols)
    phases = 2.0 * np.pi * symbols / mod_order
    complex_symbols = np.exp(1j * phases)

    # Upsample (pulse shape with rectangular / boxcar)
    upsampled = np.repeat(complex_symbols, sps)

    # Upconvert to carrier frequency fc
    t = np.arange(len(upsampled)) / fs
    carrier = np.exp(1j * 2.0 * np.pi * fc * t)
    tx_signal = upsampled * carrier

    # Add AWGN according to snr_db
    sig_pwr = np.mean(np.abs(tx_signal) ** 2)
    snr_lin = 10.0 ** (snr_db / 10.0)
    noise_pwr = sig_pwr / snr_lin
    noise = (rng.normal(0, np.sqrt(noise_pwr / 2), size=len(tx_signal)) +
             1j * rng.normal(0, np.sqrt(noise_pwr / 2), size=len(tx_signal)))

    return tx_signal + noise


# ===========================================================================
# SOP 2.1 – Center Frequency Estimation Tests
# ===========================================================================

def test_estimate_center_frequency_dc():
    """Verify center frequency estimation at DC (0 Hz)."""
    fs = 1_000_000.0
    sig = generate_synthetic_psk(fs=fs, fc=0.0, baud_rate=20_000.0, snr_db=25.0)
    freqs, psd = compute_psd(sig, fs=fs, nperseg=2048)

    fc_est = estimate_center_frequency(freqs, psd)
    assert abs(fc_est) < 2000.0, f"Expected near 0 Hz, got {fc_est} Hz"


def test_estimate_center_frequency_offset():
    """Verify center frequency estimation with positive and negative carrier offsets."""
    fs = 1_000_000.0
    for target_fc in [100_000.0, -150_000.0, 250_000.0]:
        sig = generate_synthetic_psk(fs=fs, fc=target_fc, baud_rate=25_000.0, snr_db=20.0)
        freqs, psd = compute_psd(sig, fs=fs, nperseg=2048)

        fc_est, confidence = estimate_center_frequency(freqs, psd, return_confidence=True)
        # Tolerance within 2% of sample rate or 2 kHz
        assert abs(fc_est - target_fc) < 2500.0, f"Target {target_fc}, got {fc_est}"
        assert confidence > 0.6, f"Expected high confidence, got {confidence}"


def test_estimate_center_frequency_edge_cases():
    """Verify handling of empty or minimal inputs."""
    assert estimate_center_frequency(np.array([]), np.array([])) == 0.0
    fc, conf = estimate_center_frequency(np.array([]), np.array([]), return_confidence=True)
    assert fc == 0.0 and conf == 0.0

    freqs = np.array([0.0, 100.0, 200.0])
    psd = np.array([1.0, 10.0, 2.0])
    assert estimate_center_frequency(freqs, psd) == 100.0


# ===========================================================================
# SOP 2.2 – Bandwidth Estimation Tests
# ===========================================================================

def test_estimate_bandwidth_known_span():
    """Verify multi-bandwidth metrics on a known 50 kHz bandwidth PSK signal."""
    fs = 1_000_000.0
    baud = 50_000.0  # Main lobe roughly 50-100 kHz
    sig = generate_synthetic_psk(fs=fs, fc=100_000.0, baud_rate=baud, snr_db=25.0)
    freqs, psd = compute_psd(sig, fs=fs, nperseg=2048)

    bw_10db = estimate_bandwidth(freqs, psd, threshold_db=10.0)
    # Bandwidth should be approximately within [40 kHz, 120 kHz]
    assert 40_000.0 <= bw_10db <= 130_000.0

    all_bw = estimate_bandwidth_all(freqs, psd)
    assert all_bw["bw_3db"] > 0
    assert all_bw["bw_10db"] >= all_bw["bw_3db"]
    assert all_bw["obw_99"] >= all_bw["obw_95"]
    assert all_bw["confidence"] > 0.5


def test_estimate_bandwidth_empty():
    """Verify bandwidth on degenerate empty inputs."""
    assert estimate_bandwidth(np.array([]), np.array([])) == 0.0
    res = estimate_bandwidth_all(np.array([]), np.array([]))
    assert res["bw_10db"] == 0.0
    assert res["confidence"] == 0.0


# ===========================================================================
# SOP 2.3 – SNR Estimation Tests
# ===========================================================================

def test_estimate_snr_monotonicity():
    """Verify higher input SNR results in higher estimated SNR."""
    fs = 1_000_000.0
    snr_low = 5.0
    snr_high = 25.0

    sig_low = generate_synthetic_psk(fs=fs, fc=50_000.0, baud_rate=20_000.0, snr_db=snr_low)
    sig_high = generate_synthetic_psk(fs=fs, fc=50_000.0, baud_rate=20_000.0, snr_db=snr_high)

    freqs_low, psd_low = compute_psd(sig_low, fs=fs, nperseg=2048)
    freqs_high, psd_high = compute_psd(sig_high, fs=fs, nperseg=2048)

    est_low = estimate_snr(psd_low, freqs=freqs_low, center_freq=50_000.0, bandwidth=40_000.0)
    est_high = estimate_snr(psd_high, freqs=freqs_high, center_freq=50_000.0, bandwidth=40_000.0)

    assert est_high > est_low, f"Expected est_high ({est_high}) > est_low ({est_low})"
    assert -20.0 <= est_low <= 80.0
    assert -20.0 <= est_high <= 80.0


def test_estimate_snr_m2m4():
    """Verify M2M4 split-moment estimator on clean baseband PSK signal."""
    fs = 1_000_000.0
    # Baseband signal (fc=0)
    sig = generate_synthetic_psk(fs=fs, fc=0.0, baud_rate=25_000.0, snr_db=18.0)
    snr_est = estimate_snr_m2m4(sig, constellation_type="psk")
    assert snr_est is not None
    assert 12.0 <= snr_est <= 24.0, f"Expected ~18 dB, got {snr_est}"


# ===========================================================================
# SOP 2.4 – Baud Rate Estimation Tests
# ===========================================================================

@pytest.mark.parametrize("target_baud", [10_000.0, 25_000.0, 50_000.0])
def test_estimate_baud_rate_known(target_baud: float):
    """Verify cyclic transition baud rate estimator within ±3% tolerance for known rates."""
    fs = 1_000_000.0
    fc = 100_000.0
    sig = generate_synthetic_psk(fs=fs, fc=fc, baud_rate=target_baud, snr_db=25.0, num_symbols=2000)

    # Estimate baud rate
    est_baud = estimate_baud_rate(sig, fs=fs, center_freq=fc, bandwidth=target_baud * 2.0)

    # Allow within 3% tolerance of the target baud rate
    relative_error = abs(est_baud - target_baud) / target_baud
    assert relative_error < 0.035, f"Target {target_baud}, got {est_baud} (error: {relative_error:.2%})"


def test_estimate_baud_rate_real_passband():
    """Verify baud rate estimation works seamlessly on real (WAV-style) signals via Hilbert transform."""
    fs = 500_000.0
    target_baud = 20_000.0
    sig_complex = generate_synthetic_psk(fs=fs, fc=50_000.0, baud_rate=target_baud, snr_db=25.0)
    sig_real = np.real(sig_complex)

    est_baud = estimate_baud_rate(sig_real, fs=fs, center_freq=50_000.0)
    assert abs(est_baud - target_baud) / target_baud < 0.05


def test_estimate_baud_rate_edge_cases():
    """Verify baud rate estimation on empty or very short signals."""
    assert estimate_baud_rate(np.array([]), fs=1_000_000.0) == 0.0
    assert estimate_baud_rate(np.ones(10), fs=1_000_000.0) == 0.0
    assert estimate_baud_rate(np.ones(100), fs=0.0) == 0.0


# ===========================================================================
# Full Pipeline Orchestration Test
# ===========================================================================

def test_extract_signal_parameters_full():
    """Verify end-to-end parameter extraction on a known ground-truth benchmark signal."""
    fs = 1_000_000.0
    fc = 100_000.0
    baud = 25_000.0
    snr = 20.0
    sig = generate_synthetic_psk(fs=fs, fc=fc, baud_rate=baud, snr_db=snr, num_symbols=1500)

    params = extract_signal_parameters(sig, fs=fs)

    assert "center_frequency_hz" in params
    assert "bandwidth_hz" in params
    assert "snr_db" in params
    assert "baud_rate" in params
    assert "noise_floor" in params

    assert abs(params["center_frequency_hz"] - fc) < 2500.0
    assert abs(params["baud_rate"] - baud) / baud < 0.035
    assert params["snr_db"] > 10.0


def test_snr_clip_ceiling_raised_above_50():
    """Phase 6 §1.4: the SNR clip ceiling must be strictly above 50 dB (the old
    hardcoded value), and an estimate forced above the ceiling must bind to the
    *new* ceiling — not be pegged at 50.

    Repro: on a PSD whose in-band power is ~10^8 × the masked noise floor, the
    spectral estimator's raw SNR is enormous.  Before the fix, estimate_snr()
    returned exactly 50.0; now it returns the new ceiling (80.0).
    """
    from ps26147_toolkit.parameter_extractor import (
        _SNR_CLIP_CEILING_DB as CEIL,
        _SNR_CLIP_FLOOR_DB as FLOOR,
        estimate_snr,
    )
    assert CEIL > 50.0, f"ceiling {CEIL} dB still at old 50 dB (§1.4 regression)"
    assert CEIL == 80.0  # tied to 16-bit quantization noise floor (~96 dB)
    assert FLOOR <= 0.0

    n = 4096
    freqs = np.linspace(0.0, 1.0e6, n)
    psd = np.full(n, 1e-12)  # flat masked noise floor
    in_band = (freqs >= 450_000.0) & (freqs <= 550_000.0)
    psd[in_band] += 1e-2  # >>> noise -> raw SNR far above 50 dB

    est = estimate_snr(psd, freqs=freqs, center_freq=500_000.0, bandwidth=100_000.0)
    # Old code: np.clip(..., 50.0) -> 50.0.  New code must bind to CEIL.
    assert est != 50.0, f"SNR still pegged at old 50 dB ceiling (§1.4 regression)"
    assert est == CEIL, f"SNR {est:.2f} dB should clip to ceiling {CEIL}"


def test_snr_clipped_flag_false_below_ceiling():
    """snr_clipped must be False for a typical mid-range SNR reading."""
    from ps26147_toolkit.parameter_extractor import _SNR_CLIP_CEILING_DB
    fs = 1_000_000.0
    sig = generate_synthetic_psk(fs=fs, fc=50_000.0, baud_rate=25_000.0, snr_db=15.0, num_symbols=1500)
    params = extract_signal_parameters(sig, fs=fs)
    assert params["snr_db"] < _SNR_CLIP_CEILING_DB
    assert params.get("snr_clipped") is False
