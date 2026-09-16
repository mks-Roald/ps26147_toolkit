"""
ps26147_toolkit/parameter_extractor.py
Phase 2 – Parametric Signal Estimation Calibration

SOP 2.1 – Center Frequency (fc) Estimation & Confidence
SOP 2.2 – Multi-Bandwidth (BW -3dB, -10dB, OBW 95%, OBW 99%) with Masked Noise Floor
SOP 2.3 – Calibrated In-Band Integrated SNR & M2M4 Split-Moment Estimator
SOP 2.4 – Baseband-Downconverted, Windowed Cyclic Transition Baud Rate Estimator
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
from scipy.signal import find_peaks, savgol_filter, hilbert


def estimate_center_frequency(
    freqs: np.ndarray,
    psd: np.ndarray,
    return_confidence: bool = False,
) -> Union[float, Tuple[float, float]]:
    """Estimate the center frequency robustly using smoothed PSD and spectral centroid around peak.

    Parameters
    ----------
    freqs : np.ndarray
        Array of frequency bins (in Hz).
    psd : np.ndarray
        Power Spectral Density values corresponding to `freqs`.
    return_confidence : bool, default=False
        If True, returns a tuple `(center_freq, confidence)` where confidence in [0.0, 1.0].

    Returns
    -------
    float or Tuple[float, float]
        Center frequency in Hz, and optionally confidence score in [0.0, 1.0].
    """
    if len(psd) == 0 or len(freqs) == 0:
        return (0.0, 0.0) if return_confidence else 0.0

    if len(psd) < 4:
        best_idx = int(np.argmax(psd))
        fc = float(freqs[best_idx])
        return (fc, 0.5) if return_confidence else fc

    # Smooth PSD to prevent noisy spikes from skewing peak location
    win_len = min(31, len(psd) - (1 if len(psd) % 2 == 0 else 0))
    if win_len >= 5:
        psd_smooth = savgol_filter(psd, win_len, 2)
    else:
        psd_smooth = np.copy(psd)

    # Ensure non-negative smoothed PSD
    psd_smooth = np.maximum(psd_smooth, 1e-15)

    peak_idx = int(np.argmax(psd_smooth))
    peak_val = psd_smooth[peak_idx]

    # Find continuous region within 3 dB (half power) of the peak
    half_power = peak_val * 0.5
    left = peak_idx
    while left > 0 and psd_smooth[left - 1] >= half_power:
        left -= 1
    right = peak_idx
    while right < len(psd_smooth) - 1 and psd_smooth[right + 1] >= half_power:
        right += 1

    # Weighted centroid in the half-power cluster
    weights = psd_smooth[left : right + 1]
    w_sum = float(np.sum(weights))
    if w_sum > 0:
        center_freq = float(np.sum(freqs[left : right + 1] * weights) / w_sum)
    else:
        center_freq = float(freqs[peak_idx])

    if not return_confidence:
        return center_freq

    # Calculate confidence based on Peak-to-Average Power Ratio (PAPR) and prominence
    mean_power = float(np.mean(psd_smooth)) + 1e-15
    papr_lin = peak_val / mean_power
    papr_db = 10.0 * np.log10(max(papr_lin, 1.0))
    # Map PAPR (0 to 20 dB) into confidence [0.0, 1.0]
    confidence = float(np.clip(1.0 - np.exp(-papr_db / 6.0), 0.0, 1.0))

    return center_freq, confidence


def _compute_masked_noise_floor(psd: np.ndarray, mask_ratio: float = 0.05) -> float:
    """Compute out-of-band noise floor by masking out dominant signal energy bins.

    Parameters
    ----------
    psd : np.ndarray
        Power Spectral Density values.
    mask_ratio : float
        Threshold ratio relative to peak power to classify in-band vs out-of-band bins.

    Returns
    -------
    float
        Noise floor power level.
    """
    if len(psd) == 0:
        return 1e-12

    peak_val = np.max(psd)
    if peak_val <= 0:
        return 1e-12

    # Signal mask: bins with power >= 5% (-13 dB) of peak
    signal_mask = psd >= (peak_val * mask_ratio)
    out_of_band = ~signal_mask

    # If at least 10% of bins are out of band, estimate noise floor from lower 25th percentile
    if np.sum(out_of_band) >= max(4, int(0.10 * len(psd))):
        noise_floor = float(np.percentile(psd[out_of_band], 25))
    else:
        # Broadband signal covering most bins: take bottom 10th percentile
        noise_floor = float(np.percentile(psd, 10))

    return max(noise_floor, 1e-15)


def estimate_bandwidth(freqs: np.ndarray, psd: np.ndarray, threshold_db: float = 10.0) -> float:
    """Estimate bandwidth using noise-floor compensated -10dB (or custom threshold) spectral contour.

    Parameters
    ----------
    freqs : np.ndarray
        Frequency array in Hz.
    psd : np.ndarray
        PSD values.
    threshold_db : float, default=10.0
        Threshold down from peak in dB (e.g. 3.0 for -3dB, 10.0 for -10dB).

    Returns
    -------
    float
        Occupied bandwidth in Hz.
    """
    if len(psd) < 2 or len(freqs) < 2:
        return 0.0

    noise_floor = _compute_masked_noise_floor(psd)
    psd_sub = np.maximum(psd - noise_floor, 1e-15)
    peak_val = np.max(psd_sub)

    thresh = peak_val * (10.0 ** (-threshold_db / 10.0))
    indices = np.where(psd_sub >= thresh)[0]

    if len(indices) == 0:
        return 0.0

    bw = abs(float(freqs[indices[-1]] - freqs[indices[0]]))
    # Ensure minimum non-zero bandwidth if single bin detected
    if bw == 0.0 and len(freqs) > 1:
        bw = abs(float(freqs[1] - freqs[0]))

    return bw


def estimate_bandwidth_all(freqs: np.ndarray, psd: np.ndarray) -> Dict[str, float]:
    """Compute comprehensive multi-bandwidth measurements per SOP 2.2:
    - BW -3dB (Half-power bandwidth)
    - BW -10dB (Mask bandwidth)
    - OBW 95% (95% fractional power bandwidth)
    - OBW 99% (99% fractional power bandwidth)

    Parameters
    ----------
    freqs : np.ndarray
        Frequency array in Hz.
    psd : np.ndarray
        Power Spectral Density values.

    Returns
    -------
    Dict[str, float]
        Dictionary containing 'bw_3db', 'bw_10db', 'obw_95', 'obw_99', 'noise_floor', 'confidence'.
    """
    if len(psd) < 2 or len(freqs) < 2:
        return {
            "bw_3db": 0.0,
            "bw_10db": 0.0,
            "obw_95": 0.0,
            "obw_99": 0.0,
            "noise_floor": 0.0,
            "confidence": 0.0,
        }

    noise_floor = _compute_masked_noise_floor(psd)
    psd_sub = np.maximum(psd - noise_floor, 1e-15)
    peak_val = np.max(psd_sub)

    # 1. BW -3dB
    thresh_3db = peak_val * 0.5
    idx_3db = np.where(psd_sub >= thresh_3db)[0]
    bw_3db = abs(float(freqs[idx_3db[-1]] - freqs[idx_3db[0]])) if len(idx_3db) > 0 else 0.0

    # 2. BW -10dB
    thresh_10db = peak_val * 0.10
    idx_10db = np.where(psd_sub >= thresh_10db)[0]
    bw_10db = abs(float(freqs[idx_10db[-1]] - freqs[idx_10db[0]])) if len(idx_10db) > 0 else 0.0

    # 3. Fractional Occupied Bandwidth (OBW 95% and 99%)
    # Sort frequencies and corresponding PSD in monotonic order
    sort_order = np.argsort(freqs)
    f_sorted = freqs[sort_order]
    p_sorted = psd_sub[sort_order]

    cum_power = np.cumsum(p_sorted)
    tot_power = cum_power[-1]

    if tot_power > 0:
        # 95% OBW: between 2.5% and 97.5% cumulative power
        idx_95_low = np.searchsorted(cum_power, 0.025 * tot_power)
        idx_95_high = min(len(f_sorted) - 1, np.searchsorted(cum_power, 0.975 * tot_power))
        obw_95 = abs(float(f_sorted[idx_95_high] - f_sorted[idx_95_low]))

        # 99% OBW: between 0.5% and 99.5% cumulative power
        idx_99_low = np.searchsorted(cum_power, 0.005 * tot_power)
        idx_99_high = min(len(f_sorted) - 1, np.searchsorted(cum_power, 0.995 * tot_power))
        obw_99 = abs(float(f_sorted[idx_99_high] - f_sorted[idx_99_low]))
    else:
        obw_95 = bw_10db
        obw_99 = bw_10db

    # Resolution floor
    bin_width = abs(float(freqs[1] - freqs[0])) if len(freqs) > 1 else 1.0
    bw_3db = max(bw_3db, bin_width) if len(idx_3db) > 0 else 0.0
    bw_10db = max(bw_10db, bin_width) if len(idx_10db) > 0 else 0.0
    obw_95 = max(obw_95, bin_width)
    obw_99 = max(obw_99, bin_width)

    # Confidence based on peak-to-noise ratio and bandwidth validity
    papr_lin = peak_val / (noise_floor + 1e-15)
    papr_db = 10.0 * np.log10(max(papr_lin, 1.0))
    confidence = float(np.clip(1.0 - np.exp(-papr_db / 8.0), 0.1, 1.0))

    return {
        "bw_3db": bw_3db,
        "bw_10db": bw_10db,
        "obw_95": obw_95,
        "obw_99": obw_99,
        "noise_floor": float(noise_floor),
        "confidence": confidence,
    }


# Upper bound for reported SNR (dB).  A reading at this ceiling means the true
# SNR is *at least* this high; callers comparing against it can surface a
# "≥80 dB (clipped)" reading instead of a pegged 50.00 dB.  80 dB sits just
# below the ~96 dB theoretical noise floor of full-scale 16-bit PCM, so a clean
# synthetic signal no longer saturates the estimator (Phase 6 §1.4).
_SNR_CLIP_CEILING_DB = 80.0
_SNR_CLIP_FLOOR_DB = -20.0


def estimate_snr_m2m4(signal: np.ndarray, constellation_type: str = "psk") -> Optional[float]:
    """M2M4 split-moment SNR estimator for complex baseband signals.

    Let M2 = E[|r|^2], M4 = E[|r|^4].
    For complex circular Gaussian noise and signal with constellation kurtosis ka:
    S^2 * (2 - ka) = 2*M2^2 - M4.

    Parameters
    ----------
    signal : np.ndarray
        Complex baseband signal.
    constellation_type : str, default='psk'
        One of 'psk' (ka=1.0), 'qam16' (ka=1.32), 'qam64' (ka=1.38).

    Returns
    -------
    Optional[float]
        Estimated SNR in dB, or None if moments are unphysical (e.g. noise dominated).
    """
    if len(signal) < 64:
        return None

    r = signal - np.mean(signal)
    r2 = np.abs(r) ** 2
    m2 = float(np.mean(r2))
    m4 = float(np.mean(r2 ** 2))

    if m2 <= 0:
        return None

    ka_map = {
        "psk": 1.0,
        "qam16": 1.32,
        "qam64": 1.38,
        "constant_modulus": 1.0,
    }
    ka = ka_map.get(constellation_type.lower(), 1.0)

    # 2*M2^2 - M4
    disc = 2.0 * (m2 ** 2) - m4
    denom = 2.0 - ka

    if disc <= 0 or denom <= 0:
        return None

    s_power = np.sqrt(disc / denom)
    n_power = m2 - s_power

    if n_power <= 0 or s_power <= 0:
        return 40.0  # High SNR condition

    snr_lin = s_power / n_power
    snr_db = float(10.0 * np.log10(snr_lin))
    return float(np.clip(snr_db, _SNR_CLIP_FLOOR_DB, _SNR_CLIP_CEILING_DB))


def estimate_snr(
    psd: np.ndarray,
    freqs: Optional[np.ndarray] = None,
    center_freq: Optional[float] = None,
    bandwidth: Optional[float] = None,
    signal: Optional[np.ndarray] = None,
    fs: Optional[float] = None,
) -> float:
    """Robust calibrated in-band integrated Signal-to-Noise Ratio (SNR) estimation in dB.

    Combines:
    1. Spectral Integration Method: P_signal / P_noise over in-band vs masked out-of-band regions.
    2. Optional M2M4 Split-Moment estimator when complex baseband signal is provided.

    Parameters
    ----------
    psd : np.ndarray
        Power Spectral Density array.
    freqs : Optional[np.ndarray]
        Frequency bin values in Hz.
    center_freq : Optional[float]
        Center frequency of the signal in Hz.
    bandwidth : Optional[float]
        Occupied bandwidth of the signal in Hz.
    signal : Optional[np.ndarray]
        Optional raw or baseband complex signal for M2M4 estimation.
    fs : Optional[float]
        Sample rate in Hz.

    Returns
    -------
    float
        Calibrated SNR in dB.  Upper bound is 80.0 dB; a value at the ceiling
        means the true SNR is *at least* that high (the two return paths and
        M2M4 clip constant below).

    The float is returned directly; callers that need to distinguish a clipped
    reading from a real one can compare against `_SNR_CLIP_CEILING_DB`.
    """
    if len(psd) == 0:
        return 0.0

    noise_floor = _compute_masked_noise_floor(psd)

    snr_spectral: Optional[float] = None

    if freqs is not None and center_freq is not None and bandwidth is not None and bandwidth > 0:
        # In-band mask
        half_bw = bandwidth / 2.0
        in_band = (freqs >= (center_freq - half_bw)) & (freqs <= (center_freq + half_bw))
        n_in_band = np.sum(in_band)

        if n_in_band > 0:
            in_band_psd = psd[in_band]
            total_in_band_power = float(np.sum(in_band_psd))
            noise_in_band_power = float(noise_floor * n_in_band)
            signal_power = max(total_in_band_power - noise_in_band_power, 1e-15)
            snr_lin = signal_power / max(noise_in_band_power, 1e-15)
            snr_spectral = float(10.0 * np.log10(max(snr_lin, 1e-3)))

    if snr_spectral is None:
        # Fallback when frequency grid/mask is not provided:
        # Integrate top 20% bins against masked noise floor
        k = max(1, int(len(psd) * 0.15))
        top_bins_power = float(np.mean(np.sort(psd)[-k:]))
        signal_power = max(top_bins_power - noise_floor, 1e-15)
        snr_lin = signal_power / noise_floor
        snr_spectral = float(10.0 * np.log10(max(snr_lin, 1e-3)))

    # M2M4 cross-validation if baseband signal is provided
    if signal is not None and len(signal) >= 128 and np.iscomplexobj(signal):
        snr_m2 = estimate_snr_m2m4(signal)
        if snr_m2 is not None:
            # Weighted average between spectral and moment estimators
            snr_val = 0.65 * snr_spectral + 0.35 * snr_m2
            return float(np.clip(snr_val, _SNR_CLIP_FLOOR_DB, _SNR_CLIP_CEILING_DB))

    return float(np.clip(snr_spectral, _SNR_CLIP_FLOOR_DB, _SNR_CLIP_CEILING_DB))


def estimate_baud_rate(
    signal: np.ndarray,
    fs: float,
    center_freq: float = 0.0,
    bandwidth: Optional[float] = None,
) -> float:
    """Robust symbol/baud rate (Rs) estimator for digital modulations (PSK, QAM, FSK, ASK).

    Implements SOP 2.4:
    1. Downconverts passband signal to complex baseband using center_freq.
    2. Constructs non-linear transition envelope:
       Delta s(t) = |d/dt |s(t)|| / std_mag + alpha * |unwrap(dphi/dt)| / std_phase
    3. Windowed (Hann) high-resolution FFT with zero-padding.
    4. Bounded harmonic peak search in [BW/10, BW] or [10, fs/2].
    5. Parabolic peak interpolation and cyclic autocorrelation lag verification.

    Parameters
    ----------
    signal : np.ndarray
        Signal samples (real or complex).
    fs : float
        Sample rate in Hz.
    center_freq : float, default=0.0
        Center frequency of the signal in Hz.
    bandwidth : Optional[float]
        Estimated bandwidth in Hz to bound the baud search.

    Returns
    -------
    float
        Estimated symbol/baud rate in Baud (Hz).
    """
    if len(signal) < 64 or fs <= 0:
        return 0.0

    max_samples = 65536
    sig_chunk = signal[:max_samples] if len(signal) > max_samples else signal

    # For real signals, obtain complex analytic signal via Hilbert transform
    if not np.iscomplexobj(sig_chunk):
        sig_chunk = hilbert(sig_chunk)

    # 1. Baseband downconversion
    if abs(center_freq) > 1e-3:
        t = np.arange(len(sig_chunk), dtype=np.float64) / fs
        sig_bb = sig_chunk * np.exp(-1j * 2.0 * np.pi * center_freq * t)
    else:
        sig_bb = sig_chunk

    # Remove DC component from baseband
    sig_bb = sig_bb - np.mean(sig_bb)

    # 2. Non-linear transition envelope signals
    mag = np.abs(sig_bb)
    mag_diff = np.abs(np.diff(mag))

    # Phase difference (detects PSK/FSK phase transitions)
    # Conjugate product gives the signed instantaneous frequency directly:
    #   dphi = angle(sig_bb[n] * conj(sig_bb[n-1]))
    # For continuous-phase FSK, dphi alternates between +Δf·Ts and −Δf·Ts at
    # the baud rate — that alternating SIGN is what encodes bit boundaries.
    # Taking |dphi| (the old code) collapsed it to a near-constant series and
    # destroyed the periodicity the baud search depends on (Phase 6 §1.5).
    # Instead, DIFF the signed frequency: an impulse appears exactly at each
    # bit transition and is zero elsewhere, giving a genuinely periodic signal
    # regardless of the data's bit pattern.
    conj_prod = sig_bb[1:] * np.conj(sig_bb[:-1])
    dphi = np.angle(conj_prod)
    phase_diff = np.abs(np.diff(dphi))
    # np.diff shrinks by 1; pad to align with mag_diff length for the sum below
    if len(phase_diff) > 0:
        phase_diff = np.concatenate([[0.0], phase_diff])

    std_m = float(np.std(mag_diff)) + 1e-12
    std_p = float(np.std(phase_diff)) + 1e-12

    # Combined transition indicator
    transition_signal = (mag_diff / std_m) + (phase_diff / std_p)
    transition_signal = transition_signal - np.mean(transition_signal)

    n = len(transition_signal)
    if n < 16:
        return 0.0

    # 3. Apply Hann window to eliminate spectral leakage
    window = np.hanning(n)
    windowed_signal = transition_signal * window

    # Zero-padded FFT for fine frequency interpolation
    n_fft = max(2048, 4 * int(2 ** np.ceil(np.log2(n))))
    fft_trans = np.fft.rfft(windowed_signal, n=n_fft)
    freqs_trans = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    psd_trans = np.abs(fft_trans) ** 2

    # 4. Search bounds for Baud Rate
    # Baud rate physically cannot exceed bandwidth (Nyquist criterion) or fs/2
    if bandwidth is not None and bandwidth > 0:
        min_baud = max(10.0, bandwidth * 0.05, fs * 0.001)
        max_baud = min(fs * 0.495, bandwidth * 1.5)
    else:
        min_baud = max(10.0, fs * 0.002)
        max_baud = fs * 0.495

    if min_baud >= max_baud:
        min_baud = 10.0
        max_baud = fs * 0.495

    # 5. Autocorrelation-based fundamental period search
    # Autocorrelation of transition envelope has its first peak at lag = fs / Rs
    autocorr = np.fft.irfft(psd_trans)[:n]
    # Normalize autocorrelation
    if autocorr[0] > 0:
        autocorr = autocorr / autocorr[0]

    min_lag = max(2, int(fs / max_baud))
    max_lag = min(n - 1, int(fs / min_baud))

    ac_baud: Optional[float] = None
    if max_lag > min_lag + 2:
        ac_search = autocorr[min_lag : max_lag + 1]
        ac_peaks, _ = find_peaks(ac_search, height=0.02, distance=max(2, int(min_lag * 0.5)))
        if len(ac_peaks) > 0:
            # First prominent lag peak corresponds to fundamental symbol period
            first_peak_idx = ac_peaks[0]
            # Parabolic interpolation on autocorrelation peak
            peak_lag_idx = min_lag + first_peak_idx
            if 0 < peak_lag_idx < len(autocorr) - 1:
                y0 = autocorr[peak_lag_idx]
                y_m = autocorr[peak_lag_idx - 1]
                y_p = autocorr[peak_lag_idx + 1]
                denom = 2.0 * (2.0 * y0 - y_m - y_p)
                d_lag = (y_m - y_p) / denom if abs(denom) > 1e-15 else 0.0
                refined_lag = peak_lag_idx + np.clip(d_lag, -0.5, 0.5)
            else:
                refined_lag = float(peak_lag_idx)

            if refined_lag > 1.0:
                cand_baud = fs / refined_lag
                if min_baud <= cand_baud <= max_baud:
                    ac_baud = float(cand_baud)

    # 6. Frequency-domain spectral peak search with harmonic resolution
    valid_mask = (freqs_trans >= min_baud) & (freqs_trans <= max_baud)
    if not np.any(valid_mask):
        return ac_baud if ac_baud is not None else 0.0

    valid_indices = np.where(valid_mask)[0]
    valid_psd = psd_trans[valid_indices]
    valid_freqs = freqs_trans[valid_indices]

    mean_val = float(np.mean(valid_psd))
    std_val = float(np.std(valid_psd))
    height_thresh = mean_val + 0.5 * std_val

    peaks, _ = find_peaks(valid_psd, height=height_thresh, distance=max(2, int(len(valid_psd) * 0.005)))

    fft_baud: Optional[float] = None
    if len(peaks) > 0:
        # Sort candidate peaks by power descending
        peak_powers = valid_psd[peaks]
        sorted_peak_order = np.argsort(-peak_powers)

        candidate_freqs = valid_freqs[peaks[sorted_peak_order]]
        candidate_powers = peak_powers[sorted_peak_order]
        top_freq = candidate_freqs[0]
        top_power = candidate_powers[0]

        # Check if there is a fundamental sub-harmonic (e.g. top_freq / 2 or top_freq / 3 or top_freq / 4)
        best_freq = top_freq
        for sub_div in [4, 3, 2]:
            sub_target = top_freq / sub_div
            if sub_target >= min_baud:
                # Look for matching peak near sub_target (within 4%)
                matches = np.where(np.abs(candidate_freqs - sub_target) <= 0.04 * sub_target)[0]
                if len(matches) > 0:
                    sub_idx = matches[0]
                    # If subharmonic has at least 15% power of top harmonic, it is the true fundamental
                    if candidate_powers[sub_idx] >= 0.15 * top_power:
                        best_freq = candidate_freqs[sub_idx]
                        break

        # Refine best_freq using parabolic interpolation in the un-windowed spectrum
        best_idx_in_valid = np.argmin(np.abs(valid_freqs - best_freq))
        global_peak_idx = valid_indices[best_idx_in_valid]

        if 0 < global_peak_idx < len(psd_trans) - 1:
            y0 = psd_trans[global_peak_idx]
            y_m = psd_trans[global_peak_idx - 1]
            y_p = psd_trans[global_peak_idx + 1]
            denom = 2.0 * (2.0 * y0 - y_m - y_p)
            if abs(denom) > 1e-15:
                delta = (y_m - y_p) / denom
                delta = np.clip(delta, -0.5, 0.5)
                df = freqs_trans[1] - freqs_trans[0]
                fft_baud = float(freqs_trans[global_peak_idx] + delta * df)
            else:
                fft_baud = float(freqs_trans[global_peak_idx])
        else:
            fft_baud = float(freqs_trans[global_peak_idx])

    # If both autocorrelation and FFT estimates are available, cross-validate:
    if ac_baud is not None and fft_baud is not None:
        # If they agree within 5%, take FFT refined estimate (better frequency resolution)
        if abs(ac_baud - fft_baud) / max(ac_baud, fft_baud) < 0.06:
            return float(fft_baud)
        # If FFT picked a 2x harmonic of AC baud, trust AC baud
        if abs(fft_baud - 2.0 * ac_baud) / (2.0 * ac_baud) < 0.06:
            return float(ac_baud)
        # Otherwise prefer FFT estimate if strong, else AC baud
        return float(fft_baud)
    elif fft_baud is not None:
        return float(fft_baud)
    elif ac_baud is not None:
        return float(ac_baud)

    # Fallback to maximum power bin within search range
    max_idx = int(np.argmax(valid_psd))
    return float(valid_freqs[max_idx])


# ---------------------------------------------------------------------------
# Calibrated occupied-bandwidth estimation (Phase 6 §1.7)
#
# The reported `bandwidth_hz` used to be a single −10 dB-from-peak contour,
# which *underestimates* true occupied bandwidth for every modulation.  A
# single contour cannot be accurate for both shaped-linear carriers (RRC,
# occupied width (1+α)·baud) and direct-keyed FSK tones (Carson 2·Δf+baud),
# so the estimator runs a *ladder* of contours and selects the one calibrated
# to each modulation class.
# ---------------------------------------------------------------------------

# Candidate contours (dB down from the compensated spectral peak) to measure.
# Contours below ~ −35 dB are excluded: on short/typical signals the estimate
# over-reads from spectral leakage and diverges from the true occupied width.
_BW_CONTOURS = (10, 15, 20, 25, 30)

# Calibrated contour per modulation class, chosen to minimise relative error
# against the corpus ground truth.  All shaped-linear modulations and 4FSK
# sit at −25 dB (the leak-robust edge); 2FSK needs −20 dB (its tone spread).
_BW_CONTOUR_DB = {
    "BPSK": 25,
    "QPSK": 25,
    "8PSK": 25,
    "16QAM": 25,
    "64QAM": 25,
    "2FSK": 20,
    "4FSK": 25,
}
# Fallback contour used when the modulation is unknown.
_DEFAULT_BW_CONTOUR_DB = 25


def _canonical_class(modulation: Optional[str]) -> str:
    """Normalise a modulation name to a calibration key, or '' if unknown."""
    if not modulation:
        return ""
    m = modulation.upper().replace(" ", "").replace("-", "")
    return m if m in _BW_CONTOUR_DB else ""


def estimate_bandwidth_ladder(
    freqs: np.ndarray,
    psd: np.ndarray,
    contours: tuple[int, ...] = _BW_CONTOURS,
) -> Dict[int, float]:
    """Measure occupied bandwidth at each dB-from-peak contour in *contours*.

    Uses the same noise-floor-compensated magnitude contour as
    :func:`estimate_bandwidth`.  Returns ``{contour_db: bandwidth_hz}`` where
    ``contour_db`` is positive (down from the peak, e.g. ``20`` == −20 dB).
    """
    if len(psd) < 2 or len(freqs) < 2:
        return {db: 0.0 for db in contours}

    noise_floor = _compute_masked_noise_floor(psd)
    psd_sub = np.maximum(psd - noise_floor, 1e-15)
    peak_val = np.max(psd_sub)
    bin_width = abs(float(freqs[1] - freqs[0])) if len(freqs) > 1 else 1.0

    out: Dict[int, float] = {}
    for db in contours:
        thresh = peak_val * (10.0 ** (-db / 10.0))
        indices = np.where(psd_sub >= thresh)[0]
        if len(indices) == 0:
            out[db] = bin_width
        else:
            bw = abs(float(freqs[indices[-1]] - freqs[indices[0]]))
            out[db] = max(bw, bin_width)
    return out


def select_bandwidth_contour(modulation: Optional[str]) -> int:
    """Return the calibrated contour dB for a modulation class (positive dB)."""
    cls = _canonical_class(modulation)
    return int(_BW_CONTOUR_DB[cls]) if cls else _DEFAULT_BW_CONTOUR_DB


def extract_signal_parameters(
    signal: np.ndarray,
    fs: float,
    nperseg: int = 1024,
    modulation: Optional[str] = None,
) -> Dict[str, Any]:
    """High-level Phase 2 pipeline orchestrator to extract all physical signal parameters.

    Parameters
    ----------
    signal : np.ndarray
        Raw signal samples.
    fs : float
        Sampling frequency in Hz.
    nperseg : int, default=1024
        FFT segment length for Welch PSD computation.
    modulation : str, optional
        Known modulation class (e.g. ``"BPSK"``).  When provided the bandwidth
        estimator selects the contour dB that yields the lowest relative error
        for that modulation class (Phase 6 §1.7 calibration).  When *None*,
        falls back to a fixed −25 dB contour that is conservative but
        reasonably accurate across all modulations tested.

    Returns
    -------
    Dict[str, Any]
        Dictionary of extracted parameters:
        - 'center_frequency_hz': float
        - 'center_frequency_confidence': float
        - 'bandwidth_hz': float — occupied bandwidth from the calibrated contour
        - 'bandwidth_contour_db': int — dB contour used (negative sign implied)
        - 'bandwidth_3db_hz': float
        - 'obw_95_hz': float
        - 'obw_99_hz': float
        - 'snr_db': float
        - 'snr_clipped': bool (True when snr_db is pegged at the ~80 dB ceiling)
        - 'baud_rate': float
        - 'noise_floor': float
    """
    from .feature_extractor import compute_psd

    nperseg = min(nperseg, max(16, len(signal)))
    freqs, psd = compute_psd(signal, fs, nperseg=nperseg)

    fc, fc_conf = estimate_center_frequency(freqs, psd, return_confidence=True)
    bw_info = estimate_bandwidth_all(freqs, psd)
    contour_db = select_bandwidth_contour(modulation)
    ladder = estimate_bandwidth_ladder(freqs, psd)
    bw = ladder[contour_db]

    snr = estimate_snr(psd, freqs=freqs, center_freq=fc, bandwidth=bw, signal=signal, fs=fs)
    baud = estimate_baud_rate(signal, fs, center_freq=fc, bandwidth=bw)

    return {
        "center_frequency_hz": float(fc),
        "center_frequency_confidence": float(fc_conf),
        "bandwidth_hz": float(bw),
        "bandwidth_contour_db": int(contour_db),
        "bandwidth_ladder_hz": {int(k): float(v) for k, v in ladder.items()},
        "bandwidth_3db_hz": float(bw_info["bw_3db"]),
        "obw_95_hz": float(bw_info["obw_95"]),
        "obw_99_hz": float(bw_info["obw_99"]),
        "snr_db": float(snr),
        "snr_clipped": bool(snr >= _SNR_CLIP_CEILING_DB),
        "baud_rate": float(baud),
        "noise_floor": float(bw_info["noise_floor"]),
    }
