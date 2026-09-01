import numpy as np
from scipy.signal import find_peaks, savgol_filter


def estimate_center_frequency(freqs: np.ndarray, psd: np.ndarray) -> float:
    """Estimate the center frequency robustly using smoothed PSD and spectral centroid around peak.
    Returns frequency in Hz.
    """
    if len(psd) < 4:
        return float(freqs[np.argmax(psd)]) if len(psd) > 0 else 0.0

    # Smooth PSD to prevent noisy spikes from skewing peak location
    win_len = min(31, len(psd) - (1 if len(psd) % 2 == 0 else 0))
    if win_len >= 5:
        psd_smooth = savgol_filter(psd, win_len, 2)
    else:
        psd_smooth = psd

    peak_idx = int(np.argmax(psd_smooth))
    peak_val = psd_smooth[peak_idx]

    # Find region within 3 dB of the peak
    half_power = peak_val * 0.5
    left = peak_idx
    while left > 0 and psd_smooth[left] >= half_power:
        left -= 1
    right = peak_idx
    while right < len(psd_smooth) - 1 and psd_smooth[right] >= half_power:
        right += 1

    # Weighted centroid in the peak cluster
    weights = psd_smooth[left : right + 1]
    w_sum = np.sum(weights)
    if w_sum > 0:
        center_freq = np.sum(freqs[left : right + 1] * weights) / w_sum
    else:
        center_freq = freqs[peak_idx]

    return float(center_freq)


def estimate_bandwidth(freqs: np.ndarray, psd: np.ndarray, threshold_db: float = 10.0) -> float:
    """Estimate bandwidth using noise-floor compensated -10dB / -3dB spectral contour.
    Returns bandwidth in Hz.
    """
    if len(psd) < 2:
        return 0.0

    noise_floor = np.percentile(psd, 25)
    psd_sub = np.maximum(psd - noise_floor, 1e-12)
    peak_val = np.max(psd_sub)

    # 10 dB down threshold from peak (or 0.1 of peak power)
    thresh = peak_val * (10.0 ** (-threshold_db / 10.0))
    indices = np.where(psd_sub >= thresh)[0]

    if len(indices) == 0:
        return 0.0

    # Continuous occupied span
    bw = abs(freqs[indices[-1]] - freqs[indices[0]])
    return float(bw)


def estimate_snr(
    psd: np.ndarray,
    freqs: np.ndarray = None,
    center_freq: float = None,
    bandwidth: float = None,
) -> float:
    """Robust in-band integrated Signal-to-Noise Ratio (SNR) estimation in dB.
    Calculates total in-band energy vs out-of-band noise floor power.
    """
    if len(psd) == 0:
        return 0.0

    # Noise floor: lower quartile of the spectrum (resilient against in-band signals)
    noise_floor = np.percentile(psd, 25)
    if noise_floor <= 0:
        noise_floor = np.median(psd)
    if noise_floor <= 0:
        noise_floor = 1e-12

    if freqs is not None and center_freq is not None and bandwidth is not None and bandwidth > 0:
        # In-band mask
        half_bw = bandwidth / 2.0
        in_band = (freqs >= (center_freq - half_bw)) & (freqs <= (center_freq + half_bw))
        if np.any(in_band):
            in_band_psd = psd[in_band]
            total_in_band_power = np.sum(in_band_psd)
            noise_in_band_power = noise_floor * len(in_band_psd)
            signal_power = max(total_in_band_power - noise_in_band_power, 1e-12)
            snr_lin = signal_power / max(noise_in_band_power, 1e-12)
            return float(np.clip(10.0 * np.log10(snr_lin), -30.0, 60.0))

    # Fallback when frequency grid is not provided:
    # Use top 20% bins vs bottom 25% noise floor
    k = max(1, int(len(psd) * 0.1))
    top_bins_power = np.mean(np.sort(psd)[-k:])
    signal_power = max(top_bins_power - noise_floor, 1e-12)
    snr_lin = signal_power / noise_floor
    return float(np.clip(10.0 * np.log10(snr_lin), -30.0, 60.0))


def estimate_baud_rate(
    signal: np.ndarray,
    fs: float,
    center_freq: float = 0.0,
    bandwidth: float = None,
) -> float:
    """Robust symbol/baud rate estimator for digital modulations (PSK, QAM, FSK, ASK).
    Uses baseband envelope transitions, phase derivatives, and cyclic spectral line analysis.
    """
    max_samples = 65536
    sig_chunk = signal[:max_samples] if len(signal) > max_samples else signal
    if len(sig_chunk) < 64 or fs <= 0:
        return 0.0

    # For real signals (like WAV passband signals), convert to complex analytic signal via Hilbert transform
    if not np.iscomplexobj(sig_chunk):
        from scipy.signal import hilbert
        sig_chunk = hilbert(sig_chunk)

    # Downconvert to baseband if center_freq is provided
    if abs(center_freq) > 0.01:
        t = np.arange(len(sig_chunk)) / fs
        sig_bb = sig_chunk * np.exp(-1j * 2 * np.pi * center_freq * t)
    else:
        sig_bb = sig_chunk

    # Nonlinear transition detection signals:
    # 1. Magnitude derivative (envelope transitions)
    mag = np.abs(sig_bb)
    mag_diff = np.abs(np.diff(mag))

    # 2. Instantaneous phase difference (detects PSK/FSK phase jumps)
    if np.iscomplexobj(sig_bb):
        # Angle of product with conjugate delay
        phase_diff = np.abs(np.angle(sig_bb[1:] * np.conj(sig_bb[:-1])))
    else:
        phase_diff = np.zeros_like(mag_diff)

    # Combined transition indicator
    transition_signal = mag_diff / (np.std(mag_diff) + 1e-12) + phase_diff / (np.std(phase_diff) + 1e-12)
    transition_signal = transition_signal - np.mean(transition_signal)

    # Compute FFT of transition signal to find cyclic symbol clock frequency
    n = len(transition_signal)
    fft_trans = np.fft.rfft(transition_signal, n=2 * n)
    freqs_trans = np.fft.rfftfreq(2 * n, d=1.0 / fs)
    psd_trans = np.abs(fft_trans) ** 2

    # Search window for baud rate:
    # Baud rate is physically bounded by Nyquist (fs/2) and signal bandwidth if known
    min_baud = max(10.0, (bandwidth * 0.05) if bandwidth and bandwidth > 0 else fs * 0.001)
    max_baud = min(fs * 0.49, (bandwidth * 1.5) if bandwidth and bandwidth > 0 else fs * 0.49)

    valid_mask = (freqs_trans >= min_baud) & (freqs_trans <= max_baud)
    if not np.any(valid_mask):
        return 0.0

    valid_freqs = freqs_trans[valid_mask]
    valid_psd = psd_trans[valid_mask]

    # Find prominent spectral peak in the transition spectrum
    peaks, properties = find_peaks(valid_psd, height=np.mean(valid_psd), distance=max(1, int(len(valid_psd) * 0.01)))
    if len(peaks) > 0:
        best_peak = peaks[np.argmax(valid_psd[peaks])]
        est_baud = valid_freqs[best_peak]
        return float(est_baud)

    # Fallback: Envelope autocorrelation
    autocorr = np.fft.irfft(psd_trans)[:n]
    ac_peaks, _ = find_peaks(autocorr[1:], height=0)
    if len(ac_peaks) > 0:
        first_lag = ac_peaks[0] + 1
        if first_lag > 1:
            est_baud = fs / first_lag
            if min_baud <= est_baud <= max_baud:
                return float(est_baud)

    return float(np.max(valid_freqs[np.argmax(valid_psd)]))
