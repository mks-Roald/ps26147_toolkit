import numpy as np
from scipy.signal import find_peaks

def estimate_center_frequency(freqs: np.ndarray, psd: np.ndarray) -> float:
    """Estimate the center frequency as the frequency with maximum PSD.
    Returns frequency in Hz.
    """
    idx = np.argmax(psd)
    return freqs[idx]

def estimate_bandwidth(freqs: np.ndarray, psd: np.ndarray, threshold_ratio: float = 0.5) -> float:
    """Estimate bandwidth as the width of the region where PSD is above `threshold_ratio * peak`.
    Returns bandwidth in Hz.
    """
    peak = np.max(psd)
    thresh = peak * threshold_ratio
    indices = np.where(psd >= thresh)[0]
    if len(indices) == 0:
        return 0.0
    bw = freqs[indices[-1]] - freqs[indices[0]]
    return bw

def estimate_snr(psd: np.ndarray, signal_idx: int) -> float:
    """Simple SNR estimate: power at signal index vs median of noise floor.
    Returns SNR in dB.
    """
    signal_power = psd[signal_idx]
    noise_power = np.median(psd)
    if noise_power == 0:
        return np.inf
    return 10 * np.log10(signal_power / noise_power)

def estimate_baud_rate(signal: np.ndarray, fs: float) -> float:
    """Very rough baud‑rate estimate using FFT autocorrelation.
    Returns estimated symbols per second.
    """
    # Cap signal length to 65536 samples to ensure fast computation
    max_samples = 65536
    sig_chunk = signal[:max_samples] if len(signal) > max_samples else signal
    if len(sig_chunk) < 2:
        return 0.0

    n = len(sig_chunk)
    # FFT-based autocorrelation: O(N log N)
    f = np.fft.fft(sig_chunk - np.mean(sig_chunk), n=2*n)
    power = np.abs(f) ** 2
    autocorr = np.fft.ifft(power).real[:n]

    # Find first peak after zero lag
    peaks, _ = find_peaks(autocorr, height=0)
    if len(peaks) < 2:
        return 0.0
    period_samples = peaks[1] - peaks[0]
    if period_samples <= 0:
        return 0.0
    return float(fs / period_samples)
