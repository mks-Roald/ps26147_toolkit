import numpy as np
from scipy import signal as sp_signal


def remove_dc_offset(signal: np.ndarray) -> np.ndarray:
    """Remove DC bias / carrier leakage from the signal.
    Works for both complex IQ and real audio signals.
    """
    return signal - np.mean(signal)


def bandpass_filter(
    sig: np.ndarray,
    fs: float,
    center_freq: float,
    bandwidth: float,
    order: int = 4,
    margin_factor: float = 1.25,
) -> np.ndarray:
    """Apply an adaptive Butterworth bandpass/lowpass filter centered around `center_freq`
    with effective passband `bandwidth * margin_factor`.

    Supports both complex baseband/IF signals and real signals.
    """
    if len(sig) < 16 or bandwidth <= 0 or fs <= 0:
        return sig

    bw_filtered = min(bandwidth * margin_factor, fs * 0.95)
    cutoff = max(bw_filtered / 2.0, 1.0)
    nyquist = fs / 2.0

    if np.iscomplexobj(sig):
        # Shift signal to baseband (0 Hz center), apply lowpass, then shift back
        t = np.arange(len(sig)) / fs
        sig_bb = sig * np.exp(-1j * 2 * np.pi * center_freq * t)

        norm_cutoff = min(cutoff / nyquist, 0.99)
        sos = sp_signal.butter(order, norm_cutoff, btype="lowpass", output="sos")
        # Apply filter to real and imag channels independently
        filtered_real = sp_signal.sosfiltfilt(sos, sig_bb.real)
        filtered_imag = sp_signal.sosfiltfilt(sos, sig_bb.imag)
        sig_bb_filt = filtered_real + 1j * filtered_imag

        # Shift back up to original center frequency
        return sig_bb_filt * np.exp(1j * 2 * np.pi * center_freq * t)
    else:
        # Real signal: bandpass between low_f and high_f
        low_f = max(center_freq - cutoff, 1.0)
        high_f = min(center_freq + cutoff, nyquist * 0.99)

        if low_f >= high_f:
            low_f = max(1.0, high_f * 0.5)

        low_norm = low_f / nyquist
        high_norm = high_f / nyquist

        if low_norm <= 0.001:
            sos = sp_signal.butter(order, high_norm, btype="lowpass", output="sos")
        else:
            sos = sp_signal.butter(order, [low_norm, high_norm], btype="bandpass", output="sos")

        return sp_signal.sosfiltfilt(sos, sig)


def spectral_denoise(
    sig: np.ndarray,
    noise_reduction_factor: float = 2.0,
    n_fft: int = 1024,
) -> np.ndarray:
    """Spectral subtraction / soft-thresholding denoiser for noisy signals.
    Reduces stationary background noise floor.
    """
    if len(sig) < n_fft:
        return sig

    hop_length = n_fft // 4
    window = np.hanning(n_fft)

    # Compute STFT
    is_complex = np.iscomplexobj(sig)
    num_frames = (len(sig) - n_fft) // hop_length + 1
    if num_frames <= 0:
        return sig

    frames = np.array([sig[i * hop_length : i * hop_length + n_fft] * window for i in range(num_frames)])
    stft_matrix = np.fft.fft(frames, axis=1)

    mag = np.abs(stft_matrix)
    phase = np.angle(stft_matrix)

    # Estimate noise floor per frequency bin using lower quartile (25th percentile)
    noise_est = np.percentile(mag, 25, axis=0) * noise_reduction_factor

    # Spectral subtraction with noise floor clamp
    clean_mag = np.maximum(mag - noise_est, 0.05 * mag)

    # Reconstruct signal via IFFT and Overlap-Add
    clean_stft = clean_mag * np.exp(1j * phase)
    if not is_complex:
        clean_frames = np.fft.ifft(clean_stft, axis=1).real
    else:
        clean_frames = np.fft.ifft(clean_stft, axis=1)

    out_sig = np.zeros(len(sig), dtype=sig.dtype)
    window_norm = np.zeros(len(sig))

    for i in range(num_frames):
        start = i * hop_length
        out_sig[start : start + n_fft] += clean_frames[i]
        window_norm[start : start + n_fft] += window

    # Normalize overlapping windows
    nonzero = window_norm > 1e-6
    out_sig[nonzero] /= window_norm[nonzero]

    return out_sig


def median_filter_signal(sig: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Apply median filtering to suppress impulsive spikes/outliers."""
    if kernel_size % 2 == 0:
        kernel_size += 1
    if np.iscomplexobj(sig):
        from scipy.signal import medfilt
        filtered_real = medfilt(sig.real, kernel_size=kernel_size)
        filtered_imag = medfilt(sig.imag, kernel_size=kernel_size)
        return filtered_real + 1j * filtered_imag
    else:
        from scipy.signal import medfilt
        return medfilt(sig, kernel_size=kernel_size)


def clean_signal(
    sig: np.ndarray,
    fs: float,
    center_freq: float = None,
    bandwidth: float = None,
    enable_dc_removal: bool = True,
    enable_bandpass: bool = True,
    enable_denoise: bool = False,
) -> np.ndarray:
    """Convenience pipeline to clean and filter a raw noisy signal."""
    cleaned = np.copy(sig)

    if enable_dc_removal:
        cleaned = remove_dc_offset(cleaned)

    if enable_bandpass and center_freq is not None and bandwidth is not None and bandwidth > 0:
        cleaned = bandpass_filter(cleaned, fs, center_freq, bandwidth)

    if enable_denoise:
        cleaned = spectral_denoise(cleaned)

    return cleaned
