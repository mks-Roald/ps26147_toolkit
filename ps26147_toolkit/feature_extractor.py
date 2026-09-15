"""Feature extraction and spectral analysis tools for PS26147 Toolkit."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, spectrogram, hilbert, medfilt


def rrc_filter(num_taps: int = 49, alpha: float = 0.35, sps: int = 8) -> np.ndarray:
    """Generate Root-Raised Cosine (RRC) pulse shaping filter coefficients.
    
    Parameters
    ----------
    num_taps : int
        Total number of filter taps (should be odd).
    alpha : float
        Roll-off factor (typically 0.2 to 0.5).
    sps : int
        Samples per symbol.
        
    Returns
    -------
    np.ndarray
        Normalized unit-energy RRC filter impulse response.
    """
    half = (num_taps - 1) // 2
    t = np.arange(-half, half + 1) / float(sps)
    h = np.zeros(len(t), dtype=np.float64)
    for i, ti in enumerate(t):
        if np.isclose(ti, 0.0):
            h[i] = 1.0 - alpha + (4.0 * alpha / np.pi)
        elif np.isclose(np.abs(ti), 1.0 / (4.0 * alpha)):
            h[i] = (alpha / np.sqrt(2.0)) * (
                ((1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * alpha)))
                + ((1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * alpha)))
            )
        else:
            num = np.sin(np.pi * ti * (1.0 - alpha)) + 4.0 * alpha * ti * np.cos(np.pi * ti * (1.0 + alpha))
            den = np.pi * ti * (1.0 - (4.0 * alpha * ti) ** 2)
            h[i] = num / den
    energy = np.sqrt(np.sum(h ** 2))
    if energy > 1e-12:
        h = h / energy
    return h.astype(np.float32)


def compute_cumulants(signal: np.ndarray, fs: float = None, fc: float = None) -> dict:
    """Compute exact 2nd, 4th, and 6th order Higher-Order Cumulants (HOC) of a normalized complex baseband signal.
    
    If fc is provided and non-zero (with fs), downconverts the signal to baseband before calculation.
    
    Theoretical Definitions:
      C20 = E[s^2]
      C21 = E[|s|^2] = 1.0 (unit variance)
      C40 = E[s^4] - 3 * C20^2
      C41 = E[s^3 * s*] - 3 * C20 * C21
      C42 = E[|s|^4] - |C20|^2 - 2 * C21^2
      C60 = E[s^6] - 15 * E[s^4] * E[s^2] + 30 * (E[s^2])^3
      C63 = E[|s|^6] - 9 * C42 * C21 - 6 * C21^3 + 3 * |C20|^2 * C21
      
    Returns
    -------
    dict
        Dictionary containing c20, c21, c40, c41, c42, c60, c63.
    """
    if len(signal) < 32:
        return {
            "c20": 0.0 + 0.0j,
            "c21": 1.0,
            "c40": 0.0 + 0.0j,
            "c41": 0.0 + 0.0j,
            "c42": -1.0,
            "c60": 0.0 + 0.0j,
            "c63": 4.0,
        }

    # Convert real signals via analytic Hilbert transform
    if not np.iscomplexobj(signal):
        sig_c = hilbert(signal)
    else:
        sig_c = signal

    # Baseband downconversion if center frequency is given
    if fc is not None and fc != 0.0 and fs is not None and fs > 0:
        t = np.arange(len(sig_c)) / fs
        sig_c = sig_c * np.exp(-2j * np.pi * fc * t)

    # Zero-mean and unit-variance normalization
    s = sig_c - np.mean(sig_c)
    var = np.mean(np.abs(s) ** 2)
    if var > 1e-12:
        s = s / np.sqrt(var)
    else:
        return {
            "c20": 0.0 + 0.0j,
            "c21": 1.0,
            "c40": 0.0 + 0.0j,
            "c41": 0.0 + 0.0j,
            "c42": -1.0,
            "c60": 0.0 + 0.0j,
            "c63": 4.0,
        }

    # Moments
    s_conj = np.conj(s)
    abs_s_sq = np.abs(s) ** 2
    m20 = np.mean(s ** 2)
    m21 = np.mean(abs_s_sq)
    m40 = np.mean(s ** 4)
    m41 = np.mean((s ** 3) * s_conj)
    m42 = np.mean(abs_s_sq ** 2)
    m60 = np.mean(s ** 6)
    m63 = np.mean(abs_s_sq ** 3)

    # Cumulants
    c20 = m20
    c21 = m21
    c40 = m40 - 3.0 * (m20 ** 2)
    c41 = m41 - 3.0 * m20 * m21
    c42 = m42 - np.abs(m20) ** 2 - 2.0 * (m21 ** 2)
    c60 = m60 - 15.0 * m40 * m20 + 30.0 * (m20 ** 3)
    c63 = m63 - 9.0 * c42 * c21 - 6.0 * (c21 ** 3) + 3.0 * (np.abs(m20) ** 2) * c21

    return {
        "c20": complex(c20),
        "c21": float(np.real(c21)),
        "c40": complex(c40),
        "c41": complex(c41),
        "c42": float(np.real(c42)),
        "c60": complex(c60),
        "c63": float(np.real(c63)),
    }


def extract_instantaneous_features(signal: np.ndarray, fs: float = 1000000.0) -> dict:
    """Extract instantaneous amplitude, phase, and frequency statistics."""
    if not np.iscomplexobj(signal):
        sig = hilbert(signal)
    else:
        sig = signal

    # Instantaneous amplitude
    env = np.abs(sig)
    mean_env = np.mean(env) + 1e-12
    env_norm = env / mean_env
    gamma_max = float(np.max(env_norm ** 2)) if len(env_norm) > 0 else 1.0
    sigma_aa = float(np.std(env_norm))
    var_env = np.var(env)
    kurtosis_env = (
        float((np.mean((env - np.mean(env)) ** 4) / (var_env ** 2 + 1e-12)) - 3.0)
        if var_env > 1e-12
        else 0.0
    )

    # Instantaneous phase
    phase = np.unwrap(np.angle(sig))
    phase_centered = phase - np.mean(phase)
    sigma_dp = float(np.std(phase_centered)) if len(phase_centered) > 0 else 0.0

    # Instantaneous frequency with median filtering
    if len(phase) > 1:
        inst_freq = np.diff(phase) / (2.0 * np.pi) * fs
        raw_std = np.std(inst_freq) + 1e-12
        k_size = min(11, len(inst_freq))
        if k_size % 2 == 0:
            k_size -= 1
        if k_size >= 3:
            inst_freq_filt = medfilt(inst_freq, kernel_size=k_size)
        else:
            inst_freq_filt = inst_freq
        filt_std = np.std(inst_freq_filt)
        sigma_af = float(filt_std / (fs + 1e-12))
        fsk_persistence = float(filt_std / raw_std)
    else:
        inst_freq_filt = np.array([0.0])
        sigma_af = 0.0
        fsk_persistence = 0.0

    return {
        "gamma_max": gamma_max,
        "sigma_aa": sigma_aa,
        "sigma_dp": sigma_dp,
        "sigma_af": sigma_af,
        "fsk_persistence": fsk_persistence,
        "kurtosis_env": kurtosis_env,
        "inst_freq_filtered": inst_freq_filt,
    }


def extract_spectral_features(signal: np.ndarray, fs: float = 1000000.0) -> dict:
    """Extract normalized spectral entropy and power distribution features."""
    nperseg = min(512, max(16, len(signal)))
    _, psd = welch(signal, fs=fs, nperseg=nperseg, return_onesided=not np.iscomplexobj(signal))
    psd_norm = psd / (np.sum(psd) + 1e-12)
    n_bins = len(psd_norm)
    spec_entropy = float(
        -np.sum(psd_norm * np.log2(psd_norm + 1e-12)) / (np.log2(n_bins) if n_bins > 1 else 1.0)
    )
    return {
        "spec_entropy": spec_entropy,
    }


def extract_features(signal: np.ndarray, fs: float = 1000000.0, fc: float = None) -> np.ndarray:
    """Extract a comprehensive 15-dimensional discriminative feature vector for Automatic Modulation Recognition (AMR).
    
    Features extracted:
      0: |C20|
      1: |C40|
      2: |C41|
      3: Re(C42)
      4: |C60|
      5: Re(C63)
      6: gamma_max (peak normalized instantaneous envelope power)
      7: sigma_aa (normalized instantaneous amplitude std)
      8: sigma_dp (instantaneous phase std)
      9: sigma_af (median-filtered instantaneous frequency std normalized by fs)
      10: spec_entropy (normalized spectral entropy in [0, 1])
      11: kurtosis_env (excess kurtosis of envelope)
      12: fsk_persistence (ratio of median-filtered frequency std to raw std)
      13: qpsk_metric (|E[s^4]| / E[|s|^4])
      14: psk8_metric (|E[s^8]| / E[|s|^8])
    """
    if not np.iscomplexobj(signal):
        sig_c = hilbert(signal)
    else:
        sig_c = signal

    max_samples = 32768
    sig = sig_c[:max_samples] if len(sig_c) > max_samples else sig_c

    cum = compute_cumulants(sig, fs=fs, fc=fc)
    inst = extract_instantaneous_features(sig, fs=fs)
    spec = extract_spectral_features(sig, fs=fs)

    # Power fold circularity metrics
    s_norm = (sig - np.mean(sig)) / (np.sqrt(np.mean(np.abs(sig - np.mean(sig)) ** 2)) + 1e-12)
    qpsk_metric = float(np.abs(np.mean(s_norm ** 4)) / (np.mean(np.abs(s_norm) ** 4) + 1e-12))
    psk8_metric = float(np.abs(np.mean(s_norm ** 8)) / (np.mean(np.abs(s_norm) ** 8) + 1e-12))

    feats = [
        float(np.abs(cum["c20"])),
        float(np.abs(cum["c40"])),
        float(np.abs(cum["c41"])),
        float(np.real(cum["c42"])),
        float(np.abs(cum["c60"])),
        float(np.real(cum["c63"])),
        float(inst["gamma_max"]),
        float(inst["sigma_aa"]),
        float(inst["sigma_dp"]),
        float(inst["sigma_af"]),
        float(spec["spec_entropy"]),
        float(inst["kurtosis_env"]),
        float(inst["fsk_persistence"]),
        qpsk_metric,
        psk8_metric,
    ]
    return np.nan_to_num(np.array(feats, dtype=np.float32), nan=0.0, posinf=100.0, neginf=-100.0)


def compute_psd(signal: np.ndarray, fs: float, nperseg: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """Compute Power Spectral Density using Welch's method.
    Returns frequencies and PSD values.
    """
    is_complex = np.iscomplexobj(signal)
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg, return_onesided=not is_complex)
    if is_complex:
        freqs = np.fft.fftshift(freqs)
        psd = np.fft.fftshift(psd)
    return freqs, psd


def compute_spectrogram(
    signal: np.ndarray,
    fs: float,
    nperseg: int = 256,
    noverlap: int = 128,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return time, frequency, and magnitude spectrogram (in dB)."""
    max_samples = 500000
    sig_chunk = signal[:max_samples] if len(signal) > max_samples else signal
    is_complex = np.iscomplexobj(sig_chunk)
    f, t, Sxx = spectrogram(
        sig_chunk,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        return_onesided=not is_complex,
    )
    if is_complex:
        f = np.fft.fftshift(f)
        Sxx = np.fft.fftshift(Sxx, axes=0)
    Sxx_db = 10 * np.log10(Sxx + 1e-12)
    return t, f, Sxx_db


def plot_spectrogram(t: np.ndarray, f: np.ndarray, Sxx_db: np.ndarray, title: str = "Spectrogram"):
    fig, ax = plt.subplots(figsize=(8, 4))
    mesh = ax.pcolormesh(t, f, Sxx_db, shading="gouraud", cmap="viridis")
    ax.set_ylabel("Frequency [Hz]")
    ax.set_xlabel("Time [sec]")
    ax.set_title(title)
    fig.colorbar(mesh, ax=ax, label="dB")
    fig.tight_layout()
    return fig


def plot_constellation(symbols: np.ndarray, title: str = "I-Q Constellation Diagram"):
    """Plot complex symbols on the In-Phase (I) vs Quadrature (Q) plane."""
    fig, ax = plt.subplots(figsize=(5, 5))
    if len(symbols) > 0:
        ax.scatter(symbols.real, symbols.imag, alpha=0.5, s=12, c="#1f77b4", edgecolors="none")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_xlabel("In-Phase (I)")
    ax.set_ylabel("Quadrature (Q)")
    ax.set_title(title)
    ax.grid(True, linestyle=":", alpha=0.6)
    lim = max(1.8, np.max(np.abs(symbols)) * 1.15) if len(symbols) > 0 else 2.0
    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])
    fig.tight_layout()
    return fig
