import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, spectrogram

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

def compute_spectrogram(signal: np.ndarray, fs: float, nperseg: int = 256, noverlap: int = 128) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return time, frequency, and magnitude spectrogram (in dB)."""
    max_samples = 500000
    sig_chunk = signal[:max_samples] if len(signal) > max_samples else signal
    is_complex = np.iscomplexobj(sig_chunk)
    f, t, Sxx = spectrogram(sig_chunk, fs=fs, nperseg=nperseg, noverlap=noverlap, return_onesided=not is_complex)
    if is_complex:
        f = np.fft.fftshift(f)
        Sxx = np.fft.fftshift(Sxx, axes=0)
    Sxx_db = 10 * np.log10(Sxx + 1e-12)
    return t, f, Sxx_db

def plot_spectrogram(t: np.ndarray, f: np.ndarray, Sxx_db: np.ndarray, title: str = "Spectrogram"):
    fig, ax = plt.subplots(figsize=(8, 4))
    mesh = ax.pcolormesh(t, f, Sxx_db, shading='gouraud')
    ax.set_ylabel('Frequency [Hz]')
    ax.set_xlabel('Time [sec]')
    ax.set_title(title)
    fig.colorbar(mesh, ax=ax, label='dB')
    fig.tight_layout()
    return fig
