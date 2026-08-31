import numpy as np
import scipy.io.wavfile as wav
from pathlib import Path

def load_iq(file_path: str, dtype=np.complex64) -> np.ndarray:
    """Load a raw .iq file (binary interleaved int16 or float32).
    The function tries to infer the format based on file size.
    """
    data = np.fromfile(file_path, dtype=np.float32)
    if data.size % 2 != 0:
        raise ValueError("IQ file must contain an even number of samples (I and Q).")
    i = data[0::2]
    q = data[1::2]
    return i + 1j * q

def load_wav(file_path: str) -> tuple[int, np.ndarray]:
    """Load a .wav audio file. Returns sample_rate and mono signal (float32)."""
    sr, sig = wav.read(file_path)
    if sig.ndim > 1:
        sig = np.mean(sig, axis=1)
    orig_dtype = sig.dtype
    sig = sig.astype(np.float32)
    if np.issubdtype(orig_dtype, np.integer):
        sig = sig / np.iinfo(orig_dtype).max
    # For floating‑point WAVs we keep the values as‑is (already cast to float32)
    return sr, sig

def segment_signal(sig: np.ndarray, segment_len: int, overlap: int = 0) -> list[np.ndarray]:
    """Split a 1‑D signal into overlapping segments.
    Returns a list of segments of length `segment_len`.
    """
    step = segment_len - overlap
    segments = []
    for start in range(0, len(sig) - segment_len + 1, step):
        segments.append(sig[start:start + segment_len])
    return segments
