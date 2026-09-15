"""
ps26147_toolkit/preprocess.py
Phase 1 – Ingestion, Preprocessing & Signal Normalization

SOP 1.1 – Multi-Dtype .iq Loader & Auto-Detection
SOP 1.2 – Explicit Sample-Rate Handling & SignalMetadata
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import scipy.io.wavfile as wav

# ---------------------------------------------------------------------------
# SOP 1.2 – SignalMetadata dataclass
# ---------------------------------------------------------------------------

@dataclass
class SignalMetadata:
    """Structured metadata carrier for every ingested signal.

    Parameters
    ----------
    fs : float
        Sample rate in Hz.  **Must be provided explicitly** — no silent
        fallback.  Load from a companion ``.sigmf-meta`` file when available.
    center_freq_hint : float, optional
        RF centre frequency hint in Hz (0 for baseband captures).
    source_format : str
        One of ``'iq'``, ``'wav'``, ``'sigmf'``.
    dtype : str
        Original on-disk dtype string, e.g. ``'int16'``, ``'float32'``.
    duration_sec : float
        Duration of the captured signal in seconds.
    num_samples : int
        Total number of complex samples (I+Q pairs).
    """

    fs: float
    center_freq_hint: float = 0.0
    source_format: str = "iq"
    dtype: str = "float32"
    duration_sec: float = 0.0
    num_samples: int = 0

    def __post_init__(self) -> None:
        if self.fs <= 0:
            raise ValueError(
                f"SignalMetadata.fs must be positive, got {self.fs!r}. "
                "Provide an explicit sample rate — there is no safe default."
            )


# ---------------------------------------------------------------------------
# SOP 1.1 – Internal helpers
# ---------------------------------------------------------------------------

#: Supported explicit dtype tokens
_DTYPE_MAP: dict[str, np.dtype] = {
    "int8":      np.dtype(np.int8),
    "uint8":     np.dtype(np.uint8),
    "int16":     np.dtype(np.int16),
    "float32":   np.dtype(np.float32),
    "complex64": np.dtype(np.complex64),
}

#: Probe block size (bytes) for auto-detection
_PROBE_BYTES = 4096


def _probe_dtype(file_path: Path) -> str:
    """Heuristically detect the on-disk sample dtype by reading the first
    ``_PROBE_BYTES`` bytes.

    Detection strategy (per SOP 1.1 §2):
    1. If read as float32 and values are finite in normalized range [0.001, 2.0]
       -> complex64 (if file size % 8 == 0) or float32.
    2. If bytes have mean around 127.5 and bounded standard deviation -> uint8 (RTL-SDR).
    3. If read as int16 and peak values > 100 -> int16.
    4. Fallback -> float32.
    """
    raw = file_path.read_bytes()[:_PROBE_BYTES]
    if len(raw) < 8:
        return "float32"

    total_file_bytes = file_path.stat().st_size

    # 1. Normal float32/complex64 check
    if total_file_bytes % 4 == 0:
        n_f32 = len(raw) // 4
        block_f32 = np.frombuffer(raw[:n_f32 * 4], dtype=np.float32)
        if np.all(np.isfinite(block_f32)):
            max_f = float(np.max(np.abs(block_f32)))
            mean_f = float(np.mean(np.abs(block_f32)))
            non_zero = block_f32[block_f32 != 0]
            subnormal_ratio = float(np.mean(np.abs(non_zero) < 1e-7)) if len(non_zero) > 0 else 0.0
            # Real float32 IQ has samples in normalized range without integer-reinterpreted subnormals (<1e-7)
            if 0.001 <= max_f <= 2.0 and mean_f >= 0.0005 and subnormal_ratio < 0.05:
                return "complex64" if total_file_bytes % 8 == 0 else "float32"

    # 2. uint8 RTL-SDR (unsigned 8-bit offset-binary, bounded std < 65)
    block_u8 = np.frombuffer(raw, dtype=np.uint8)
    mean_u8 = float(np.mean(block_u8))
    std_u8 = float(np.std(block_u8))
    if abs(mean_u8 - 127.5) < 30 and std_u8 < 65:
        return "uint8"

    # 3. int16 check (SDR int16 has high peak integer amplitudes > 100)
    if total_file_bytes % 2 == 0:
        n_i16 = len(raw) // 2
        block_i16 = np.frombuffer(raw[:n_i16 * 2], dtype=np.int16)
        if np.max(np.abs(block_i16.astype(np.int32))) > 100:
            return "int16"

    return "float32"


def _normalize_raw(raw_data: np.ndarray, dtype_str: str) -> np.ndarray:
    """Convert a flat real array loaded with ``dtype_str`` into a normalised
    ``float32`` array in the range roughly ``[-1.0, +1.0]``.

    Raises ``ValueError`` if an odd number of elements is found (broken I/Q
    interleaving), per SOP 1.1 §3.
    """
    if raw_data.size % 2 != 0:
        raise ValueError(
            f"IQ file contains an odd number of samples ({raw_data.size}). "
            "Data must contain interleaved I/Q pairs (even count required)."
        )

    if dtype_str == "int16":
        return raw_data.astype(np.float32) / 32767.0
    elif dtype_str == "int8":
        return raw_data.astype(np.float32) / 127.0
    elif dtype_str == "uint8":
        # RTL-SDR: unsigned 8-bit, offset 127.5
        return (raw_data.astype(np.float32) - 127.5) / 127.5
    else:
        # float32 – pass through
        return raw_data.astype(np.float32)


# ---------------------------------------------------------------------------
# SOP 1.1 – Public API
# ---------------------------------------------------------------------------

DTypeLiteral = Literal["int8", "uint8", "int16", "float32", "complex64", "auto"]


def load_iq(
    file_path: str | Path,
    dtype: Optional[DTypeLiteral] = "auto",
    fs: Optional[float] = None,
    center_freq_hint: float = 0.0,
) -> tuple[np.ndarray, SignalMetadata]:
    """Load a raw binary ``.iq`` file and return a ``complex64`` array with
    associated :class:`SignalMetadata`.

    Parameters
    ----------
    file_path : str or Path
        Path to the raw IQ binary file.
    dtype : {'auto', 'int8', 'uint8', 'int16', 'float32', 'complex64'}, default 'auto'
        On-disk sample dtype.  When ``'auto'`` the loader probes the first
        4 KB to determine the most likely dtype automatically.

        Auto-detection priority:

        1. ``int16``    – native int16 max > 1000
        2. ``complex64``– file size % 8 == 0 and float magnitudes ≤ 1.5
        3. ``float32``  – float magnitudes ≤ 1.5
        4. ``uint8``    – byte mean ≈ 127.5 (RTL-SDR)
        5. ``float32``  – fallback

    fs : float, optional
        Sample rate in Hz.  If omitted a ``UserWarning`` is emitted and
        ``fs=1.0`` is used — downstream estimators will be **uncalibrated**.
    center_freq_hint : float, default 0.0
        RF centre frequency hint in Hz, stored in metadata only.

    Returns
    -------
    iq : np.ndarray, dtype=complex64
        Complex baseband samples, normalised to approximately ``[-1, +1]``
        (per channel).
    meta : SignalMetadata
        Structured metadata for the loaded signal.

    Raises
    ------
    FileNotFoundError
        If ``file_path`` does not exist.
    ValueError
        If the file contains an odd number of samples (broken I/Q pairs).
    TypeError
        If ``dtype`` is not one of the accepted tokens.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"IQ file not found: {file_path}")

    # --- dtype validation
    valid_dtypes = {*_DTYPE_MAP, "auto", None}
    if dtype not in valid_dtypes:
        raise TypeError(
            f"Unsupported dtype {dtype!r}. "
            f"Choose from: {sorted(_DTYPE_MAP)} or 'auto'."
        )

    # --- sample-rate guard
    if fs is None:
        warnings.warn(
            "No sample rate (fs) provided to load_iq(). Defaulting to 1.0 Hz. "
            "All downstream frequency/bandwidth/baud estimators will be "
            "UNCALIBRATED. Pass an explicit fs= or load from a .sigmf-meta "
            "companion file.",
            UserWarning,
            stacklevel=2,
        )
        fs = 1.0

    # --- dtype resolution
    if dtype in (None, "auto"):
        dtype_str = _probe_dtype(file_path)
    else:
        dtype_str = dtype  # type: ignore[assignment]

    # --- load & normalise
    if dtype_str == "complex64":
        raw_complex = np.fromfile(file_path, dtype=np.complex64)
        if raw_complex.size == 0:
            raise ValueError(f"IQ file is empty: {file_path}")
        iq = raw_complex.astype(np.complex64)
    else:
        np_dtype = _DTYPE_MAP[dtype_str]
        raw = np.fromfile(file_path, dtype=np_dtype)
        normalised = _normalize_raw(raw, dtype_str)
        iq = (normalised[0::2] + 1j * normalised[1::2]).astype(np.complex64)

    num_samples = iq.size
    meta = SignalMetadata(
        fs=fs,
        center_freq_hint=center_freq_hint,
        source_format="iq",
        dtype=dtype_str,
        duration_sec=num_samples / fs,
        num_samples=num_samples,
    )
    return iq, meta


# ---------------------------------------------------------------------------
# WAV loader (returns metadata too)
# ---------------------------------------------------------------------------

def load_wav(
    file_path: str | Path,
    center_freq_hint: float = 0.0,
) -> tuple[np.ndarray, SignalMetadata]:
    """Load a ``.wav`` audio file.

    Returns a mono ``float32`` signal and associated :class:`SignalMetadata`.
    """
    file_path = Path(file_path)
    sr, sig = wav.read(str(file_path))

    if sig.ndim > 1:
        sig = np.mean(sig, axis=1)

    orig_dtype = sig.dtype
    sig = sig.astype(np.float32)
    if np.issubdtype(orig_dtype, np.integer):
        sig = sig / np.iinfo(orig_dtype).max

    meta = SignalMetadata(
        fs=float(sr),
        center_freq_hint=center_freq_hint,
        source_format="wav",
        dtype=str(orig_dtype),
        duration_sec=len(sig) / float(sr),
        num_samples=len(sig),
    )
    return sig, meta


# ---------------------------------------------------------------------------
# SigMF companion metadata loader (SOP 1.2)
# ---------------------------------------------------------------------------

def load_sigmf_meta(meta_path: str | Path) -> dict:
    """Parse a ``.sigmf-meta`` JSON companion file.

    Returns a dict with at minimum:
    ``{'fs': float, 'center_freq': float, 'dtype': str, 'raw': dict}``
    mapped from SigMF global fields.
    """
    meta_path = Path(meta_path)
    with meta_path.open("r") as fh:
        raw_meta = json.load(fh)

    global_meta = raw_meta.get("global", {})
    return {
        "fs": float(global_meta.get("core:sample_rate", 0.0)),
        "center_freq": float(global_meta.get("core:frequency", 0.0)),
        "dtype": global_meta.get("core:datatype", "cf32_le"),
        "raw": raw_meta,
    }


def load_iq_with_sigmf(
    iq_path: str | Path,
    meta_path: Optional[str | Path] = None,
) -> tuple[np.ndarray, SignalMetadata]:
    """Convenience wrapper: load IQ + auto-locate ``.sigmf-meta`` companion.

    Companion file search order when ``meta_path`` is not given:

    1. ``<iq_path>.sigmf-meta``  (e.g. ``capture.iq.sigmf-meta``)
    2. ``<iq_stem>.sigmf-meta``  (e.g. ``capture.sigmf-meta``)
    """
    iq_path = Path(iq_path)

    if meta_path is None:
        # Try full-name companion first, then stem-based
        for candidate in [
            iq_path.parent / (iq_path.name + ".sigmf-meta"),
            iq_path.with_suffix(".sigmf-meta"),
        ]:
            if candidate.exists():
                meta_path = candidate
                break

    fs: Optional[float] = None
    center_freq_hint = 0.0
    dtype_hint: DTypeLiteral = "auto"

    if meta_path is not None:
        sigmf = load_sigmf_meta(meta_path)
        fs = sigmf["fs"] if sigmf["fs"] > 0 else None
        center_freq_hint = sigmf.get("center_freq", 0.0)

        _sigmf_dtype_map: dict[str, DTypeLiteral] = {
            "cf32_le": "complex64",
            "ci16_le": "int16",
            "ci8":     "int8",
            "cu8":     "uint8",
        }
        dtype_hint = _sigmf_dtype_map.get(sigmf["dtype"], "auto")  # type: ignore[assignment]

    return load_iq(
        iq_path,
        dtype=dtype_hint,
        fs=fs,
        center_freq_hint=center_freq_hint,
    )


# ---------------------------------------------------------------------------
# Signal segmentation helper
# ---------------------------------------------------------------------------

def segment_signal(
    sig: np.ndarray,
    segment_len: int,
    overlap: int = 0,
) -> list[np.ndarray]:
    """Split a 1-D signal into overlapping segments of length ``segment_len``."""
    step = segment_len - overlap
    return [
        sig[start : start + segment_len]
        for start in range(0, len(sig) - segment_len + 1, step)
    ]
