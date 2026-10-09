import io
import re
import tempfile
from fractions import Fraction
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import scipy.io.wavfile as wavfile
from scipy.signal import resample_poly
from ps26147_toolkit.preprocess import _wav_to_signal, load_iq, load_wav

_FS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*([kKmMgG]?)\s*(?:sps|sa/s)", re.I)


def build_demodulation_quality(demod: dict, snr_db: float) -> dict[str, float]:
    """Return a consistent numeric quality summary for API responses."""
    def finite_float(value: object) -> float:
        converted = float(value)
        return converted if np.isfinite(converted) else 0.0

    beta = demod.get("front_end_beta")
    return {
        "evm_db": finite_float(demod.get("evm_db", 0.0)),
        "evm_percent": finite_float(demod.get("evm_percent", 0.0)),
        "snr_db": finite_float(snr_db),
        "cfo_corrected_hz": finite_float(demod.get("cfo_corrected_hz", 0.0)),
        "rrc_beta": finite_float(beta) if beta is not None else 0.0,
    }


def infer_fs_from_filename(filename: str) -> Optional[float]:
    """Infer a sample rate from names such as ``capture_1Msps.iq``."""
    match = _FS_RE.search(filename or "")
    if match is None:
        return None
    multiplier = {"": 1.0, "k": 1e3, "m": 1e6, "g": 1e9}[match.group(2).lower()]
    return float(match.group(1)) * multiplier

def load_signal_from_bytes(
    contents: bytes,
    filename: str = "",
    default_fs: Optional[float] = None,
    target_fs: Optional[float] = None,
) -> Tuple[np.ndarray, float]:
    """Parse uploaded file bytes into a numpy signal array and sample rate.
    
    Supports WAV audio files, SigMF / IQ captures, and raw binary streams.
    """
    if len(contents) == 0:
        raise ValueError("Uploaded file is empty.")

    if default_fs is None:
        default_fs = infer_fs_from_filename(filename) or 1_000_000.0

    def resample_wav(signal: np.ndarray, source_fs: float) -> Tuple[np.ndarray, float]:
        if target_fs is None or float(target_fs) == float(source_fs):
            return signal, float(source_fs)
        if not np.isfinite(target_fs) or target_fs <= 0:
            raise ValueError("Target sample rate must be a finite positive number.")
        ratio = Fraction(float(target_fs) / float(source_fs)).limit_denominator(100_000)
        out_dtype = np.complex64 if np.iscomplexobj(signal) else np.float32
        return resample_poly(signal, ratio.numerator, ratio.denominator).astype(out_dtype), float(target_fs)

    # Check for WAV signature or extension
    if filename.lower().endswith(".wav") or (contents[:4] == b"RIFF" and contents[8:12] == b"WAVE"):
        try:
            sr, sig = wavfile.read(io.BytesIO(contents))
            sig = _wav_to_signal(sig, stereo="iq")
            return resample_wav(sig, float(sr))
        except Exception:
            pass

    # Save to temp file to leverage toolkit's robust IQ / SigMF detection
    suffix = Path(filename).suffix if filename else ".iq"
    if not suffix:
        suffix = ".iq"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(contents)
        tmp_path = Path(tmp.name)

    try:
        if suffix.lower() == ".wav":
            sig, meta = load_wav(tmp_path, stereo="iq")
            return resample_wav(sig, float(meta.fs))
        else:
            sig, meta = load_iq(tmp_path, dtype="auto", fs=default_fs)
            return sig, meta.fs
    except Exception as e:
        # Fallback to direct raw numpy interpretation
        try:
            # Try float32
            if len(contents) % 4 == 0:
                raw_f32 = np.frombuffer(contents, dtype=np.float32)
                if np.all(np.isfinite(raw_f32)) and len(raw_f32) >= 2:
                    if len(raw_f32) % 2 == 0:
                        iq = (raw_f32[0::2] + 1j * raw_f32[1::2]).astype(np.complex64)
                        return iq, default_fs
                    return raw_f32, default_fs
            # Try int16
            raw_i16 = np.frombuffer(contents, dtype=np.int16).astype(np.float32) / 32768.0
            if len(raw_i16) % 2 == 0 and len(raw_i16) >= 4:
                iq = (raw_i16[0::2] + 1j * raw_i16[1::2]).astype(np.complex64)
                return iq, default_fs
            return raw_i16, default_fs
        except Exception:
            raise ValueError(f"Failed to decode signal from uploaded file: {str(e)}")
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass
