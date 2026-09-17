import io
import tempfile
from pathlib import Path
from typing import Tuple
import numpy as np
import scipy.io.wavfile as wavfile
from ps26147_toolkit.preprocess import load_iq, load_wav

def load_signal_from_bytes(
    contents: bytes,
    filename: str = "",
    default_fs: float = 1_000_000.0,
) -> Tuple[np.ndarray, float]:
    """Parse uploaded file bytes into a numpy signal array and sample rate.
    
    Supports WAV audio files, SigMF / IQ captures, and raw binary streams.
    """
    if len(contents) == 0:
        raise ValueError("Uploaded file is empty.")

    # Check for WAV signature or extension
    if filename.lower().endswith(".wav") or (contents[:4] == b"RIFF" and contents[8:12] == b"WAVE"):
        try:
            sr, sig = wavfile.read(io.BytesIO(contents))
            if sig.ndim > 1:
                sig = np.mean(sig, axis=1)
            orig_dtype = sig.dtype
            sig = sig.astype(np.float32)
            if np.issubdtype(orig_dtype, np.integer):
                sig = sig / np.iinfo(orig_dtype).max
            return sig, float(sr)
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
            sig, meta = load_wav(tmp_path)
            return sig, meta.fs
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
