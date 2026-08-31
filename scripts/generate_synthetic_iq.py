import numpy as np
from pathlib import Path

def generate_synthetic_iq(file_path: str, freq_hz: float = 100.0, fs: float = 1000.0, duration_sec: float = 1.0):
    """Generate a simple complex sinusoid and save as interleaved float32 IQ.
    Parameters
    ----------
    file_path: str
        Destination filename (e.g., 'synthetic.iq').
    freq_hz: float, default 100.0
        Tone frequency.
    fs: float, default 1000.0
        Sampling rate.
    duration_sec: float, default 1.0
        Length of the signal in seconds.
    """
    t = np.arange(0, duration_sec, 1/fs)
    signal = np.exp(1j * 2 * np.pi * freq_hz * t)  # complex exponential
    # Interleave I and Q as float32
    interleaved = np.empty(signal.size * 2, dtype=np.float32)
    interleaved[0::2] = signal.real.astype(np.float32)
    interleaved[1::2] = signal.imag.astype(np.float32)
    Path(file_path).write_bytes(interleaved.tobytes())
    print(f"Synthetic IQ file written to {file_path}")

if __name__ == "__main__":
    generate_synthetic_iq('synthetic.iq')
