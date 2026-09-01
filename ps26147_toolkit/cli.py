import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np

from .preprocess import load_iq, load_wav
from .filters import clean_signal, remove_dc_offset, bandpass_filter, spectral_denoise
from .feature_extractor import compute_psd, compute_spectrogram, plot_spectrogram
from .parameter_extractor import (
    estimate_center_frequency,
    estimate_bandwidth,
    estimate_snr,
    estimate_baud_rate,
)
from .classifier import ModulationClassifier


def process_file(
    file_path: str,
    fs_iq: float = 1000000.0,
    filter_noise: bool = False,
    denoise: bool = False,
    plot: bool = False,
) -> dict:
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"File {file_path} not found")

    # Determine type by extension
    if p.suffix.lower() == ".iq":
        signal = load_iq(str(p))
        fs = fs_iq
    elif p.suffix.lower() == ".wav":
        fs, signal = load_wav(str(p))
    else:
        raise ValueError("Unsupported file type. Use .iq or .wav")

    # Initial spectral estimation
    nperseg = min(1024, signal.size)
    freqs, psd = compute_psd(signal, fs, nperseg=nperseg)
    center_freq = estimate_center_frequency(freqs, psd)
    bw = estimate_bandwidth(freqs, psd)

    # Optional noise filtering pipeline
    if filter_noise or denoise:
        signal = clean_signal(
            signal,
            fs=fs,
            center_freq=center_freq,
            bandwidth=bw,
            enable_dc_removal=True,
            enable_bandpass=filter_noise,
            enable_denoise=denoise,
        )
        # Recompute spectral parameters on cleaned signal
        freqs, psd = compute_psd(signal, fs, nperseg=nperseg)
        center_freq = estimate_center_frequency(freqs, psd)
        bw = estimate_bandwidth(freqs, psd)

    snr = estimate_snr(psd, freqs=freqs, center_freq=center_freq, bandwidth=bw)
    baud = estimate_baud_rate(signal, fs, center_freq=center_freq, bandwidth=bw)

    # Modulation classification
    classifier = ModulationClassifier()
    modulation = classifier.predict(signal, fs)

    if plot:
        t, f, Sxx_db = compute_spectrogram(signal, fs)
        plot_spectrogram(t, f, Sxx_db, title=f"Spectrogram – {p.name}")

    report = {
        "file": str(p),
        "modulation": modulation,
        "center_frequency_hz": float(center_freq),
        "bandwidth_hz": float(bw),
        "snr_db": float(snr),
        "baud_rate": float(baud),
    }
    return report


def main():
    parser = argparse.ArgumentParser(description="PS26147 – .IQ/.wav signal analysis toolkit")
    parser.add_argument(
        "input",
        help="Path to a .iq, .wav file, a directory, or a glob pattern (e.g. '*.wav')",
    )
    parser.add_argument("--fs", type=float, default=1000000.0, help="Sampling rate for .iq files (default: 1 MHz)")
    parser.add_argument("--filter", action="store_true", help="Apply adaptive bandpass filter around detected carrier")
    parser.add_argument("--denoise", action="store_true", help="Apply spectral subtraction denoising")
    parser.add_argument("--plot", action="store_true", help="Show spectrogram plot for each file")
    parser.add_argument(
        "--output",
        default="output.json",
        help="JSON file to write a single‑file report (ignored when --csv is used)",
    )
    parser.add_argument(
        "--csv",
        help="Path to CSV file for batch summary. If provided, all files are processed and the CSV is written.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    reports = []

    if input_path.is_dir():
        files = list(input_path.rglob("*.iq")) + list(input_path.rglob("*.wav"))
    else:
        files = list(Path(".").glob(str(input_path)))
        files = [p for p in files if p.suffix.lower() in {".iq", ".wav"}]

    for f in files:
        rep = process_file(
            str(f),
            fs_iq=args.fs,
            filter_noise=args.filter,
            denoise=args.denoise,
            plot=args.plot,
        )
        reports.append(rep)
        if not args.csv:
            with open(args.output, "w", encoding="utf-8") as jf:
                json.dump(rep, jf, indent=2)
            print(f"Report for {f.name} written to {args.output}")

    if args.csv:
        df = pd.DataFrame(reports)
        df.to_csv(args.csv, index=False)
        print(f"Batch CSV summary written to {args.csv}")


if __name__ == "__main__":
    main()
