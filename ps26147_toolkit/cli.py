import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np

from .preprocess import load_iq, load_wav
from .feature_extractor import compute_psd, compute_spectrogram, plot_spectrogram
from .parameter_extractor import (
    estimate_center_frequency,
    estimate_bandwidth,
    estimate_snr,
    estimate_baud_rate,
)
from .classifier import ModulationClassifier

def process_file(file_path: str, plot: bool = False) -> dict:
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"File {file_path} not found")
    # Determine type by extension
    if p.suffix.lower() == ".iq":
        # Load complex IQ (interleaved float32)
        signal = load_iq(str(p))
        fs = 1.0  # placeholder – user can embed sampling rate metadata later
    elif p.suffix.lower() == ".wav":
        fs, signal = load_wav(str(p))
    else:
        raise ValueError("Unsupported file type. Use .iq or .wav")

    # Feature extraction
    # Use a sensible nperseg based on signal length to avoid warnings
    nperseg = min(1024, signal.size)
    freqs, psd = compute_psd(signal, fs, nperseg=nperseg)
    center_freq = estimate_center_frequency(freqs, psd)
    bw = estimate_bandwidth(freqs, psd)
    peak_idx = int(np.argmax(psd))
    snr = estimate_snr(psd, peak_idx)
    baud = estimate_baud_rate(signal, fs)

    # Modulation classification (pre‑trained RandomForest)
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
    parser = argparse.ArgumentParser(description="PS26147 – .IQ/.wav analysis toolkit")
    parser.add_argument(
        "input",
        help="Path to a .iq, .wav file, a directory, or a glob pattern (e.g. '*.wav')",
    )
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
        # Process every supported file in the directory (recursive)
        files = list(input_path.rglob("*.iq")) + list(input_path.rglob("*.wav"))
    else:
        # Treat argument as a glob pattern relative to current working dir
        files = list(Path('.').glob(str(input_path)))
        files = [p for p in files if p.suffix.lower() in {".iq", ".wav"}]

    for f in files:
        rep = process_file(str(f), plot=args.plot)
        reports.append(rep)
        if not args.csv:
            # Write per‑file JSON immediately (overwrites previous if multiple files)
            with open(args.output, "w", encoding="utf-8") as jf:
                json.dump(rep, jf, indent=2)
            print(f"Report for {f.name} written to {args.output}")

    if args.csv:
        df = pd.DataFrame(reports)
        df.to_csv(args.csv, index=False)
        print(f"Batch CSV summary written to {args.csv}")

if __name__ == "__main__":
    main()
