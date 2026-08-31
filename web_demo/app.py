
import streamlit as st
import json
import pandas as pd
import numpy as np
import os
import zipfile
import io
import tempfile
from pathlib import Path

from ps26147_toolkit.preprocess import load_iq, load_wav
from ps26147_toolkit.feature_extractor import compute_psd, compute_spectrogram, plot_spectrogram
from ps26147_toolkit.parameter_extractor import (
    estimate_center_frequency,
    estimate_bandwidth,
    estimate_snr,
    estimate_baud_rate,
)
from ps26147_toolkit.classifier import ModulationClassifier

st.set_page_config(page_title="PS26147 Signal Analyzer", layout="wide")
st.title("PS26147 – .IQ / .wav Signal Analyzer")
st.markdown("Drag‑and‑drop one or more `.iq` or `.wav` files below. The app will extract parameters, classify modulation, and show a spectrogram.")

uploaded_files = st.file_uploader("Upload files", type=["iq", "wav", "zip"], accept_multiple_files=True)

def process_file(file_path: Path, display_name: str, fs_iq: float = 1000000.0) -> dict:
    """Load, extract features, classify, and display results for a single file.
    Returns a dict suitable for batch reporting."""
    # Load signal
    if file_path.suffix.lower() == ".iq":
        signal = load_iq(str(file_path))
        fs = fs_iq
    else:
        fs, signal = load_wav(str(file_path))

    # Feature extraction & parameter estimation
    freqs, psd = compute_psd(signal, fs)
    center_freq = estimate_center_frequency(freqs, psd)
    bw = estimate_bandwidth(freqs, psd)
    peak_idx = int(np.argmax(psd))
    snr = estimate_snr(psd, peak_idx)
    baud = estimate_baud_rate(signal, fs)

    # Classification
    clf = ModulationClassifier()
    modulation = clf.predict(signal, fs)

    # Spectrogram
    t, f, Sxx_db = compute_spectrogram(signal, fs)

    # Display results
    st.subheader(f"File: {display_name}")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.write("**Parameters**")
        st.json({
            "modulation": modulation,
            "center_frequency_hz": float(center_freq),
            "bandwidth_hz": float(bw),
            "snr_db": float(snr),
            "baud_rate": float(baud),
        })
    with col2:
        st.write("**Spectrogram**")
        fig = plot_spectrogram(t, f, Sxx_db, title=f"Spectrogram – {display_name}")
        st.pyplot(fig)

    return {
        "file": str(file_path),
        "modulation": modulation,
        "center_frequency_hz": float(center_freq),
        "bandwidth_hz": float(bw),
        "snr_db": float(snr),
        "baud_rate": float(baud),
    }

# Sidebar settings
st.sidebar.header("IQ Settings")
fs_iq = st.sidebar.number_input(
    "IQ Sampling Rate (Hz)",
    min_value=1.0,
    value=1000000.0,
    step=1000.0,
    format="%.1f",
    help="IQ files do not store sample rate metadata. Specify the rate used during recording."
)

if uploaded_files:
    reports = []
    for uploaded in uploaded_files:
        # Save uploaded content to a temporary location
        temp_path = Path("temp") / uploaded.name
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        with open(temp_path, "wb") as f:
            f.write(uploaded.getbuffer())

        if temp_path.suffix.lower() == ".zip":
            # Extract zip and process supported files
            extracted_any = False
            with tempfile.TemporaryDirectory() as tmpdir:
                with zipfile.ZipFile(temp_path, "r") as z:
                    for member in z.namelist():
                        if member.lower().endswith((".iq", ".wav")):
                            extracted_any = True
                            extracted_path = Path(tmpdir) / Path(member).name
                            z.extract(member, tmpdir)
                            report = process_file(extracted_path, Path(member).name, fs_iq=fs_iq)
                            reports.append(report)
            if not extracted_any:
                st.error("The uploaded zip does not contain any .iq or .wav files.")
        else:
            report = process_file(temp_path, uploaded.name, fs_iq=fs_iq)
            reports.append(report)

    # Batch download buttons
    if reports:
        json_str = json.dumps(reports, indent=2)
        st.download_button(label="Download JSON (all files)", data=json_str,
                           file_name="reports.json", mime="application/json")
        df = pd.DataFrame(reports)
        csv_str = df.to_csv(index=False)
        st.download_button(label="Download CSV summary", data=csv_str,
                           file_name="reports.csv", mime="text/csv")
else:
    st.info("Awaiting file upload…")
