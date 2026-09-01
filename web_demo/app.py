import streamlit as st
import json
import pandas as pd
import numpy as np
import os
import sys
import zipfile
import io
import tempfile
from pathlib import Path

# Ensure local ps26147_toolkit directory is at the front of sys.path
_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from ps26147_toolkit.preprocess import load_iq, load_wav
from ps26147_toolkit.filters import (
    remove_dc_offset,
    bandpass_filter,
    spectral_denoise,
    clean_signal,
)
from ps26147_toolkit.feature_extractor import (
    compute_psd,
    compute_spectrogram,
    plot_spectrogram,
    plot_constellation,
)
from ps26147_toolkit.parameter_extractor import (
    estimate_center_frequency,
    estimate_bandwidth,
    estimate_snr,
    estimate_baud_rate,
)
from ps26147_toolkit.classifier import ModulationClassifier
from ps26147_toolkit.demodulator import demodulate_signal

st.set_page_config(page_title="PS26147 Signal & Demodulation Toolkit", layout="wide", page_icon="📡")
st.title("📡 PS26147 – Signal Analysis, Demodulation & Spectrum Toolkit")
st.markdown(
    "Drag‑and‑drop `.iq`, `.wav`, or `.zip` files below. "
    "The system performs adaptive filtering, parameter extraction, automatic modulation recognition (AMR), "
    "carrier/timing recovery demodulation, constellation diagram plotting, and bitstream recovery."
)

# Sidebar settings
st.sidebar.header("⚙️ Signal Acquisition Settings")
fs_iq = st.sidebar.number_input(
    "IQ Sampling Rate (Hz)",
    min_value=1.0,
    value=1000000.0,
    step=10000.0,
    format="%.1f",
    help="IQ files do not store sample rate metadata. Specify the rate used during recording (e.g. 1,000,000 for 1 Msps)."
)

st.sidebar.header("🧹 Noise Filtering & Conditioning")
enable_filtering = st.sidebar.checkbox("Enable Noise Filters", value=True, help="Apply digital filtering to attenuate out-of-band noise.")

if enable_filtering:
    remove_dc = st.sidebar.checkbox("Remove DC Offset", value=True, help="Eliminate LO leakage / DC bias.")
    use_bandpass = st.sidebar.checkbox("Adaptive Bandpass Filter", value=True, help="Butterworth filter tracking detected signal center frequency and bandwidth.")
    bp_margin = st.sidebar.slider("Bandpass Margin Multiplier", min_value=1.1, max_value=2.0, value=1.3, step=0.05)
    use_denoise = st.sidebar.checkbox("Spectral Denoising (Noise Floor Reduction)", value=False, help="Perform spectral subtraction to lower broadband noise floor.")
    denoise_strength = st.sidebar.slider("Denoising Factor", min_value=0.5, max_value=3.0, value=1.5, step=0.1) if use_denoise else 1.5
    show_comparison = st.sidebar.checkbox("Show Raw vs Filtered Comparison", value=True, help="Display both raw and filtered spectrograms side-by-side.")
else:
    remove_dc = False
    use_bandpass = False
    bp_margin = 1.3
    use_denoise = False
    denoise_strength = 1.5
    show_comparison = False

st.sidebar.header("🔓 Demodulation Settings")
enable_demod = st.sidebar.checkbox("Enable Demodulation & Constellation Slicing", value=True)
mod_override = st.sidebar.selectbox(
    "Modulation Scheme",
    ["Auto-Detect", "BPSK", "QPSK", "8PSK", "16QAM", "64QAM", "2FSK", "4FSK", "AM"],
    index=0,
    help="Select 'Auto-Detect' to use the AI/HOC Classifier or choose a specific scheme to override."
)

uploaded_files = st.file_uploader("Upload files", type=["iq", "wav", "zip"], accept_multiple_files=True)


def process_file(file_path: Path, display_name: str, fs_iq: float = 1000000.0) -> dict:
    """Load, filter, extract features, classify, demodulate, and display results."""
    # 1. Load signal
    if file_path.suffix.lower() == ".iq":
        raw_signal = load_iq(str(file_path))
        fs = fs_iq
    else:
        fs, raw_signal = load_wav(str(file_path))

    # 2. Raw spectral analysis
    nperseg = min(1024, max(2, raw_signal.size))
    raw_freqs, raw_psd = compute_psd(raw_signal, fs, nperseg=nperseg)
    raw_center_freq = estimate_center_frequency(raw_freqs, raw_psd)
    raw_bw = estimate_bandwidth(raw_freqs, raw_psd)
    raw_snr = estimate_snr(raw_psd, freqs=raw_freqs, center_freq=raw_center_freq, bandwidth=raw_bw)
    raw_baud = estimate_baud_rate(raw_signal, fs, center_freq=raw_center_freq, bandwidth=raw_bw)

    # 3. Apply digital filtering if enabled
    processed_signal = raw_signal
    if enable_filtering:
        if remove_dc:
            processed_signal = remove_dc_offset(processed_signal)
        if use_bandpass and raw_bw > 0:
            processed_signal = bandpass_filter(
                processed_signal,
                fs=fs,
                center_freq=raw_center_freq,
                bandwidth=raw_bw,
                margin_factor=bp_margin,
            )
        if use_denoise:
            processed_signal = spectral_denoise(
                processed_signal,
                noise_reduction_factor=denoise_strength,
            )

        # Recompute spectral parameters on cleaned signal
        freqs, psd = compute_psd(processed_signal, fs, nperseg=nperseg)
        center_freq = estimate_center_frequency(freqs, psd)
        bw = estimate_bandwidth(freqs, psd)
        snr = estimate_snr(psd, freqs=freqs, center_freq=center_freq, bandwidth=bw)
        baud = estimate_baud_rate(processed_signal, fs, center_freq=center_freq, bandwidth=bw)
    else:
        freqs, psd = raw_freqs, raw_psd
        center_freq, bw, snr, baud = raw_center_freq, raw_bw, raw_snr, raw_baud

    # 4. Modulation classification
    clf = ModulationClassifier()
    detected_mod = clf.predict(processed_signal, fs)
    effective_mod = detected_mod if mod_override == "Auto-Detect" else mod_override

    # 5. Demodulation & Bit Extraction
    demod_data = None
    if enable_demod:
        demod_data = demodulate_signal(
            processed_signal,
            fs=fs,
            modulation=effective_mod,
            center_freq=center_freq,
            baud_rate=baud,
        )

    # 6. UI Presentation
    st.subheader(f"📁 {display_name}")

    # Summary metrics tiles
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Modulation", effective_mod, help="Auto-detected or manually overridden modulation")
    m2.metric("Center Freq", f"{center_freq:,.1f} Hz")
    m3.metric("Bandwidth", f"{bw:,.1f} Hz")
    m4.metric("In-Band SNR", f"{snr:.2f} dB", delta=f"{snr - raw_snr:+.2f} dB" if enable_filtering else None)
    m5.metric("Est. Baud Rate", f"{baud:,.1f} Baud")
    if demod_data:
        m6.metric("EVM (dB)", f"{demod_data['evm_db']} dB")
    else:
        m6.metric("Demod Status", "Disabled")

    # Tabs for detailed views
    tab1, tab2, tab3 = st.tabs(["📊 Spectral & Spectrogram Analysis", "🌌 Constellation Diagram (I-Q)", "💾 Demodulated Bitstream"])

    with tab1:
        if enable_filtering and show_comparison:
            col1, col2 = st.columns(2)
            with col1:
                st.write("📊 **Raw Signal Spectrogram**")
                t_raw, f_raw, Sxx_raw_db = compute_spectrogram(raw_signal, fs)
                fig_raw = plot_spectrogram(t_raw, f_raw, Sxx_raw_db, title=f"Raw – {display_name}")
                st.pyplot(fig_raw)
            with col2:
                st.write("✨ **Filtered Signal Spectrogram**")
                t_proc, f_proc, Sxx_proc_db = compute_spectrogram(processed_signal, fs)
                fig_proc = plot_spectrogram(t_proc, f_proc, Sxx_proc_db, title=f"Filtered – {display_name}")
                st.pyplot(fig_proc)
        else:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.write("**Extracted Parameters**")
                st.json({
                    "file": display_name,
                    "modulation": effective_mod,
                    "center_frequency_hz": round(float(center_freq), 2),
                    "bandwidth_hz": round(float(bw), 2),
                    "snr_db": round(float(snr), 2),
                    "baud_rate_baud": round(float(baud), 2),
                    "filtering_applied": enable_filtering,
                })
            with col2:
                t, f, Sxx_db = compute_spectrogram(processed_signal, fs)
                fig = plot_spectrogram(t, f, Sxx_db, title=f"Spectrogram – {display_name}")
                st.pyplot(fig)

    with tab2:
        if demod_data:
            c1, c2 = st.columns([1, 1])
            with c1:
                st.write(f"**Recovered Constellation for {effective_mod}**")
                fig_const = plot_constellation(demod_data["symbols"], title=f"Constellation – {effective_mod} ({display_name})")
                st.pyplot(fig_const)
            with c2:
                st.write("**Demodulation Metrics**")
                st.markdown(f"- **Modulation:** `{effective_mod}`")
                st.markdown(f"- **Recovered Symbols:** `{len(demod_data['symbols']):,}` symbols")
                st.markdown(f"- **Recovered Bits:** `{demod_data['num_bits']:,}` bits")
                st.markdown(f"- **EVM (Error Vector Magnitude):** `{demod_data['evm_db']} dB`")
                st.info("Symbols have been downsampled via symbol timing recovery and synchronized using Costas carrier tracking.")
        else:
            st.warning("Enable demodulation in the sidebar to view constellation plots.")

    with tab3:
        if demod_data and demod_data["num_bits"] > 0:
            st.write("**Recovered Bitstream Data**")
            st.text_area("Binary Bitstream Preview (first 512 bits):", value=demod_data["bit_string_preview"], height=100)
            st.text_area("Hex Payload Preview (first 64 bytes):", value=demod_data["hex_preview"], height=70)

            # Download binary bitstream
            bits_bytes = np.packbits(demod_data["bits"]).tobytes()
            st.download_button(
                label="📥 Download Demodulated Bits (.bin)",
                data=bits_bytes,
                file_name=f"{Path(display_name).stem}_demod_bits.bin",
                mime="application/octet-stream",
            )
        else:
            st.info("No bitstream available.")

    st.divider()

    return {
        "file": display_name,
        "modulation": effective_mod,
        "center_frequency_hz": float(center_freq),
        "bandwidth_hz": float(bw),
        "snr_db": float(snr),
        "baud_rate": float(baud),
        "num_bits": demod_data["num_bits"] if demod_data else 0,
        "evm_db": demod_data["evm_db"] if demod_data else None,
        "filtering_applied": enable_filtering,
    }


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
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            json_str = json.dumps(reports, indent=2)
            st.download_button(label="📥 Download JSON Report (all files)", data=json_str,
                               file_name="signal_reports.json", mime="application/json")
        with col_d2:
            df = pd.DataFrame(reports)
            csv_str = df.to_csv(index=False)
            st.download_button(label="📥 Download CSV Summary", data=csv_str,
                               file_name="signal_reports.csv", mime="text/csv")
else:
    st.info("👆 Drag-and-drop or select `.iq`, `.wav`, or `.zip` files to analyze.")
