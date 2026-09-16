import streamlit as st
import json
import pandas as pd
import numpy as np
import sys
import zipfile
import tempfile
from pathlib import Path
import time

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
)
from ps26147_toolkit.parameter_extractor import (
    estimate_center_frequency,
    estimate_bandwidth,
    estimate_snr,
    estimate_baud_rate,
    _SNR_CLIP_CEILING_DB,
)
from ps26147_toolkit.classifier import ModulationClassifier
from ps26147_toolkit.demodulator import demodulate_signal
from ps26147_toolkit.deinterleaver import deinterleave, auto_detect_and_deinterleave
from ps26147_toolkit.fec_decoders import decode_fec
from ps26147_toolkit.correlator import (
    STANDARD_SYNC_WORDS,
    hex_to_bits,
    correlate_bitstream,
    frame_synchronize,
    auto_discover_preamble,
)

# Import Phase 7 interactive visualization components
from plotly_visualizations import (
    plot_interactive_spectrogram_2d,
    plot_waterfall_3d,
    plot_constellation_interactive,
    plot_time_domain_iq,
    plot_psd_interactive,
    create_telemetry_card,
    plot_correlation_interactive,
)

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

st.sidebar.header("🔀 De-Interleaving Settings")
enable_deinterleave = st.sidebar.checkbox("Enable De-Interleaving", value=False, help="Apply de-interleaving to the demodulated bitstream.")

if enable_deinterleave:
    deinterleave_method = st.sidebar.selectbox(
        "De-Interleaving Method",
        ["Auto-Detect", "Block", "Convolutional", "Diagonal", "Pseudo-Random"],
        index=0,
        help="Select 'Auto-Detect' to try all methods and pick the best, or choose a specific method.",
    )

    if deinterleave_method in ("Block", "Diagonal"):
        di_rows = st.sidebar.number_input("Matrix Rows", min_value=2, max_value=256, value=8, step=1)
        di_cols = st.sidebar.number_input("Matrix Cols", min_value=2, max_value=256, value=8, step=1)
    else:
        di_rows, di_cols = 8, 8

    if deinterleave_method == "Convolutional":
        di_branches = st.sidebar.number_input("Branches (N)", min_value=2, max_value=64, value=4, step=1)
        di_delay = st.sidebar.number_input("Unit Delay (D)", min_value=1, max_value=256, value=8, step=1)
    else:
        di_branches, di_delay = 4, 8

    if deinterleave_method == "Pseudo-Random":
        di_block_size = st.sidebar.number_input("Block Size", min_value=8, max_value=4096, value=128, step=8)
        di_seed = st.sidebar.number_input("PRBS Seed", min_value=0, max_value=65535, value=42, step=1)
    else:
        di_block_size, di_seed = 128, 42
st.sidebar.header("🛡️ FEC Decoding Settings")
enable_fec = st.sidebar.checkbox("Enable FEC Decoder", value=False, help="Decode error correction codes from demodulated / de-interleaved bits.")

if enable_fec:
    fec_scheme = st.sidebar.selectbox(
        "FEC Code Type",
        [
            "Viterbi (Convolutional K=7, Rate 1/2)",
            "Reed-Solomon RS(255, 223)",
            "Concatenated (Viterbi + RS)",
            "LDPC (Min-Sum)",
        ],
        index=0,
        help="Select the Forward Error Correction decoding algorithm.",
    )
st.sidebar.header("🎯 Bitstream Correlation & Sync")
enable_sync = st.sidebar.checkbox("Enable Frame Synchronization", value=False, help="Perform cross-correlation with sync markers to frame-align bitstream.")

if enable_sync:
    sync_mode = st.sidebar.selectbox(
        "Sync Word Preset",
        ["Auto-Discover"] + list(STANDARD_SYNC_WORDS.keys()) + ["Custom Hex Pattern"],
        index=0,
    )
    if sync_mode == "Custom Hex Pattern":
        custom_sync_hex = st.sidebar.text_input("Custom Sync Pattern (Hex)", value="1ACFFC1D")
    else:
        custom_sync_hex = "1ACFFC1D"

    sync_threshold = st.sidebar.slider("Correlation Threshold", min_value=0.5, max_value=1.0, value=0.85, step=0.05)
    tolerate_inv = st.sidebar.checkbox("Tolerate 180° Inverted Carrier", value=True)
else:
    sync_mode = "Auto-Discover"
    custom_sync_hex = "1ACFFC1D"
    sync_threshold = 0.85
    tolerate_inv = True

uploaded_files = st.file_uploader("Upload files", type=["iq", "wav", "zip"], accept_multiple_files=True)


def format_snr(snr_db: float) -> str:
    """Format an SNR reading, flagging a ceiling-pegged value instead of letting
    it masquerade as a precise measurement (Phase 6 §1.4)."""
    if snr_db >= _SNR_CLIP_CEILING_DB:
        return f"≥{_SNR_CLIP_CEILING_DB:.0f} dB (clipped)"
    return f"{snr_db:.2f} dB"


def process_file(file_path: Path, display_name: str, fs_iq: float = 1000000.0) -> dict:
    """Load, filter, extract features, classify, demodulate, and display results."""

    # Initialize telemetry tracking
    telemetry = {}

    # 1. Load signal
    t_start = time.time()
    if file_path.suffix.lower() == ".iq":
        raw_signal, _ = load_iq(str(file_path))
        fs = fs_iq
    else:
        raw_signal, meta = load_wav(str(file_path))
        fs = meta.fs
    telemetry['Signal Ingestion'] = {
        'status': 'complete',
        'metric': f'{len(raw_signal):,} samples',
        'duration_ms': (time.time() - t_start) * 1000
    }

    # 2. Raw spectral analysis
    t_start = time.time()
    nperseg = min(1024, max(2, raw_signal.size))
    raw_freqs, raw_psd = compute_psd(raw_signal, fs, nperseg=nperseg)
    raw_center_freq = estimate_center_frequency(raw_freqs, raw_psd)
    raw_bw = estimate_bandwidth(raw_freqs, raw_psd)
    raw_snr = estimate_snr(raw_psd, freqs=raw_freqs, center_freq=raw_center_freq, bandwidth=raw_bw)
    raw_baud = estimate_baud_rate(raw_signal, fs, center_freq=raw_center_freq, bandwidth=raw_bw)
    telemetry['Parameter Extraction'] = {
        'status': 'complete',
        'metric': f'SNR: {raw_snr:.1f} dB',
        'duration_ms': (time.time() - t_start) * 1000
    }

    # 3. Apply digital filtering if enabled
    processed_signal = raw_signal
    if enable_filtering:
        t_start = time.time()
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
        telemetry['Signal Conditioning'] = {
            'status': 'complete',
            'metric': f'SNR Gain: +{raw_snr:.1f} dB',
            'duration_ms': (time.time() - t_start) * 1000
        }

        # Recompute spectral parameters on cleaned signal
        freqs, psd = compute_psd(processed_signal, fs, nperseg=nperseg)
        center_freq = estimate_center_frequency(freqs, psd)
        bw = estimate_bandwidth(freqs, psd)
        snr = estimate_snr(psd, freqs=freqs, center_freq=center_freq, bandwidth=bw)
        baud = estimate_baud_rate(processed_signal, fs, center_freq=center_freq, bandwidth=bw)
    else:
        freqs, psd = raw_freqs, raw_psd
        center_freq, bw, snr, baud = raw_center_freq, raw_bw, raw_snr, raw_baud
        telemetry['Signal Conditioning'] = {'status': 'skipped', 'metric': 'Disabled'}

    # 4. Modulation classification
    t_start = time.time()
    clf = ModulationClassifier()
    detected_mod = clf.predict(processed_signal, fs)
    effective_mod = detected_mod if mod_override == "Auto-Detect" else mod_override
    telemetry['Modulation Classification'] = {
        'status': 'complete',
        'metric': effective_mod,
        'duration_ms': (time.time() - t_start) * 1000
    }

    # 5. Demodulation & Bit Extraction
    demod_data = None
    if enable_demod:
        t_start = time.time()
        demod_data = demodulate_signal(
            processed_signal,
            fs=fs,
            modulation=effective_mod,
            center_freq=center_freq,
            baud_rate=baud,
        )
        telemetry['Demodulation'] = {
            'status': 'complete',
            'metric': f'{demod_data["num_bits"]:,} bits',
            'duration_ms': (time.time() - t_start) * 1000
        }
    else:
        telemetry['Demodulation'] = {'status': 'skipped', 'metric': 'Disabled'}

    # 6. De-interleaving
    deinterleave_result = None
    if enable_deinterleave and demod_data and demod_data["num_bits"] > 0:
        raw_bits = demod_data["bits"]
        if deinterleave_method == "Auto-Detect":
            deinterleave_result = auto_detect_and_deinterleave(raw_bits)
        else:
            method_map = {
                "Block": "block",
                "Convolutional": "convolutional",
                "Diagonal": "diagonal",
                "Pseudo-Random": "pseudo-random",
            }
            deinterleave_result = deinterleave(
                raw_bits,
                method=method_map[deinterleave_method],
                rows=di_rows,
                cols=di_cols,
                num_branches=di_branches,
                delay=di_delay,
                block_size=di_block_size,
                seed=di_seed,
            )

    # 7. FEC Decoding
    fec_result = None
    if enable_fec and demod_data and demod_data["num_bits"] > 0:
        # If de-interleaving is enabled, feed de-interleaved bits into FEC decoder; otherwise feed raw demodulated bits
        input_fec_bits = deinterleave_result["bits"] if (deinterleave_result and len(deinterleave_result["bits"]) > 0) else demod_data["bits"]
        # Pass soft LLRs only when using the original demodulated bits
        # (de-interleaving operates on hard bits, so LLRs don't survive re-ordering)
        input_llr = demod_data["llr"] if (deinterleave_result is None or len(deinterleave_result["bits"]) == 0) else None
        fec_result = decode_fec(input_fec_bits, scheme=fec_scheme, llr=input_llr)

    # 8. Bitstream Correlation & Frame Synchronization
    sync_result = None
    if enable_sync and demod_data and demod_data["num_bits"] > 0:
        # Determine bitstream to synchronize (priority: FEC Decoded > De-interleaved > Demodulated)
        if fec_result and len(fec_result.get("bits", [])) > 0:
            target_stream = fec_result["bits"]
        elif deinterleave_result and len(deinterleave_result.get("bits", [])) > 0:
            target_stream = deinterleave_result["bits"]
        else:
            target_stream = demod_data["bits"]

        if sync_mode == "Auto-Discover":
            discovery = auto_discover_preamble(target_stream)
            if discovery.get("matched_standard_sync"):
                active_sync_word = STANDARD_SYNC_WORDS[discovery["matched_standard_sync"]]
            elif discovery.get("candidate_preamble_bits") is not None:
                active_sync_word = discovery["candidate_preamble_bits"]
            else:
                active_sync_word = STANDARD_SYNC_WORDS["Barker-13"]
            sync_result = frame_synchronize(target_stream, active_sync_word, threshold=sync_threshold, tolerate_inverted=tolerate_inv)
            sync_result["discovery_info"] = discovery
        elif sync_mode == "Custom Hex Pattern":
            custom_bits = hex_to_bits(custom_sync_hex)
            sync_result = frame_synchronize(target_stream, custom_bits, threshold=sync_threshold, tolerate_inverted=tolerate_inv)
            sync_result["discovery_info"] = None
        else:
            active_sync_word = STANDARD_SYNC_WORDS[sync_mode]
            sync_result = frame_synchronize(target_stream, active_sync_word, threshold=sync_threshold, tolerate_inverted=tolerate_inv)
            sync_result["discovery_info"] = None

    # UI Presentation
    st.subheader(f"📁 {display_name}")

    # Summary metrics tiles
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Modulation", effective_mod, help="Auto-detected or manually overridden modulation")
    m2.metric("Center Freq", f"{center_freq:,.1f} Hz")
    m3.metric("Bandwidth", f"{bw:,.1f} Hz")
    m4.metric("In-Band SNR", format_snr(snr), delta=f"{snr - raw_snr:+.2f} dB" if enable_filtering else None)
    m5.metric("Est. Baud Rate", f"{baud:,.1f} Baud")
    if sync_result and sync_result.get("sync_found"):
        m6.metric("Frame Sync", f"{sync_result['num_frames']} Frames", help=f"Frame len: {sync_result.get('detected_frame_length')} bits")
    elif fec_result:
        m6.metric("FEC Status", "Decoded", help=f"FEC: {fec_result.get('decoder', 'Active')}")
    elif demod_data:
        m6.metric("EVM (dB)", f"{demod_data['evm_db']} dB")
    else:
        m6.metric("Demod Status", "Disabled")

    # Tabs for detailed views
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Spectral & Spectrogram Analysis",
        "🌌 Constellation Diagram (I-Q)",
        "💾 Demodulated Bitstream",
        "🔀 De-Interleaved Output",
        "🛡️ FEC Decoded Stream",
        "🎯 Frame Correlation & Sync",
    ])

    with tab1:
        # Display telemetry dashboard
        telemetry_html = create_telemetry_card(telemetry)
        st.markdown(telemetry_html, unsafe_allow_html=True)

        if enable_filtering and show_comparison:
            col1, col2 = st.columns(2)
            with col1:
                st.write("📊 **Raw Signal Spectrogram**")
                t_raw, f_raw, Sxx_raw_db = compute_spectrogram(raw_signal, fs)
                fig_raw = plot_interactive_spectrogram_2d(t_raw, f_raw, Sxx_raw_db, title=f"Raw – {display_name}", fs=fs)
                st.plotly_chart(fig_raw, use_container_width=True)

                # Also show PSD for raw signal
                raw_psd_db = 10 * np.log10(np.abs(np.fft.fft(raw_signal))**2 / len(raw_signal))
                freqs = np.fft.fftfreq(len(raw_signal), 1/fs)
                fig_psd_raw = plot_psd_interactive(freqs, raw_psd_db,
                                                  center_freq=raw_center_freq,
                                                  bandwidth=raw_bw,
                                                  title=f"Raw PSD – {display_name}")
                st.plotly_chart(fig_psd_raw, use_container_width=True)
            with col2:
                st.write("✨ **Filtered Signal Spectrogram**")
                t_proc, f_proc, Sxx_proc_db = compute_spectrogram(processed_signal, fs)
                fig_proc = plot_interactive_spectrogram_2d(t_proc, f_proc, Sxx_proc_db, title=f"Filtered – {display_name}", fs=fs)
                st.plotly_chart(fig_proc, use_container_width=True)

                # Also show PSD for filtered signal
                proc_psd_db = 10 * np.log10(np.abs(np.fft.fft(processed_signal))**2 / len(processed_signal))
                fig_psd_proc = plot_psd_interactive(freqs, proc_psd_db,
                                                   center_freq=center_freq,
                                                   bandwidth=bw,
                                                   title=f"Filtered PSD – {display_name}")
                st.plotly_chart(fig_psd_proc, use_container_width=True)

                # Add 3D waterfall for filtered signal
                st.write("🌊 **3D Waterfall View (Filtered)**")
                fig_waterfall = plot_waterfall_3d(t_proc, f_proc, Sxx_proc_db, title=f"Waterfall – {display_name}")
                st.plotly_chart(fig_waterfall, use_container_width=True)
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
                st.write("📊 **Interactive Spectrogram**")
                t, f, Sxx_db = compute_spectrogram(processed_signal, fs)
                fig = plot_interactive_spectrogram_2d(t, f, Sxx_db, title=f"Spectrogram – {display_name}", fs=fs)
                st.plotly_chart(fig, use_container_width=True)

                # PSD plot
                proc_psd_db = 10 * np.log10(np.abs(np.fft.fft(processed_signal))**2 / len(processed_signal))
                fig_psd = plot_psd_interactive(freqs, proc_psd_db,
                                              center_freq=center_freq,
                                              bandwidth=bw,
                                              title=f"Power Spectral Density – {display_name}")
                st.plotly_chart(fig_psd, use_container_width=True)

    with tab2:
        if demod_data:
            # Time-domain I/Q analysis
            st.write("📈 **Time-Domain I/Q Signal Analysis**")
            fig_time = plot_time_domain_iq(demod_data["symbols"], fs=fs, title=f"I/Q Waveforms – {display_name}")
            st.plotly_chart(fig_time, use_container_width=True)

            st.divider()

            # Constellation diagram
            c1, c2 = st.columns([1, 1])
            with c1:
                st.write(f"**🎯 Interactive Constellation Diagram for {effective_mod}**")
                fig_const = plot_constellation_interactive(demod_data["symbols"], modulation=effective_mod,
                                                         title=f"Constellation – {effective_mod} ({display_name})",
                                                         show_ideal=True, use_hexbin=True)
                st.plotly_chart(fig_const, use_container_width=True)
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

    with tab4:
        if deinterleave_result and deinterleave_result["method"] != "none":
            di = deinterleave_result
            st.write("**De-Interleaving Results**")

            d1, d2, d3 = st.columns(3)
            d1.metric("Method", di["method"].replace("-", " ").title())
            d2.metric("Output Entropy", f"{di['entropy']:.2f} bits/byte")
            baseline_ent = di.get("baseline_entropy", di["entropy"])
            ent_delta = di["entropy"] - baseline_ent
            d3.metric("Input Entropy", f"{baseline_ent:.2f} bits/byte",
                       delta=f"{ent_delta:+.2f}" if abs(ent_delta) > 0.001 else None)

            st.write("**Parameters Used:**")
            st.json(di["params"])

            di_bits = di["bits"]
            di_bit_str = "".join(str(b) for b in di_bits[:512])
            di_byte_arr = np.packbits(di_bits)
            di_hex_str = di_byte_arr[:64].tobytes().hex().upper()

            st.text_area("De-Interleaved Binary Preview (first 512 bits):", value=di_bit_str, height=100)
            st.text_area("De-Interleaved Hex Preview (first 64 bytes):", value=di_hex_str, height=70)

            di_bytes = np.packbits(di_bits).tobytes()
            st.download_button(
                label="📥 Download De-Interleaved Bits (.bin)",
                data=di_bytes,
                file_name=f"{Path(display_name).stem}_deinterleaved.bin",
                mime="application/octet-stream",
            )
        elif enable_deinterleave:
            if not enable_demod:
                st.warning("Enable demodulation first to produce a bitstream for de-interleaving.")
            elif deinterleave_result and deinterleave_result["method"] == "none":
                st.info("Auto-detection found no interleaving pattern. The bitstream may not be interleaved.")
            else:
                st.info("No bitstream available for de-interleaving.")
        else:
            st.info("Enable de-interleaving in the sidebar to process the demodulated bitstream.")

    with tab5:
        if fec_result and len(fec_result.get("bits", [])) > 0:
            st.write(f"**FEC Decoded Information Stream ({fec_result.get('decoder', 'FEC')})**")
            f1, f2, f3 = st.columns(3)
            f1.metric("FEC Decoder", fec_result.get("decoder", "FEC"))
            f1.caption(f"Input Bits: {fec_result.get('input_bits', 0):,} → Output Info Bits: {fec_result.get('output_bits', 0):,}")
            
            if "errors_corrected" in fec_result:
                f2.metric("Errors Corrected", f"{fec_result['errors_corrected']} bytes")
            elif "blocks_converged" in fec_result:
                f2.metric("LDPC Convergence", f"{fec_result['blocks_converged']}/{fec_result['total_blocks']} blks")
            else:
                f2.metric("Code Rate", fec_result.get("rate", "1/2"))

            f_bits = fec_result["bits"]
            f_bit_str = "".join(str(b) for b in f_bits[:512])
            f_byte_arr = np.packbits(f_bits)
            f_hex_str = f_byte_arr[:64].tobytes().hex().upper()
            f3.metric("Recovered Bytes", f"{len(f_byte_arr):,} bytes")

            st.text_area("Decoded Binary Preview (first 512 bits):", value=f_bit_str, height=100)
            st.text_area("Decoded Hex Payload Preview (first 64 bytes):", value=f_hex_str, height=70)

            # ASCII text attempt if printable
            try:
                raw_ascii = f_byte_arr.tobytes().decode("ascii", errors="replace")
                st.text_area("Decoded ASCII / Text Preview:", value=raw_ascii[:256], height=70)
            except Exception:
                pass

            st.download_button(
                label="📥 Download Clean Decoded Data (.bin)",
                data=f_byte_arr.tobytes(),
                file_name=f"{Path(display_name).stem}_fec_decoded.bin",
                mime="application/octet-stream",
            )
        elif enable_fec:
            if not enable_demod:
                st.warning("Enable demodulation first to generate a bitstream for FEC decoding.")
            else:
                st.info("No bitstream available for FEC decoding.")
        else:
            st.info("Enable FEC Decoder in the sidebar to correct errors and decode information bits.")

    with tab6:
        if sync_result and sync_result.get("sync_found"):
            st.write(f"**🎯 Bitstream Cross-Correlation & Frame Synchronization ({sync_result['num_frames']} Frames Locked)**")
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Sync Status", sync_result["status"])
            s2.metric("Max Correlation", f"{sync_result['max_correlation']:.2f}", help="Peak normalized bipolar correlation score")
            s3.metric("Detected Frame Len", f"{sync_result['detected_frame_length']} bits" if sync_result.get("detected_frame_length") else "Variable")
            s4.metric("Phase Ambiguity", "180° Inverted" if sync_result.get("is_inverted") else "0° Normal")

            if sync_result.get("discovery_info") and sync_result["discovery_info"].get("matched_standard_sync"):
                disc = sync_result["discovery_info"]
                st.success(f"🔍 Auto-Discovered Standard Sync: **{disc['matched_standard_sync']}** (Confidence: {disc['standard_sync_confidence']:.2f})")

            # Plot correlation curve
            if len(sync_result["correlation_curve"]) > 0:
                corr_sub = sync_result["correlation_curve"][:2048]
                peak_indices_sub = [p for p in sync_result["peak_indices"] if p < len(corr_sub)]
                fig_corr = plot_correlation_interactive(
                    corr_sub,
                    peak_indices_sub,
                    sync_threshold,
                    title=f"Sliding Window Cross-Correlation – {display_name}"
                )
                st.plotly_chart(fig_corr, use_container_width=True)

            # Display first 5 extracted frames
            st.write("**Extracted Synchronized Frames Preview:**")
            frame_records = []
            for idx, fr in enumerate(sync_result["frames"][:8]):
                pl_bytes = np.packbits(fr["payload_bits"]).tobytes()
                frame_records.append({
                    "Frame #": idx + 1,
                    "Start Bit": fr["start_bit"],
                    "End Bit": fr["end_bit"],
                    "Correlation Score": round(fr["correlation"], 2),
                    "Payload Bits": len(fr["payload_bits"]),
                    "Hex Payload (First 16B)": pl_bytes[:16].hex().upper(),
                })
            st.dataframe(pd.DataFrame(frame_records), use_container_width=True)

            # Download synchronized framed payloads
            if len(sync_result["frames"]) > 0:
                all_payload_bits = np.concatenate([fr["payload_bits"] for fr in sync_result["frames"]])
                all_payload_bytes = np.packbits(all_payload_bits).tobytes()
                st.download_button(
                    label="📥 Download Extracted Frame Payloads (.bin)",
                    data=all_payload_bytes,
                    file_name=f"{Path(display_name).stem}_synced_payloads.bin",
                    mime="application/octet-stream",
                )
        elif enable_sync:
            if not enable_demod:
                st.warning("Enable demodulation first to generate a bitstream for frame synchronization.")
            else:
                st.warning(f"⚠️ {sync_result.get('status', 'No sync markers detected')}")
                if sync_result and len(sync_result.get("correlation_curve", [])) > 0:
                    corr_sub = sync_result["correlation_curve"][:2048]
                    peak_indices_sub = []  # No peaks when no lock
                    fig_corr = plot_correlation_interactive(
                        corr_sub,
                        peak_indices_sub,
                        sync_threshold,
                        title="Correlation Profile (No Lock)"
                    )
                    st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("Enable Frame Synchronization in the sidebar to search for sync words and extract aligned frames.")

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
        "deinterleave_method": deinterleave_result["method"] if deinterleave_result else None,
        "deinterleave_params": deinterleave_result["params"] if deinterleave_result else None,
        "deinterleave_entropy": deinterleave_result["entropy"] if deinterleave_result else None,
        "fec_decoder": fec_result["decoder"] if fec_result else None,
        "fec_output_bits": fec_result["output_bits"] if fec_result else None,
        "sync_status": sync_result["status"] if sync_result else None,
        "sync_frames_found": sync_result["num_frames"] if (sync_result and sync_result.get("sync_found")) else 0,
        "sync_frame_length": sync_result.get("detected_frame_length") if sync_result else None,
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
