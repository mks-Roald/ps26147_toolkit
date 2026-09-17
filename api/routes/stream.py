import asyncio
import json
import time
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from ps26147_toolkit import classifier, parameter_extractor, feature_extractor

router = APIRouter()

# Constellation dictionaries for fast SDR simulation
_MOD_MAP = {
    "BPSK": np.array([-1.0, 1.0], dtype=np.complex64),
    "QPSK": np.array([-1-1j, -1+1j, 1-1j, 1+1j], dtype=np.complex64) / np.sqrt(2),
    "8PSK": np.exp(1j * np.arange(8) * (2 * np.pi / 8)).astype(np.complex64),
    "16QAM": (
        np.tile(np.array([-3, -1, 1, 3]), 4) + 1j * np.repeat(np.array([-3, -1, 1, 3]), 4)
    ).astype(np.complex64) / np.sqrt(10),
    "64QAM": (
        np.tile(np.arange(-7, 8, 2), 8) + 1j * np.repeat(np.arange(-7, 8, 2), 8)
    ).astype(np.complex64) / np.sqrt(42),
}

def generate_sdr_frame(
    modulation: str = "QPSK",
    snr_db: float = 20.0,
    baud_rate: float = 50000.0,
    fs: float = 1000000.0,
    cfo_hz: float = 0.0,
    n_samples: int = 2048,
    frame_idx: int = 0,
) -> dict:
    """Synthesize an active SDR baseband signal frame with channel impairments."""
    mod_upper = modulation.upper().replace("-", "").replace(" ", "")
    sps = int(round(fs / max(100.0, baud_rate)))
    sps = max(2, min(sps, 64))
    n_symbols = n_samples // sps

    if mod_upper in _MOD_MAP:
        constellation = _MOD_MAP[mod_upper]
        sym_indices = np.random.randint(0, len(constellation), size=n_symbols)
        symbols = constellation[sym_indices]
        # Upsample & apply simple pulse shaping
        sig = np.repeat(symbols, sps)
    elif "FSK" in mod_upper:
        # FSK modulation
        m_ary = 4 if "4" in mod_upper else 2
        f_dev = baud_rate * 0.5
        bits = np.random.randint(0, m_ary, size=n_symbols)
        freq_offsets = (bits - (m_ary - 1) / 2.0) * (2 * f_dev / (m_ary - 1))
        freq_seq = np.repeat(freq_offsets, sps)
        phase = 2 * np.pi * np.cumsum(freq_seq) / fs
        sig = np.exp(1j * phase).astype(np.complex64)
    else:
        # Fallback QPSK
        constellation = _MOD_MAP["QPSK"]
        sym_indices = np.random.randint(0, len(constellation), size=n_symbols)
        symbols = constellation[sym_indices]
        sig = np.repeat(symbols, sps)

    # Pad or truncate to exact n_samples
    if len(sig) < n_samples:
        sig = np.pad(sig, (0, n_samples - len(sig)))
    else:
        sig = sig[:n_samples]

    # Carrier Frequency Offset & phase drift
    t = np.arange(n_samples) / fs
    if abs(cfo_hz) > 0.0:
        sig = sig * np.exp(1j * 2 * np.pi * cfo_hz * t)

    # Add AWGN noise according to target SNR
    sig_power = np.mean(np.abs(sig) ** 2)
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_power = sig_power / max(1e-6, snr_linear)
    noise = np.sqrt(noise_power / 2.0) * (
        np.random.randn(n_samples) + 1j * np.random.randn(n_samples)
    )
    noisy_sig = sig + noise

    # Parameter & Spectral Extractions
    freqs, psd = feature_extractor.compute_psd(noisy_sig, fs, nperseg=min(256, n_samples))
    psd_db = 10.0 * np.log10(np.maximum(psd, 1e-15))

    # Fast Modulation Classifier Prediction
    clf = classifier.ModulationClassifier()
    clf_res = clf.predict_with_confidence(noisy_sig, fs=fs)

    # Constellation subset (150 points)
    c_step = max(1, len(noisy_sig) // 150)
    c_subset = noisy_sig[::c_step][:150]
    constellation_pts = [
        {"i": round(float(pt.real), 4), "q": round(float(pt.imag), 4)}
        for pt in c_subset
    ]

    # Waveform subset (100 points)
    w_step = max(1, len(noisy_sig) // 100)
    wave_pts = [round(float(v.real), 4) for v in noisy_sig[::w_step][:100]]

    # PSD subset (60 bins)
    p_step = max(1, len(freqs) // 60)
    psd_pts = [
        {"freq": round(float(f)), "psd": round(float(p), 2)}
        for f, p in zip(freqs[::p_step], psd_db[::p_step])
    ]

    return {
        "type": "sdr_frame",
        "frame_idx": frame_idx,
        "timestamp": time.time(),
        "configured_modulation": modulation,
        "detected_modulation": clf_res["modulation"],
        "confidence": round(float(clf_res["confidence"]), 3),
        "snr_db": round(float(snr_db), 1),
        "baud_rate": round(float(baud_rate), 1),
        "sample_rate": round(float(fs), 1),
        "constellation": constellation_pts,
        "waveform": wave_pts,
        "psd": psd_pts,
    }


@router.websocket("/live")
async def sdr_live_stream_endpoint(websocket: WebSocket):
    """Real-time WebSocket streaming endpoint for SDR RF signal visualization."""
    await websocket.accept()
    
    # Default stream configuration
    config = {
        "modulation": "QPSK",
        "snr_db": 22.0,
        "baud_rate": 50000.0,
        "fs": 1000000.0,
        "cfo_hz": 500.0,
        "is_paused": False,
        "fps": 15,
    }

    frame_idx = 0

    async def client_listener():
        """Listen for client control messages (e.g. changing modulation or SNR on the fly)."""
        while True:
            try:
                data_text = await websocket.receive_text()
                msg = json.loads(data_text)
                action = msg.get("action")
                if action == "configure":
                    if "modulation" in msg:
                        config["modulation"] = str(msg["modulation"])
                    if "snr_db" in msg:
                        config["snr_db"] = float(msg["snr_db"])
                    if "baud_rate" in msg:
                        config["baud_rate"] = float(msg["baud_rate"])
                    if "cfo_hz" in msg:
                        config["cfo_hz"] = float(msg["cfo_hz"])
                    if "fps" in msg:
                        config["fps"] = max(1, min(30, int(msg["fps"])))
                elif action == "pause":
                    config["is_paused"] = True
                elif action == "resume":
                    config["is_paused"] = False
            except (WebSocketDisconnect, asyncio.CancelledError):
                break
            except Exception:
                pass

    listener_task = asyncio.create_task(client_listener())

    try:
        while True:
            if not config["is_paused"]:
                frame = generate_sdr_frame(
                    modulation=config["modulation"],
                    snr_db=config["snr_db"],
                    baud_rate=config["baud_rate"],
                    fs=config["fs"],
                    cfo_hz=config["cfo_hz"],
                    frame_idx=frame_idx,
                )
                await websocket.send_json(frame)
                frame_idx += 1

            # Control frame rate
            sleep_time = 1.0 / max(1, config["fps"])
            await asyncio.sleep(sleep_time)

    except (WebSocketDisconnect, asyncio.CancelledError):
        pass
    finally:
        listener_task.cancel()
