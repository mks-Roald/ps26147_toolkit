"""Live physically-based SDR waveform simulator and analyzer."""
import asyncio
import json
import time

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ps26147_toolkit.analysis import analyze_signal
from ps26147_toolkit.feature_extractor import compute_psd, rrc_filter
from ps26147_toolkit.demodulator import demodulate_signal

router = APIRouter()
_LEVELS = np.arange(-7, 8, 2)


def _symbols(modulation: str, n: int, rng: np.random.Generator) -> np.ndarray:
    name = modulation.upper().replace("-", "").replace(" ", "")
    if name == "BPSK":
        return rng.choice([-1., 1.], n).astype(np.complex64)
    if name == "QPSK":
        # Match the classifier corpus: independent equiprobable I/Q signs.
        return ((rng.choice([-1., 1.], n) + 1j * rng.choice([-1., 1.], n)) / np.sqrt(2)).astype(np.complex64)
    if name == "8PSK":
        return np.exp(1j * rng.integers(0, 8, n) * np.pi / 4).astype(np.complex64)
    if name == "16QAM":
        return ((rng.choice([-3., -1., 1., 3.], n) + 1j * rng.choice([-3., -1., 1., 3.], n)) / np.sqrt(10)).astype(np.complex64)
    if name == "64QAM":
        return ((rng.choice(_LEVELS, n) + 1j * rng.choice(_LEVELS, n)) / np.sqrt(42)).astype(np.complex64)
    raise ValueError(f"Unsupported modulation: {modulation}")


def _waveform(modulation: str, baud_rate: float, fs: float, n_samples: int, rng: np.random.Generator):
    sps = max(2, min(64, int(round(fs / baud_rate))))
    n_symbols = int(np.ceil(n_samples / sps)) + 8
    name = modulation.upper().replace("-", "").replace(" ", "")
    if name in {"BPSK", "QPSK", "8PSK", "16QAM", "64QAM"}:
        syms = _symbols(name, n_symbols, rng)
        up = np.zeros(n_symbols * sps, dtype=np.complex64)
        up[::sps] = syms
        # Same classifier-corpus RRC (49 taps, alpha=.35, unit energy), including
        # the same same-mode convolution convention.
        base = np.convolve(up, rrc_filter(49, .35, sps), mode="same")[:n_samples]
    elif name in {"2FSK", "4FSK"}:
        order = 2 if name == "2FSK" else 4
        idx = rng.integers(0, order, n_symbols)
        # Deviation is the outer tone offset: +/-Rs/2 for 2FSK and
        # {-3,-1,+1,+3} Rs/2 for 4FSK (adjacent spacing Rs).
        tones = (idx - (order - 1) / 2) * baud_rate
        offsets = np.repeat(tones, sps)[:n_samples]
        # Integrate frequency continuously; phase is carried across boundaries.
        base = np.exp(1j * (2 * np.pi * np.cumsum(offsets) / fs)).astype(np.complex64)
    else:
        raise ValueError(f"Unsupported modulation: {modulation}")
    return base, sps


def generate_sdr_frame(modulation="QPSK", snr_db=20., baud_rate=50000., fs=1e6,
                       cfo_hz=0., n_samples=2048, frame_idx=0, seed=None):
    """Generate one channel-impaired frame and run the shared signal analysis."""
    rng = np.random.default_rng(seed)
    signal, sps = _waveform(modulation, baud_rate, fs, n_samples, rng)
    # Random initial carrier phase, CFO, then AWGN measured against actual signal power.
    phase = rng.uniform(-np.pi, np.pi)
    t = np.arange(n_samples) / fs
    signal = signal * np.exp(1j * (phase + 2 * np.pi * cfo_hz * t))
    p = np.mean(np.abs(signal) ** 2)
    noise_power = p / (10 ** (snr_db / 10))
    noisy = signal + np.sqrt(noise_power / 2) * (rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples))

    # The live view needs prediction/confidence but not the extra HOC/rule
    # diagnostics; the canonical analysis still performs parameter estimation,
    # baseband conversion, and the same fitted classifier prediction.
    analysis = analyze_signal(noisy, fs, include_diagnostics=False)
    detected = analysis.classification["modulation"]
    params = analysis.parameters
    detected_baud = float(params.get("baud_rate", 0) or 0)
    # Modulation-aware symbol recovery. The simulator's supplied baud is used as
    # the timing initial condition; detection itself always comes from analysis.
    # The known simulated format is a valid demodulator side-information hint.
    # Keep it separate from detected_modulation, which remains the independent
    # classifier decision above. This prevents classifier errors from rotating
    # or mis-slicing the recovered 8PSK/QAM points shown in the plot.
    demod = demodulate_signal(analysis.baseband_signal, fs, modulation,
                              center_freq=0., baud_rate=baud_rate)
    symbols = np.asarray(demod.get("symbols", []))
    if "FSK" in modulation.upper():
        freq = np.angle(noisy[1:] * np.conj(noisy[:-1])) * fs / (2 * np.pi)
        view = [{"sample": int(i), "frequency": round(float(v), 1)} for i, v in enumerate(freq[::max(1, len(freq)//150)][:150])]
        constellation = []
    else:
        view = []
        take = symbols[::max(1, len(symbols)//150)][:150]
        constellation = [{"i": round(float(v.real), 4), "q": round(float(v.imag), 4)} for v in take]

    freqs, psd = compute_psd(noisy, fs, nperseg=min(256, n_samples))
    psd_db = 10 * np.log10(np.maximum(psd, 1e-15))
    idx = np.arange(0, len(freqs), max(1, len(freqs)//60))[:60]
    wave = noisy.real[::max(1, n_samples//100)][:100]
    return {
        "type": "sdr_frame", "frame_idx": frame_idx, "timestamp": time.time(),
        "configured_modulation": modulation, "detected_modulation": detected,
        "classification_correct": detected == modulation,
        "confidence": round(float(analysis.classification.get("confidence", 0)), 3),
        "snr_db": float(snr_db), "configured_snr": float(snr_db),
        "estimated_snr": params.get("snr_db"), "baud_rate": float(baud_rate),
        "configured_baud": float(baud_rate), "estimated_baud": detected_baud,
        "configured_cfo": float(cfo_hz), "estimated_cfo": params.get("center_frequency_hz"),
        "sample_rate": float(fs), "constellation": constellation,
        "frequency_states": view, "visualization": "frequency" if "FSK" in modulation.upper() else "constellation",
        "waveform": [round(float(v), 4) for v in wave],
        "psd": [{"freq": round(float(freqs[i])), "psd": round(float(psd_db[i]), 2)} for i in idx],
    }


@router.websocket("/live")
async def sdr_live_stream_endpoint(websocket: WebSocket):
    await websocket.accept()
    config = {"modulation":"QPSK", "snr_db":22., "baud_rate":50000., "fs":1e6, "cfo_hz":500., "is_paused":False, "fps":30}
    async def listener():
        while True:
            try:
                msg = json.loads(await websocket.receive_text())
                if msg.get("action") == "configure":
                    for key in ("modulation", "snr_db", "baud_rate", "cfo_hz"):
                        if key in msg: config[key] = str(msg[key]) if key == "modulation" else float(msg[key])
                    if "fps" in msg: config["fps"] = max(1, min(30, int(msg["fps"])))
                elif msg.get("action") == "pause": config["is_paused"] = True
                elif msg.get("action") == "resume": config["is_paused"] = False
            except (WebSocketDisconnect, asyncio.CancelledError): break
            except Exception: pass
    task = asyncio.create_task(listener())
    frame = 0
    try:
        while True:
            if not config["is_paused"]:
                payload = {k:v for k,v in config.items() if k != "is_paused" and k != "fps"}
                result = await asyncio.to_thread(generate_sdr_frame, **payload, frame_idx=frame)
                await websocket.send_json(result)
                frame += 1
            await asyncio.sleep(1/max(1, config["fps"]))
    except (WebSocketDisconnect, asyncio.CancelledError): pass
    finally: task.cancel()
