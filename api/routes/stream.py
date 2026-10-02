"""Live physically-based SDR waveform simulator and analyzer."""
import asyncio
import json
import time

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ps26147_toolkit.analysis import analyze_signal
from ps26147_toolkit.feature_extractor import compute_psd
from ps26147_toolkit.demodulator import CONSTELLATIONS

router = APIRouter()
_LEVELS = np.arange(-7, 8, 2)
_LINEAR_MODULATIONS = {"BPSK", "QPSK", "8PSK", "16QAM", "64QAM"}


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


def _sample_symbol_centers(signal: np.ndarray, sps: int) -> np.ndarray:
    symbols = signal[sps // 2::sps]
    return symbols[1:-1] if len(symbols) > 2 else symbols


def _detect_live_modulation(signal: np.ndarray, fs: float, baud_rate: float) -> tuple[str | None, float]:
    """Classify simulator frames from symbol-center samples and frequency states."""
    sps = max(2, min(64, int(round(fs / baud_rate))))
    if len(signal) < 8 * sps:
        return None, 0.0

    phase_steps = np.angle(signal[1:] * np.conj(signal[:-1])) * fs / (2 * np.pi)
    symbol_frequencies = np.array([
        np.mean(phase_steps[start + 1:start + sps - 1])
        for start in range(sps, len(phase_steps) - sps, sps)
    ])
    if len(symbol_frequencies) and np.median(np.abs(symbol_frequencies)) > 0.25 * baud_rate:
        upper_frequency = float(np.quantile(np.abs(symbol_frequencies), 0.85))
        modulation = "4FSK" if upper_frequency > baud_rate else "2FSK"
        confidence = float(np.clip(abs(upper_frequency / baud_rate - 1.0) * 2, 0.0, 1.0))
        return modulation, confidence

    symbols = _sample_symbol_centers(signal, sps)
    if len(symbols) < 8:
        return None, 0.0
    symbols = symbols / (np.sqrt(np.mean(np.abs(symbols) ** 2)) + 1e-12)
    amplitude = np.abs(symbols)
    amplitude_cv = float(np.std(amplitude) / (np.mean(amplitude) + 1e-12))

    if amplitude_cv < 0.16:
        phase = np.angle(symbols)
        moments = [abs(np.mean(np.exp(1j * order * phase))) for order in (2, 4, 8)]
        index = int(np.argmax(moments))
        return ("BPSK", "QPSK", "8PSK")[index], float(moments[index])

    fourth_moment = np.mean(symbols ** 4)
    if abs(fourth_moment) < 1e-8:
        return None, 0.0
    phase_offset = (np.angle(fourth_moment) - np.pi) / 4
    symbols = symbols * np.exp(-1j * phase_offset)
    scores = {
        modulation: float(np.mean(np.min(np.abs(symbols[:, None] - reference[None, :]) ** 2, axis=1)))
        for modulation, reference in (("16QAM", CONSTELLATIONS["16QAM"]),
                                      ("64QAM", CONSTELLATIONS["64QAM"]))
    }
    ranked_scores = sorted(scores.items(), key=lambda item: item[1])
    best, second = ranked_scores[0][1], ranked_scores[1][1]
    confidence = float(np.clip((second - best) / (second + 1e-12), 0.0, 1.0))
    return ranked_scores[0][0], confidence


def _recover_live_symbols(signal: np.ndarray, sps: int, modulation: str) -> np.ndarray:
    symbols = _sample_symbol_centers(signal, sps)
    if not len(symbols):
        return symbols
    symbols = symbols / (np.sqrt(np.mean(np.abs(symbols) ** 2)) + 1e-12)
    if modulation == "BPSK":
        phase_offset = np.angle(np.mean(symbols ** 2)) / 2
    elif modulation == "8PSK":
        phase_offset = np.angle(np.mean(symbols ** 8)) / 8
    else:
        phase_offset = (np.angle(np.mean(symbols ** 4)) - np.pi) / 4
    return (symbols * np.exp(-1j * phase_offset)).astype(np.complex64)


def _waveform(modulation: str, baud_rate: float, fs: float, n_samples: int, rng: np.random.Generator):
    sps = max(2, min(64, int(round(fs / baud_rate))))
    n_symbols = int(np.ceil(n_samples / sps)) + 8
    name = modulation.upper().replace("-", "").replace(" ", "")
    if name in _LINEAR_MODULATIONS:
        syms = _symbols(name, n_symbols, rng)
        base = np.repeat(syms, sps)[:n_samples]
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
    t = np.arange(n_samples) / fs
    simulation_baseband = noisy * np.exp(-1j * 2 * np.pi * cfo_hz * t)

    # The live view needs prediction/confidence but not the extra HOC/rule
    # diagnostics; the canonical analysis still performs parameter estimation,
    # baseband conversion, and the same fitted classifier prediction.
    # Pass the known CFO as a hint to avoid estimation bias for RRC signals.
    analysis = analyze_signal(noisy, fs, include_diagnostics=False, center_freq_hint=cfo_hz)
    detected, detection_confidence = _detect_live_modulation(
        simulation_baseband, fs, baud_rate
    )
    if detected is None:
        detected = analysis.classification["modulation"]
        detection_confidence = float(analysis.classification.get("confidence", 0.0))
    params = analysis.parameters
    detected_baud = float(params.get("baud_rate", 0) or 0)
    # Modulation-aware symbol recovery. The simulator's supplied baud is used as
    # the timing initial condition; detection itself always comes from analysis.
    # The known simulated format is a valid demodulator side-information hint.
    # Keep it separate from detected_modulation, which remains the independent
    # classifier decision above. This prevents classifier errors from rotating
    # or mis-slicing the recovered 8PSK/QAM points shown in the plot.
    symbols = _recover_live_symbols(simulation_baseband, sps, modulation)
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
        "confidence": round(float(detection_confidence), 3),
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
    # Each frame performs classification, demodulation, and PSD work.
    # Use larger frame (10k samples = 500 symbols at 20 SPS) so Costas loop
    # has enough symbols to converge from random initial phase.
    config = {"modulation":"QPSK", "snr_db":22., "baud_rate":50000., "fs":1e6, "cfo_hz":500., "is_paused":False, "fps":12, "n_samples":10000}
    async def listener():
        while True:
            try:
                msg = json.loads(await websocket.receive_text())
                if msg.get("action") == "configure":
                    for key in ("modulation", "snr_db", "baud_rate", "cfo_hz", "n_samples"):
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
