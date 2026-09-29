"""Dashboard visualization payloads must match the modulation's signal model."""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import app
from ps26147_toolkit import analysis as analysis_module


client = TestClient(app)
FS = 1_000_000.0
BAUD = 25_000.0
SPS = int(FS / BAUD)


def _signal(modulation: str) -> bytes:
    rng = np.random.default_rng(26147)
    count = 512
    if modulation == "BPSK":
        symbols = rng.choice([-1, 1], count)
    elif modulation == "QPSK":
        symbols = np.exp(1j * rng.integers(0, 4, count) * np.pi / 2)
    elif modulation == "8PSK":
        symbols = np.exp(1j * rng.integers(0, 8, count) * np.pi / 4)
    elif modulation in {"16QAM", "64QAM"}:
        levels = np.array([-3, -1, 1, 3] if modulation == "16QAM" else [-7, -5, -3, -1, 1, 3, 5, 7])
        symbols = rng.choice(levels, count) + 1j * rng.choice(levels, count)
    else:
        state_count = 2 if modulation == "2FSK" else 4
        states = rng.integers(0, state_count, count)
        deviations = (states - (state_count - 1) / 2) * BAUD / 2
        freq = np.repeat(deviations, SPS)
        phase = np.cumsum(2 * np.pi * freq / FS)
        return np.exp(1j * phase).astype(np.complex64).tobytes()
    iq = np.repeat(symbols, SPS).astype(np.complex64)
    return iq.tobytes()


@pytest.mark.parametrize("modulation", ["BPSK", "QPSK", "8PSK", "16QAM", "64QAM"])
def test_psk_qam_api_returns_recovered_symbols_never_waveform_as_constellation(monkeypatch, modulation):
    original = analysis_module.analyze_signal

    def analyze(signal, fs, modulation=None):
        result = original(signal, fs, modulation)
        result.classification = dict(result.classification, modulation=modulation_name, confidence=1.0)
        return result

    modulation_name = modulation
    monkeypatch.setattr("api.routes.process.analyze_signal", analyze)
    response = client.post("/process/file", params={"fs": FS}, files={"file": ("test.iq", _signal(modulation), "application/octet-stream")})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["constellation_metadata"]["representation"] == "recovered_symbols"
    assert body["constellation_metadata"]["timing_recovery_used"] is True
    assert body["constellation_metadata"]["carrier_recovery_used"] is True
    assert body["recovered_symbols"] and len(body["recovered_symbols"]) <= 2000
    assert body["constellation_data"] is None
    raw_first = complex(np.frombuffer(_signal(modulation), dtype=np.complex64)[0])
    first = body["recovered_symbols"][0]
    assert not (np.isclose(first["i"], raw_first.real) and np.isclose(first["q"], raw_first.imag))


@pytest.mark.parametrize("modulation,state_count", [("2FSK", 2), ("4FSK", 4)])
def test_fsk_api_returns_frequency_visualization_and_no_constellation(monkeypatch, modulation, state_count):
    original = analysis_module.analyze_signal

    def analyze(signal, fs, modulation=None):
        result = original(signal, fs, modulation)
        result.classification = dict(result.classification, modulation=modulation_name, confidence=1.0)
        return result

    modulation_name = modulation
    monkeypatch.setattr("api.routes.process.analyze_signal", analyze)
    response = client.post("/process/file", params={"fs": FS}, files={"file": ("test.iq", _signal(modulation), "application/octet-stream")})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["constellation_data"] is None
    assert body["recovered_symbols"] is None
    fsk = body["fsk_visualization_data"]
    assert fsk["frequency_state_count"] == state_count
    assert len(fsk["instantaneous_frequency"]) > 0
    assert len(fsk["symbol_frequency_values"]) > 0
    assert len(fsk["recovered_frequency_states"]) == state_count
