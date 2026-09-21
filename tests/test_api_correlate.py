"""Unit tests for the /correlate API endpoint covering sync words,
direct bitstream correlation, frame synchronization, 180° phase inversion,
and preamble auto-discovery.
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient
from api.main import app
from ps26147_toolkit.correlator import STANDARD_SYNC_WORDS

client = TestClient(app)


def test_get_sync_words():
    """Verify /correlate/sync-words lists available preambles."""
    res = client.get("/correlate/sync-words")
    assert res.status_code == 200
    data = res.json()
    assert "Barker-13" in data
    assert "CCSDS-32" in data
    assert "DVB-S (0x47)" in data
    assert data["Barker-13"]["length"] == 13


def test_correlate_bits_barker13():
    """Verify /correlate/bits finds Barker-13 sync word in bitstream."""
    barker13 = STANDARD_SYNC_WORDS["Barker-13"].tolist()
    rng = np.random.default_rng(42)
    noise_prefix = rng.integers(0, 2, size=30).tolist()
    noise_suffix = rng.integers(0, 2, size=50).tolist()
    
    stream = noise_prefix + barker13 + noise_suffix
    
    res = client.post(
        "/correlate/bits",
        json={
            "bits": stream,
            "sync_word": "Barker-13",
            "threshold": 0.85,
        }
    )
    assert res.status_code == 200
    data = res.json()
    assert data["sync_found"] is True
    assert 30 in data["peak_indices"]
    assert data["max_correlation"] >= 0.99
    assert data["is_inverted"] is False
    assert len(data["bits"]) == len(stream)  # untruncated bits


def test_correlate_bits_inverted_sync():
    """Verify /correlate/bits detects 180° carrier phase inversion."""
    ccsds = STANDARD_SYNC_WORDS["CCSDS-32"]
    inv_ccsds = (1 - ccsds).tolist()
    
    rng = np.random.default_rng(99)
    noise = rng.integers(0, 2, size=100).tolist()
    stream = noise[:40] + inv_ccsds + noise[40:]
    
    res = client.post(
        "/correlate/bits",
        json={
            "bits": stream,
            "sync_word": "CCSDS-32",
            "tolerate_inverted": True,
            "threshold": 0.85,
        }
    )
    assert res.status_code == 200
    data = res.json()
    assert data["sync_found"] is True
    assert data["is_inverted"] is True
    assert 40 in data["peak_indices"]


def test_correlate_bits_frame_synchronization():
    """Verify /correlate/bits extracts aligned frames and payloads."""
    sync_word = STANDARD_SYNC_WORDS["CCSDS-32"].tolist()
    frame_len = 128
    num_frames = 4
    
    rng = np.random.default_rng(123)
    stream = []
    for _ in range(num_frames):
        payload = rng.integers(0, 2, size=frame_len - len(sync_word)).tolist()
        stream.extend(sync_word + payload)
        
    res = client.post(
        "/correlate/bits",
        json={
            "bits": stream,
            "sync_word": "CCSDS-32",
            "frame_length": frame_len,
            "threshold": 0.90,
        }
    )
    assert res.status_code == 200
    data = res.json()
    assert data["sync_found"] is True
    assert data["num_frames"] == num_frames
    assert len(data["frames"]) == num_frames
    assert len(data["frames"][0]["payload_bits"]) == frame_len - len(sync_word)


def test_correlate_bits_auto_discovery():
    """Verify /correlate/bits auto-discovers DVB-S repeating sync pattern."""
    sync_word = STANDARD_SYNC_WORDS["DVB-S (0x47)"].tolist()
    frame_len = 188 * 8
    num_frames = 4
    
    rng = np.random.default_rng(555)
    stream = []
    for _ in range(num_frames):
        payload = rng.integers(0, 2, size=frame_len - len(sync_word)).tolist()
        stream.extend(sync_word + payload)
        
    res = client.post(
        "/correlate/bits",
        json={
            "bits": stream,
            "auto_discover": True,
        }
    )
    assert res.status_code == 200
    data = res.json()
    assert data["preamble_discovery"] is not None
    assert data["preamble_discovery"]["discovered"] is True
    assert data["preamble_discovery"]["matched_standard_sync"] == "DVB-S (0x47)"


def test_correlate_signal_file():
    """Verify POST /correlate/file endpoint with synthetic IQ file."""
    # Synthesize simple BPSK signal
    rng = np.random.default_rng(42)
    fs = 1_000_000.0
    baud = 25_000.0
    sps = int(fs / baud)
    sync = STANDARD_SYNC_WORDS["Barker-13"]
    payload = rng.integers(0, 2, 200)
    bits = np.concatenate([sync, payload])
    
    baseband = np.repeat(np.where(bits == 1, 1.0, -1.0), sps)
    t = np.arange(len(baseband)) / fs
    sig = (baseband * np.exp(1j * 2 * np.pi * 50_000 * t)).astype(np.complex64)
    raw_iq = sig.tobytes()
    
    res = client.post(
        "/correlate/file?fs=1000000.0&sync_word=Barker-13",
        files={"file": ("test_signal.iq", raw_iq, "application/octet-stream")},
    )
    assert res.status_code == 200
    data = res.json()
    assert "status" in data
    assert "sync_found" in data
    assert "bits" in data
    assert len(data["bits"]) > 0
    assert data["num_bits"] == len(data["bits"])
