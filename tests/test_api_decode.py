"""Unit tests for the /decode API endpoint covering pre-FEC demodulated bits,
deinterleaving, and untruncated bitstreams.
"""

import io
import numpy as np
import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def _generate_synthetic_bpsk(num_bits: int = 1024, fs: float = 1_000_000.0, baud: float = 25_000.0) -> bytes:
    """Generate a clean BPSK signal in complex64 format with >512 bits."""
    rng = np.random.default_rng(42)
    sps = int(fs / baud)
    syms = rng.integers(0, 2, num_bits)
    baseband = np.repeat(np.where(syms == 1, 1.0, -1.0), sps)
    t = np.arange(len(baseband)) / fs
    sig = (baseband * np.exp(1j * 2 * np.pi * 50_000 * t)).astype(np.complex64)
    # Add minimal noise
    noise = (rng.normal(0, 0.01, len(sig)) + 1j * rng.normal(0, 0.01, len(sig))).astype(np.complex64)
    sig = sig + noise
    return sig.tobytes()


def test_decode_endpoint_returns_untruncated_bits_and_pre_fec():
    """Verify that /decode returns untruncated bit arrays (> 512 bits) and pre-FEC demodulated bits."""
    num_bits = 800
    raw_iq = _generate_synthetic_bpsk(num_bits=num_bits)
    
    response = client.post(
        "/decode/?fs=1000000.0&fec_scheme=none",
        files={"file": ("test_signal.iq", raw_iq, "application/octet-stream")},
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # 1. Check existing fields
    assert "modulation" in data
    assert "confidence" in data
    assert "num_bits" in data
    assert "decoded_bits" in data
    assert "decoded_bits_count" in data
    
    # 2. Verify [:512] cap is removed (untruncated bits)
    assert len(data["decoded_bits"]) > 512, f"Expected > 512 bits, got {len(data['decoded_bits'])}"
    assert data["decoded_bits_count"] == len(data["decoded_bits"])
    
    # 3. Verify pre-FEC demodulated bits are exposed
    assert "demodulated_bits" in data
    assert data["demodulated_bits"] is not None
    assert len(data["demodulated_bits"]) > 512
    assert "demodulated_bits_count" in data
    assert data["demodulated_bits_count"] == len(data["demodulated_bits"])


def test_decode_endpoint_deinterleaving_fields():
    """Verify that /decode populates deinterleaver outputs."""
    raw_iq = _generate_synthetic_bpsk(num_bits=600)
    
    response = client.post(
        "/decode/?fs=1000000.0&fec_scheme=none&auto_deinterleave=true",
        files={"file": ("test_signal.iq", raw_iq, "application/octet-stream")},
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Check deinterleaver fields presence
    assert "deinterleaved_bits" in data
    assert data["deinterleaved_bits"] is not None
    assert len(data["deinterleaved_bits"]) == len(data["decoded_bits"])
    assert "deinterleaver_method" in data
    assert data["deinterleaver_method"] is not None
    assert "deinterleaver_params" in data
    assert "deinterleaver_entropy" in data
    assert "deinterleaver_baseline_entropy" in data


def test_decode_endpoint_fec_hex_and_ascii():
    """Verify that /decode populates decoded_hex and decoded_ascii correctly."""
    num_bits = 64
    raw_iq = _generate_synthetic_bpsk(num_bits=num_bits)

    response = client.post(
        "/decode/?fs=1000000.0&fec_scheme=none",
        files={"file": ("test_signal.iq", raw_iq, "application/octet-stream")},
    )

    assert response.status_code == 200
    data = response.json()

    assert "decoded_hex" in data
    assert "decoded_ascii" in data
    assert data["decoded_hex"] is not None
    assert data["decoded_ascii"] is not None
    assert len(data["decoded_hex"]) > 0
    assert len(data["decoded_ascii"]) > 0

    # Verify hex formatting: space-separated uppercase hex pairs
    hex_parts = data["decoded_hex"].split()
    assert len(hex_parts) == int(np.ceil(len(data["decoded_bits"]) / 8))
    for part in hex_parts:
        assert len(part) == 2
        assert part == part.upper()
        int(part, 16)  # Valid hex integer


def test_decode_endpoint_frame_sync_with_sync_word():
    """Verify that /decode correctly synchronizes to a provided sync word and slices the bitstream."""
    rng = np.random.default_rng(42)
    fs = 1_000_000.0
    baud = 25_000.0
    sps = int(fs / baud)
    from ps26147_toolkit.correlator import STANDARD_SYNC_WORDS
    sync = STANDARD_SYNC_WORDS["Barker-13"]
    payload = rng.integers(0, 2, 500)
    bits = np.concatenate([sync, payload])

    baseband = np.repeat(np.where(bits == 1, 1.0, -1.0), sps)
    t = np.arange(len(baseband)) / fs
    sig = (baseband * np.exp(1j * 2 * np.pi * 50_000 * t)).astype(np.complex64)
    raw_iq = sig.tobytes()

    response = client.post(
        "/decode/?fs=1000000.0&fec_scheme=none&sync_word=Barker-13",
        files={"file": ("test_signal.iq", raw_iq, "application/octet-stream")},
    )

    assert response.status_code == 200
    data = response.json()
    assert "sync_offset" in data
    assert data["sync_offset"] is not None
    assert data["sync_confidence"] is not None
    assert data["sync_confidence"] >= 0.90
    assert data["sync_method"] == "sync_word"
    assert data["decoded_bits_count"] > 0


def test_decode_endpoint_frame_sync_fallback_when_unmatched():
    """Verify that /decode proceeds unaligned with confidence=0, method='none' when sync fails."""
    num_bits = 120
    raw_iq = _generate_synthetic_bpsk(num_bits=num_bits)

    # Search for an impossible sync word that doesn't appear
    response = client.post(
        "/decode/?fs=1000000.0&fec_scheme=none&sync_word=DEADBEEFCAFE&auto_detect_sync=false",
        files={"file": ("test_signal.iq", raw_iq, "application/octet-stream")},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["sync_offset"] is None
    assert data["sync_confidence"] == 0.0
    assert data["sync_method"] == "none"
    assert len(data["decoded_bits"]) > 0


def test_decode_endpoint_llr_mismatch_regression_guard(monkeypatch):
    """Verify regression tripwire: 500 error pointing at compute_soft_llr() when len(llr) != len(raw_bits)."""
    raw_iq = _generate_synthetic_bpsk(num_bits=64)

    # Monkeypatch demodulator.demodulate_signal to return mismatched LLR length
    from ps26147_toolkit import demodulator as demod_module
    orig_demod = demod_module.demodulate_signal

    def mock_demod(*args, **kwargs):
        res = orig_demod(*args, **kwargs)
        # Introduce deliberate mismatch
        res["llr"] = np.array([1.0, -1.0])
        return res

    monkeypatch.setattr(demod_module, "demodulate_signal", mock_demod)

    response = client.post(
        "/decode/?fs=1000000.0&fec_scheme=none",
        files={"file": ("test_signal.iq", raw_iq, "application/octet-stream")},
    )

    assert response.status_code == 500
    detail = response.json()["detail"]
    assert "compute_soft_llr" in detail
    assert "LLR length mismatch" in detail


def test_full_pipeline_bpsk_hello_world():
    from ps26147_toolkit.correlator import STANDARD_SYNC_WORDS
    from ps26147_toolkit.fec_decoders import ConvolutionalCodec

    text = "hello world\n" * 4                      # 384 bits
    payload = np.unpackbits(np.frombuffer(text.encode(), dtype=np.uint8))

    enc = ConvolutionalCodec().encode(payload, flush=True)   # viterbi first
    pad = np.random.default_rng(7).integers(0, 2, (-len(enc)) % 256).astype(np.uint8)
    enc = np.concatenate([enc, pad])                          # random pad, NOT zeros

    inter = np.concatenate([                                  # then 16x16 block interleave
        enc[i:i + 256].reshape(16, 16, order="C").ravel(order="F")
        for i in range(0, len(enc), 256)
    ])

    lead = np.random.default_rng(11).integers(0, 2, 64).astype(np.uint8)  # 64 random lead-in bits
    sync = STANDARD_SYNC_WORDS["Barker-13"].astype(np.uint8)
    bits = np.concatenate([lead, sync, inter])

    fs, baud = 1_000_000.0, 25_000.0
    sps = int(fs / baud)
    base = np.repeat(np.where(bits == 1, 1.0, -1.0), sps)
    t = np.arange(len(base)) / fs
    rng = np.random.default_rng(42)
    sig = (base * np.exp(1j * 2 * np.pi * 50_000 * t)).astype(np.complex64)
    sig += (rng.normal(0, 0.01, len(sig)) + 1j * rng.normal(0, 0.01, len(sig))).astype(np.complex64)

    r = client.post(
        "/decode/?fs=1000000.0&fec_scheme=viterbi&sync_word=Barker-13&auto_deinterleave=true",
        files={"file": ("hw.iq", sig.tobytes(), "application/octet-stream")},
    )
    assert r.status_code == 200
    d = r.json()
    print("mod:", d.get("modulation"), "| sync:", d.get("sync_offset"), d.get("sync_confidence"),
          "| deint:", d.get("deinterleaver_method"), "| bits:", d.get("demodulated_bits_count"),
          "->", d.get("decoded_bits_count"), "| ascii:", d["decoded_ascii"][:60])
    assert "hello world" in d["decoded_ascii"]
