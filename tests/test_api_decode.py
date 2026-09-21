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
