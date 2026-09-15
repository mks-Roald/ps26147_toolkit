"""
tests/test_preprocess.py
SOP Phase 1.1 – Unit tests for Multi-Dtype IQ Loader & SignalMetadata

Tests per SOP 8.1 §1:
  - Round-trip int8 / int16 / uint8 / float32 / complex64 IQ loading
  - Auto-dtype probing correctness
  - Odd-sample error guard
  - SignalMetadata population & fs validation
  - SigMF companion loading
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ps26147_toolkit.preprocess import (
    SignalMetadata,
    _probe_dtype,
    load_iq,
    load_iq_with_sigmf,
    load_sigmf_meta,
    load_wav,
    segment_signal,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_iq_file(path: Path, i_vals: np.ndarray, q_vals: np.ndarray, dtype) -> None:
    """Interleave I and Q, cast to dtype, write binary file."""
    interleaved = np.empty(len(i_vals) + len(q_vals), dtype=dtype)
    interleaved[0::2] = i_vals.astype(dtype)
    interleaved[1::2] = q_vals.astype(dtype)
    interleaved.tofile(str(path))


N = 1024        # number of complex IQ samples per test
FS = 1_000_000.0  # 1 MHz


# ---------------------------------------------------------------------------
# SignalMetadata validation
# ---------------------------------------------------------------------------

class TestSignalMetadata:
    def test_valid_construction(self):
        m = SignalMetadata(fs=1e6, source_format="iq", dtype="float32",
                          num_samples=1000, duration_sec=0.001)
        assert m.fs == 1e6
        assert m.num_samples == 1000

    def test_zero_fs_raises(self):
        with pytest.raises(ValueError, match="fs must be positive"):
            SignalMetadata(fs=0.0)

    def test_negative_fs_raises(self):
        with pytest.raises(ValueError, match="fs must be positive"):
            SignalMetadata(fs=-500.0)


# ---------------------------------------------------------------------------
# int16 round-trip
# ---------------------------------------------------------------------------

class TestInt16Loading:
    def test_values_in_range(self, tmp_path):
        rng = np.random.default_rng(42)
        i_vals = (rng.uniform(-0.9, 0.9, N) * 32767).astype(np.int16)
        q_vals = (rng.uniform(-0.9, 0.9, N) * 32767).astype(np.int16)
        p = tmp_path / "test.iq"
        _write_iq_file(p, i_vals, q_vals, np.int16)

        iq, meta = load_iq(str(p), dtype="int16", fs=FS)

        assert iq.dtype == np.complex64
        assert iq.size == N
        assert meta.dtype == "int16"
        assert meta.fs == FS
        assert meta.num_samples == N
        # Each channel independently normalised to [-1, 1]
        assert np.max(np.abs(iq.real)) <= 1.01
        assert np.max(np.abs(iq.imag)) <= 1.01

    def test_normalisation_accuracy(self, tmp_path):
        """Full-scale +32767 / -32767 → ≈ ±1.0."""
        p = tmp_path / "fs16.iq"
        data = np.array([32767, -32767, 16383, -16383], dtype=np.int16)
        data.tofile(str(p))
        iq, _ = load_iq(str(p), dtype="int16", fs=FS)
        np.testing.assert_allclose(iq.real, [32767 / 32767.0, 16383 / 32767.0], atol=1e-4)
        np.testing.assert_allclose(iq.imag, [-32767 / 32767.0, -16383 / 32767.0], atol=1e-4)


# ---------------------------------------------------------------------------
# int8 round-trip
# ---------------------------------------------------------------------------

class TestInt8Loading:
    def test_values_in_range(self, tmp_path):
        rng = np.random.default_rng(0)
        i_vals = (rng.uniform(-0.9, 0.9, N) * 127).astype(np.int8)
        q_vals = (rng.uniform(-0.9, 0.9, N) * 127).astype(np.int8)
        p = tmp_path / "test8.iq"
        _write_iq_file(p, i_vals, q_vals, np.int8)

        iq, meta = load_iq(str(p), dtype="int8", fs=FS)

        assert iq.dtype == np.complex64
        assert iq.size == N
        assert meta.dtype == "int8"
        assert np.max(np.abs(iq.real)) <= 1.01
        assert np.max(np.abs(iq.imag)) <= 1.01


# ---------------------------------------------------------------------------
# uint8 (RTL-SDR offset-127) round-trip
# ---------------------------------------------------------------------------

class TestUint8Loading:
    def test_values_centered(self, tmp_path):
        rng = np.random.default_rng(7)
        # Offset-binary: 0→-1, 127→0, 255→+1
        i_vals = (rng.uniform(-0.9, 0.9, N) * 127.5 + 127.5).astype(np.uint8)
        q_vals = (rng.uniform(-0.9, 0.9, N) * 127.5 + 127.5).astype(np.uint8)
        p = tmp_path / "rtlsdr.iq"
        _write_iq_file(p, i_vals, q_vals, np.uint8)

        iq, meta = load_iq(str(p), dtype="uint8", fs=FS)

        assert iq.dtype == np.complex64
        assert iq.size == N
        assert meta.dtype == "uint8"
        assert np.max(np.abs(iq.real)) <= 1.01
        assert np.max(np.abs(iq.imag)) <= 1.01

    def test_dc_level(self, tmp_path):
        """All-127 file → DC offset after normalization."""
        p = tmp_path / "dc.iq"
        data = np.full(N * 2, 127, dtype=np.uint8)
        data.tofile(str(p))
        iq, _ = load_iq(str(p), dtype="uint8", fs=FS)
        expected = (127 - 127.5) / 127.5
        np.testing.assert_allclose(iq.real, np.full(N, expected), atol=1e-4)
        np.testing.assert_allclose(iq.imag, np.full(N, expected), atol=1e-4)


# ---------------------------------------------------------------------------
# float32 round-trip
# ---------------------------------------------------------------------------

class TestFloat32Loading:
    def test_round_trip(self, tmp_path):
        rng = np.random.default_rng(1)
        i_vals = rng.uniform(-1, 1, N).astype(np.float32)
        q_vals = rng.uniform(-1, 1, N).astype(np.float32)
        p = tmp_path / "f32.iq"
        _write_iq_file(p, i_vals, q_vals, np.float32)

        iq, meta = load_iq(str(p), dtype="float32", fs=FS)

        assert iq.dtype == np.complex64
        assert iq.size == N
        assert meta.dtype == "float32"
        np.testing.assert_allclose(iq.real, i_vals, atol=1e-6)
        np.testing.assert_allclose(iq.imag, q_vals, atol=1e-6)


# ---------------------------------------------------------------------------
# complex64 round-trip
# ---------------------------------------------------------------------------

class TestComplex64Loading:
    def test_round_trip(self, tmp_path):
        rng = np.random.default_rng(2)
        iq_orig = (rng.standard_normal(N) + 1j * rng.standard_normal(N)).astype(np.complex64) * 0.5
        p = tmp_path / "c64.iq"
        iq_orig.tofile(str(p))

        iq, meta = load_iq(str(p), dtype="complex64", fs=FS)

        assert iq.dtype == np.complex64
        assert iq.size == N
        assert meta.dtype == "complex64"
        np.testing.assert_allclose(iq.real, iq_orig.real, atol=1e-6)
        np.testing.assert_allclose(iq.imag, iq_orig.imag, atol=1e-6)


# ---------------------------------------------------------------------------
# Auto-probe detection
# ---------------------------------------------------------------------------

class TestAutoDtypeProbing:
    def test_probe_short_file_fallback_float32(self, tmp_path):
        """Fragments < 8 bytes fall back to float32 per SOP 1.1 (§2)."""
        p = tmp_path / "tiny.iq"
        p.write_bytes(b"\x00\x01\x02")
        assert _probe_dtype(p) == "float32"

    def test_probe_int16(self, tmp_path):
        """Standard int16 SDR capture correctly detected as int16."""
        rng = np.random.default_rng(99)
        vals = (rng.uniform(-0.9, 0.9, N * 2) * 32767).astype(np.int16)
        p = tmp_path / "auto16.iq"
        vals.tofile(str(p))
        assert _probe_dtype(p) == "int16"

    def test_probe_uint8(self, tmp_path):
        """RTL-SDR style uint8 (offset 127.5) correctly detected as uint8."""
        rng = np.random.default_rng(3)
        data = (rng.uniform(-0.8, 0.8, N * 2) * 100 + 127.5).astype(np.uint8)
        p = tmp_path / "auto_u8.iq"
        data.tofile(str(p))
        assert _probe_dtype(p) == "uint8"

    def test_probe_float32(self, tmp_path):
        """Normalised float32 IQ bytes have p75 of |int16| << 5000 — not int16."""
        rng = np.random.default_rng(4)
        data = rng.uniform(-1, 1, N * 2).astype(np.float32)
        p = tmp_path / "auto_f32.iq"
        data.tofile(str(p))
        result = _probe_dtype(p)
        assert result != "int16"

    def test_probe_complex64(self, tmp_path):
        """Small-magnitude complex64 file detected as complex64."""
        rng = np.random.default_rng(5)
        data = (rng.standard_normal(N) + 1j * rng.standard_normal(N)).astype(np.complex64) * 0.3
        p = tmp_path / "auto_c64.iq"
        data.tofile(str(p))
        assert _probe_dtype(p) == "complex64"

    def test_auto_load_int16(self, tmp_path):
        """load_iq with dtype='auto' correctly loads a large-amplitude int16 file."""
        data = np.array([20000, -20000] * (N // 2), dtype=np.int16)
        p = tmp_path / "a16.iq"
        data.tofile(str(p))
        iq, meta = load_iq(str(p), dtype="auto", fs=FS)
        assert meta.dtype == "int16"
        assert np.max(np.abs(iq.real)) <= 1.01
        assert np.max(np.abs(iq.imag)) <= 1.01


# ---------------------------------------------------------------------------
# Error guards
# ---------------------------------------------------------------------------

class TestErrorGuards:
    def test_odd_sample_count_raises(self, tmp_path):
        """Odd sample count in a float32 file raises ValueError."""
        p = tmp_path / "odd.iq"
        data = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        data.tofile(str(p))
        with pytest.raises(ValueError, match="odd number"):
            load_iq(str(p), dtype="float32", fs=FS)

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_iq(str(tmp_path / "no_such_file.iq"), fs=FS)

    def test_bad_dtype_raises(self, tmp_path):
        """Unknown dtype string raises TypeError."""
        p = tmp_path / "dummy.iq"
        data = np.zeros(4, dtype=np.float32)
        data.tofile(str(p))
        with pytest.raises(TypeError, match="Unsupported dtype"):
            load_iq(str(p), dtype="float128", fs=FS)  # type: ignore[arg-type]

    def test_missing_fs_warns(self, tmp_path):
        """Omitting fs emits a UserWarning."""
        p = tmp_path / "nofs.iq"
        data = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        data.tofile(str(p))
        with pytest.warns(UserWarning, match="No sample rate"):
            load_iq(str(p), dtype="float32")


# ---------------------------------------------------------------------------
# Metadata population
# ---------------------------------------------------------------------------

class TestMetadataPopulation:
    def test_duration_computed_correctly(self, tmp_path):
        data = np.zeros(N * 2, dtype=np.float32)
        p = tmp_path / "dur.iq"
        data.tofile(str(p))
        _, meta = load_iq(str(p), dtype="float32", fs=FS)
        assert abs(meta.duration_sec - N / FS) < 1e-9

    def test_num_samples(self, tmp_path):
        data = np.zeros(N * 2, dtype=np.float32)
        p = tmp_path / "nsamp.iq"
        data.tofile(str(p))
        _, meta = load_iq(str(p), dtype="float32", fs=FS)
        assert meta.num_samples == N

    def test_center_freq_hint(self, tmp_path):
        data = np.zeros(N * 2, dtype=np.float32)
        p = tmp_path / "cf.iq"
        data.tofile(str(p))
        _, meta = load_iq(str(p), dtype="float32", fs=FS, center_freq_hint=433e6)
        assert meta.center_freq_hint == 433e6


# ---------------------------------------------------------------------------
# SigMF companion loading
# ---------------------------------------------------------------------------

class TestSigMFCompanion:
    def _write_sigmf_meta(self, path: Path, fs: float, freq: float, dtype_tok: str):
        meta = {"global": {
            "core:sample_rate": fs,
            "core:frequency": freq,
            "core:datatype": dtype_tok,
        }}
        path.write_text(json.dumps(meta))

    def test_load_sigmf_meta(self, tmp_path):
        meta_path = tmp_path / "test.sigmf-meta"
        self._write_sigmf_meta(meta_path, 2e6, 915e6, "ci16_le")
        result = load_sigmf_meta(str(meta_path))
        assert result["fs"] == 2e6
        assert result["center_freq"] == 915e6
        assert result["dtype"] == "ci16_le"

    def test_load_iq_with_sigmf_companion_full_name(self, tmp_path):
        """Auto-discovers <name>.iq.sigmf-meta companion (full-name pattern)."""
        iq_path = tmp_path / "capture.iq"
        # Companion named: capture.iq.sigmf-meta
        meta_path = tmp_path / "capture.iq.sigmf-meta"

        data = np.array([10000, -10000] * (N // 2), dtype=np.int16)
        data.tofile(str(iq_path))
        self._write_sigmf_meta(meta_path, 2e6, 433e6, "ci16_le")

        iq, meta = load_iq_with_sigmf(str(iq_path))
        assert meta.fs == 2e6
        assert meta.center_freq_hint == 433e6
        assert meta.dtype == "int16"
        assert iq.dtype == np.complex64
        assert np.max(np.abs(iq.real)) <= 1.01
        assert np.max(np.abs(iq.imag)) <= 1.01

    def test_load_iq_with_sigmf_companion_stem_name(self, tmp_path):
        """Auto-discovers <stem>.sigmf-meta companion (stem-name pattern)."""
        iq_path = tmp_path / "capture.iq"
        # Companion named: capture.sigmf-meta
        meta_path = tmp_path / "capture.sigmf-meta"

        data = np.array([10000, -10000] * (N // 2), dtype=np.int16)
        data.tofile(str(iq_path))
        self._write_sigmf_meta(meta_path, 4e6, 915e6, "ci16_le")

        iq, meta = load_iq_with_sigmf(str(iq_path))
        assert meta.fs == 4e6
        assert meta.center_freq_hint == 915e6

    def test_load_iq_with_explicit_meta_path(self, tmp_path):
        """Explicit meta_path overrides auto-discovery."""
        iq_path = tmp_path / "data.iq"
        meta_path = tmp_path / "custom_meta.sigmf-meta"

        data = np.array([5000, -5000] * (N // 2), dtype=np.int16)
        data.tofile(str(iq_path))
        self._write_sigmf_meta(meta_path, 8e6, 144e6, "ci16_le")

        iq, meta = load_iq_with_sigmf(str(iq_path), meta_path=str(meta_path))
        assert meta.fs == 8e6
        assert meta.center_freq_hint == 144e6


# ---------------------------------------------------------------------------
# WAV loading (SOP 1.x — real-audio ingestion path)
# ---------------------------------------------------------------------------

class TestWavLoading:
    def _write_wav(self, path: Path, rate: int, dtype, data: np.ndarray):
        from scipy.io import wavfile
        wavfile.write(str(path), rate, data.astype(dtype))

    def test_mono_float32(self, tmp_path):
        rng = np.random.default_rng(11)
        sig = rng.uniform(-0.8, 0.8, 1000).astype(np.float32)
        p = tmp_path / "mono.wav"
        self._write_wav(p, 8000, np.float32, sig)

        sig_out, meta = load_wav(str(p))
        assert meta.fs == 8000.0
        assert meta.source_format == "wav"
        assert meta.num_samples == 1000
        np.testing.assert_allclose(sig_out, sig, atol=1e-5)

    def test_stereo_float64_downmixed(self, tmp_path):
        """Multi-channel WAV is averaged to mono."""
        rng = np.random.default_rng(22)
        stereo = rng.uniform(-1, 1, (500, 2)).astype(np.float64)
        p = tmp_path / "stereo.wav"
        self._write_wav(p, 16000, np.float64, stereo)

        sig_out, meta = load_wav(str(p))
        assert sig_out.ndim == 1
        assert len(sig_out) == 500
        np.testing.assert_allclose(sig_out, stereo.mean(axis=1), atol=1e-6)

    def test_int16_normalisation(self, tmp_path):
        """Integer WAV samples are scaled to [-1, 1] using the dtype max."""
        rng = np.random.default_rng(33)
        raw = (rng.uniform(-1, 1, 400) * 32767).astype(np.int16)
        p = tmp_path / "int16.wav"
        self._write_wav(p, 8000, np.int16, raw)

        sig_out, meta = load_wav(str(p))
        assert meta.dtype == "int16"
        # Max |int16| maps to ≈1.0
        expected = raw.astype(np.float32) / 32767.0
        np.testing.assert_allclose(sig_out, expected, atol=1e-3)

    def test_center_freq_hint(self, tmp_path):
        rng = np.random.default_rng(44)
        p = tmp_path / "cf.wav"
        self._write_wav(p, 8000, np.float32, rng.uniform(-1, 1, 100))

        _, meta = load_wav(str(p), center_freq_hint=433e6)
        assert meta.center_freq_hint == 433e6


# ---------------------------------------------------------------------------
# segment_signal (regression)
# ---------------------------------------------------------------------------

class TestSegmentSignal:
    def test_basic_segmentation(self):
        sig = np.arange(100, dtype=np.float32)
        segs = segment_signal(sig, segment_len=10, overlap=0)
        assert len(segs) == 10
        np.testing.assert_array_equal(segs[0], np.arange(10, dtype=np.float32))

    def test_overlapping_segmentation(self):
        sig = np.arange(20, dtype=np.float32)
        segs = segment_signal(sig, segment_len=10, overlap=5)
        assert len(segs) == 3
