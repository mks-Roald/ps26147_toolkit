"""
tests/test_cli.py
Phase 8 – Unit tests for the command-line entry point & batch report generation.

Covers:
  - process_file() on a real synthetic .iq file (full extraction pipeline)
  - process_file() on a .wav file
  - error handling: missing file, unsupported extension
  - main() argument-driven single-file JSON output
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ps26147_toolkit.cli import process_file, main
from ps26147_toolkit.preprocess import load_iq


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_synthetic_iq(path: Path, fs: float = 1_000_000.0) -> None:
    """Write a 100 kHz BPSK carrier (int16 IQ) to disk for pipeline testing."""
    rng = np.random.default_rng(7)
    baud = 25_000.0
    sps = int(fs / baud)
    n_sym = 400
    syms = rng.integers(0, 2, n_sym)
    baseband = np.repeat(np.where(syms == 1, 1.0, -1.0), sps)
    t = np.arange(len(baseband)) / fs
    sig = baseband * np.exp(1j * 2 * np.pi * 100_000 * t)
    sig += (rng.normal(0, 0.02, len(sig)) + 1j * rng.normal(0, 0.02, len(sig)))
    # int16 normalized
    i16 = np.empty(2 * len(sig), dtype=np.int16)
    i16[0::2] = (np.clip(sig.real, -1, 1) * 32767).astype(np.int16)
    i16[1::2] = (np.clip(sig.imag, -1, 1) * 32767).astype(np.int16)
    i16.tofile(str(path))


def _write_synth_wav(path: Path, fs: int = 8000) -> None:
    """Write a real-signal .wav containing a 1 kHz tone."""
    from scipy.io import wavfile
    rng = np.random.default_rng(11)
    t = np.arange(2000) / fs
    sig = 0.8 * np.cos(2 * np.pi * 1000 * t) + 0.05 * rng.standard_normal(2000)
    wavfile.write(str(path), fs, sig.astype(np.float32))


# ---------------------------------------------------------------------------
# process_file – .iq path
# ---------------------------------------------------------------------------

class TestProcessFile:
    def test_iq_full_pipeline(self, tmp_path):
        p = tmp_path / "sig.iq"
        _write_synthetic_iq(p)

        report = process_file(str(p), fs_iq=1_000_000.0)

        assert report["file"] == str(p)
        assert "modulation" in report
        assert "center_frequency_hz" in report
        assert "bandwidth_hz" in report
        assert "snr_db" in report
        assert "baud_rate" in report
        # Carrier should land near 100 kHz
        assert abs(report["center_frequency_hz"] - 100_000) < 8_000
        # SNR should be positive on this clean signal
        assert report["snr_db"] > 0.0

    def test_iq_with_filter_flags(self, tmp_path):
        p = tmp_path / "sig_f.iq"
        _write_synthetic_iq(p)
        report = process_file(str(p), fs_iq=1_000_000.0,
                              filter_noise=True, denoise=True)
        assert report["bandwidth_hz"] > 0

    def test_wav_path(self, tmp_path):
        p = tmp_path / "tone.wav"
        _write_synth_wav(p)
        report = process_file(str(p))
        assert "modulation" in report
        assert report["snr_db"] > -5.0

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            process_file(str(tmp_path / "nope.iq"))

    def test_unsupported_extension_raises(self, tmp_path):
        p = tmp_path / "data.txt"
        p.write_text("hello")
        with pytest.raises(ValueError, match="Unsupported file type"):
            process_file(str(p))


# ---------------------------------------------------------------------------
# main() – argparse-driven single file & batch CSV output
# ---------------------------------------------------------------------------

class TestMain:
    def test_single_file_json(self, tmp_path, capsys, monkeypatch, caplog):
        p = tmp_path / "sig.iq"
        _write_synthetic_iq(p)
        out = tmp_path / "out.json"

        monkeypatch.setattr("sys.argv", ["cli", str(p), "--fs", "1000000", "--output", str(out)])
        main()

        assert out.exists()
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["file"] == str(p)
        assert "modulation" in data

    def test_batch_csv_output(self, tmp_path, monkeypatch):
        p1 = tmp_path / "a.iq"
        p2 = tmp_path / "b.iq"
        _write_synthetic_iq(p1)
        _write_synthetic_iq(p2)
        csv_path = tmp_path / "summary.csv"

        monkeypatch.setattr("sys.argv", ["cli", str(tmp_path), "--csv", str(csv_path)])
        main()

        assert csv_path.exists()
        text = csv_path.read_text(encoding="utf-8")
        # Header + two rows
        assert text.count("\n") == 3

    def test_directory_glob_and_output_json(self, tmp_path, monkeypatch):
        """Directory input walks *.iq / *.wav recursively; json output reflects last file."""
        sub = tmp_path / "nested"
        sub.mkdir()
        _write_synthetic_iq(sub / "one.iq")
        _write_synth_wav(sub / "two.wav")
        out = tmp_path / "last.json"

        monkeypatch.setattr("sys.argv", ["cli", str(tmp_path), "--output", str(out)])
        main()

        assert out.exists()
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["file"].endswith(("one.iq", "two.wav"))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])