#!/usr/bin/env python3
"""Ground-truth synthetic signal corpus generator for PS26147.

Generates `.wav` / `.iq` files whose content is *known a priori* — every file
is paired with a ground-truth JSON giving the exact values a correct pipeline
should report (modulation, center frequency, bandwidth, injected SNR, baud
rate, and, where applicable, the FEC/interleave/sync parameters and payload).

This is the foundation of Phase 6 ("Accuracy Hardening"): without files of
known content, no accuracy claim is falsifiable.

Focus for the first tier: **modulation axis in isolation** — clean, no FEC,
no interleaving, no sync, so the demodulator + parameter extractor can be
validated per modulation.  The payload is always a human-readable ASCII string
so a decode failure is visually obvious.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.io.wavfile as wav

# ---------------------------------------------------------------------------
# Modulation symbol mappers  (bits -> complex baseband symbols)
# ---------------------------------------------------------------------------

_PANGRAM = "the quick brown fox jumps over the lazy dog 0123456789"


def text_to_bits(text: str) -> np.ndarray:
    """UTF-8 bytes -> uint8 bit array (MSB-first)."""
    raw = text.encode("utf-8")
    return np.unpackbits(np.frombuffer(raw, dtype=np.uint8))


def _canonical_modulation(modulation: str) -> str:
    """Resolve a modulation name to a canonical token.

    Uses exact token identity (after stripping spaces/hyphens and uppercasing)
    so that ``64QAM`` is never shadowed by the substring ``4QAM`` inside an
    ``in`` check (a real bug: ``"QPSK" in m or "4QAM" in m`` matched
    ``"64QAM"`` and silently produced QPSK corners).  Aliases like ``QPSK`` /
    ``4QAM`` / ``4-QAM`` and ``2FSK`` / ``2-FSK`` collapse to one canonical
    form.
    """
    m = modulation.upper().replace(" ", "").replace("-", "")
    for token in ("64QAM", "16QAM", "4QAM", "8PSK", "QPSK", "BPSK", "4FSK", "2FSK"):
        if m == token:
            return token
    # Bare "FSK" and "QAM" (no order) default to the binary forms
    if m == "FSK":
        return "2FSK"
    if m == "QAM":
        return "4QAM"
    return m


def bits_to_symbols(bits: np.ndarray, modulation: str) -> np.ndarray:
    """Map an info-bit array to complex baseband symbols (one per symbol)."""
    m = _canonical_modulation(modulation)

    bps = {"BPSK": 1, "QPSK": 2, "8PSK": 3, "16QAM": 4, "64QAM": 6, "2FSK": 1, "4FSK": 2}.get(m)
    if bps is None:
        raise ValueError(f"Unsupported modulation: {modulation}")

    usable = (len(bits) // bps) * bps
    bits = bits[:usable].reshape(-1, bps)

    symbols: list[complex] = []
    if m == "BPSK":
        for row in bits:
            symbols.append(1.0 if row[0] else -1.0)
    elif m == "QPSK":
        for b0, b1 in bits:
            re = (1.0 if b0 == 0 else -1.0) / np.sqrt(2)
            im = (1.0 if b1 == 0 else -1.0) / np.sqrt(2)
            symbols.append(re + 1j * im)
    elif m == "8PSK":
        # Gray map matching demodulator.slice_symbols_to_bits
        gray_map = {tuple(v): k for k, v in {
            0: [0, 0, 0], 1: [0, 0, 1], 2: [0, 1, 1], 3: [0, 1, 0],
            4: [1, 1, 0], 5: [1, 1, 1], 6: [1, 0, 1], 7: [1, 0, 0],
        }.items()}
        for row in bits:
            sec = gray_map[tuple(int(b) for b in row)]
            symbols.append(np.exp(1j * sec * (np.pi / 4)))
    elif m == "64QAM":
        levels = np.arange(-7, 8, 2) / np.sqrt(42)
        for row in bits:
            idx_i = (row[0] << 2) | (row[1] << 1) | row[2]
            idx_q = (row[3] << 2) | (row[4] << 1) | row[5]
            symbols.append(levels[idx_i] + 1j * levels[idx_q])
    elif m == "16QAM":
        levels = np.array([-3, -1, 1, 3]) / np.sqrt(10)
        # Gray-coded I/Q levels (matches demodulator level_bits)
        level_idx = {(0, 0): 0, (0, 1): 1, (1, 1): 2, (1, 0): 3}
        for row in bits:
            idx_i = level_idx[tuple(int(b) for b in row[:2])]
            idx_q = level_idx[tuple(int(b) for b in row[2:4])]
            symbols.append(levels[idx_i] + 1j * levels[idx_q])
    elif m == "4FSK":
        # Signs for 4-level frequency deviation
        for row in bits:
            code = (row[0] << 1) | row[1]
            dev = {0: -0.75, 1: -0.25, 2: 0.25, 3: 0.75}[code]
            symbols.append(complex(dev, 0.0))
    else:  # 2FSK
        for row in bits:
            symbols.append(1.0 if row[0] else -1.0)

    return np.array(symbols, dtype=np.complex64)


def symbols_to_passband(
    symbols: np.ndarray,
    fs: float,
    center_freq: float,
    baud: float,
) -> np.ndarray:
    """Root-raised-cosine (RRC) pulse-shape baseband symbols and upconvert to a
    passband real waveform around *center_freq*.

    The ground-truth theory labels linear modulations with the RRC-0.35
    occupied bandwidth ``(1 + 0.35) * baud``, so the *generated signal must use
    the same shaping* or the reference and the measured spectrum disagree.
    RRC pulse shaping concentrates the spectrum in a sharp-sided lobe with the
    true occupied width ``(1 + alpha) * baud``; the bandwidth-calibration
    harness (Phase 6 §1.7) reads this edge via a -40 dB contour.  The residual
    inter-symbol interference from a TX-only RRC pulse is *not* garbling —
    the current demodulator recovers clean bits on RRC-shaped signals
    (verified 1.000 match), because the SPS stays well below the Nyquist limit.

    Returns a real-valued array suitable for a `.wav` file.
    """
    from ps26147_toolkit.feature_extractor import rrc_filter

    sps = max(8, int(round(fs / baud)))
    n_sym = len(symbols)
    # Upsample by a single impulse per symbol, then shape with RRC.
    # num_taps ~ 4x SPS (odd) gives ~2 symbols of rolloff on either side so the
    # spectrum reaches its true (1+alpha)*baud edge cleanly.
    num_taps = max(49, 4 * sps + 1)
    if num_taps % 2 == 0:
        num_taps += 1
    rrc = rrc_filter(num_taps=num_taps, alpha=0.35, sps=sps)
    # Place one complex symbol impulse per symbol period, then shape with RRC.
    up = np.zeros(n_sym * sps, dtype=np.complex64)
    up[::sps] = symbols
    bb = np.convolve(up, rrc, mode="same").astype(np.complex64)

    # Upconvert to passband
    t = np.arange(len(bb)) / fs
    sig = (bb * np.exp(1j * 2 * np.pi * center_freq * t)).real
    # Normalise to [-0.9, 0.9] to avoid clipping when written as int16
    peak = np.max(np.abs(sig))
    if peak > 0:
        sig = sig * (0.9 / peak)
    return sig.astype(np.float32)


def symbols_to_passband_fsk(
    dev_bits: np.ndarray,
    fs: float,
    center_freq: float,
    baud: float,
    deviation: float,
) -> np.ndarray:
    """Direct-FSK passband modulator.

    *dev_bits* is an array of signed frequency deviations (in Hz) per symbol.
    The instantaneous frequency is ``center_freq + dev``, so the spectrum is
    centred at *center_freq* with the occupied bandwidth set by *deviation*.
    """
    sps = max(8, int(round(fs / baud)))
    n_sym = len(dev_bits)
    n = n_sym * sps
    t = np.arange(n) / fs
    # Hold each symbol's frequency for a full symbol period
    freq = np.repeat(center_freq + dev_bits, sps)
    phase = 2 * np.pi * np.cumsum(freq) / fs
    sig = np.sin(phase)
    peak = np.max(np.abs(sig))
    if peak > 0:
        sig = sig * (0.9 / peak)
    return sig.astype(np.float32)


# ---------------------------------------------------------------------------
# Noise injection  (known SNR)
# ---------------------------------------------------------------------------

def add_awgn(sig: np.ndarray, snr_db: Optional[float]) -> np.ndarray:
    """Inject AWGN to achieve a target SNR (dB), or return the signal unchanged
    when *snr_db* is None (i.e. "clean")."""
    if snr_db is None:
        return sig
    sig_pow = np.mean(sig ** 2)
    if sig_pow <= 0:
        return sig
    noise_pow = sig_pow / (10 ** (snr_db / 10.0))
    rng = np.random.default_rng(1234)
    noise = rng.standard_normal(sig.shape).astype(np.float32)
    noise = noise * np.sqrt(noise_pow)
    return (sig + noise).astype(np.float32)


# ---------------------------------------------------------------------------
# File writers
# ---------------------------------------------------------------------------

def write_wav(sig: np.ndarray, fs: float, path: Path) -> None:
    """Write a real signal as 16-bit PCM mono WAV (0dbfs ≈ full scale)."""
    pcm = np.round(sig * 32767.0).astype(np.int16)
    wav.write(str(path), int(fs), pcm)


def write_iq(sig: np.ndarray, fs: float, center_freq: float, path: Path) -> None:
    """Write a real passband signal as interleaved complex64 I/Q.

    The analytic (Hilbert) signal is written so downstream `load_iq` recovers
    the full passband spectrum centred at *center_freq*.
    """
    from scipy.signal import hilbert
    analytic = hilbert(sig).astype(np.complex64)
    interleaved = np.empty(analytic.size * 2, dtype=np.float32)
    interleaved[0::2] = analytic.real
    interleaved[1::2] = analytic.imag
    path.write_bytes(interleaved.tobytes())


# ---------------------------------------------------------------------------
# Main generation API
# ---------------------------------------------------------------------------

def _theoretical_bandwidth(modulation: str, baud: float, deviation: float) -> float:
    """Return a theoretical occupied bandwidth for ground-truth labelling."""
    m = modulation.upper()
    if "FSK" in m:
        # Carson's rule: 2*(dev / 2) * ... -> BW ≈ 2*dev + baud
        return float(2.0 * deviation + baud)
    elif "QPSK" in m or "8PSK" in m:
        # Approx null-to-null bandwidth for linear modulations with RRC 0.35
        return float((1 + 0.35) * baud)
    else:
        return float((1 + 0.35) * baud)


def generate_test_case(
    modulation: str,
    payload_text: str = _PANGRAM,
    snr_db: Optional[float] = None,
    fs: float = 48000,
    center_freq: float = 10_000,
    baud: float = 1200,
    fsk_deviation: float = 498.0,
    file_format: str = "wav",
    out_dir: str = "ground_truth",
) -> tuple[Path, dict]:
    """Produce a single ground-truth test file.

    .. note:: The default ``fs`` (48000 Hz) and ``baud`` (1200) give an integer
       samples-per-symbol (40), keeping the rectangular-shaped signal free of
       fractional-timing drift across long payloads.
    """
    """Produce a single ground-truth test file.

    Returns
    -------
    (file_path, ground_truth_dict)
        The generated signal file and its ground-truth metadata.  The ground
        truth is also written to ``<file_path>.json``.
    """
    bits = text_to_bits(payload_text)

    m = modulation.upper()
    if "FSK" in m:
        # 2FSK/4FSK: build a signed frequency-deviation bitstream
        if "4FSK" in m:
            symbols = bits_to_symbols(bits, modulation)
            dev_bits = symbols.real * fsk_deviation
            baud_effective = baud
        else:  # 2FSK
            dev_bits = np.where(bits == 1, 1.0, -1.0) * fsk_deviation
            baud_effective = baud
        sig = symbols_to_passband_fsk(
            dev_bits.astype(np.float32), fs, center_freq, baud, deviation=fsk_deviation
        )
        n_symbols = len(dev_bits)
    else:
        symbols = bits_to_symbols(bits, modulation)
        sig = symbols_to_passband(symbols, fs, center_freq, baud)
        n_symbols = len(symbols)
        baud_effective = baud

    sig = add_awgn(sig, snr_db)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{modulation.lower().replace(' ', '')}"
    stem += "_clean" if snr_db is None else f"_snr{str(snr_db).replace('.', 'p')}db"
    stem += f"_{file_format}"
    file_path = out / f"{stem}.{file_format}"
    if file_format == "wav":
        write_wav(sig, fs, file_path)
    else:
        write_iq(sig, fs, center_freq, file_path)

    ground_truth: dict = {
        "file": file_path.name,
        "modulation": modulation,
        "center_freq_hz": float(center_freq),
        "baud_rate": float(baud_effective),
        "bandwidth_hz_theoretical": _theoretical_bandwidth(modulation, baud, fsk_deviation),
        "fsk_deviation_hz": float(fsk_deviation) if "FSK" in m else None,
        "snr_db_injected": snr_db if snr_db is not None else "clean",
        "sample_rate": float(fs),
        "num_symbols": int(n_symbols),
        "num_samples": int(len(sig)),
        "file_format": file_format,
        "payload_text": payload_text,
        "payload_bits": [int(b) for b in bits],
        "fec_scheme": "none",
        "interleave_method": "none",
        "sync_word": None,
    }
    json_path = file_path.with_suffix(file_path.suffix + ".json")
    json_path.write_text(json.dumps(ground_truth, indent=2))
    return file_path, ground_truth


# ---------------------------------------------------------------------------
# CLI: build the minimum viable corpus (modulation axis, tier 1)
# ---------------------------------------------------------------------------

_MIN_MODULATIONS = ["BPSK", "QPSK", "8PSK", "16QAM", "64QAM", "2FSK", "4FSK"]


def build_minimum_corpus(out_dir: str = "ground_truth", verbose: bool = True) -> list[Path]:
    """Generate tier-1 minimum viable corpus: 7 modulations x clean, both formats."""
    paths: list[Path] = []
    for mod in _MIN_MODULATIONS:
        for fmt in ("wav", "iq"):
            p, gt = generate_test_case(mod, file_format=fmt, out_dir=out_dir)
            paths.append(p)
            if verbose:
                print(f"  wrote {p.name} ({gt['num_samples']} samples)")
    return paths


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate ground-truth corpus")
    parser.add_argument("--out", default="ground_truth", help="Output directory")
    parser.add_argument("--corpus", action="store_true", help="Build tier-1 minimum corpus")
    parser.add_argument("--mod", default="QPSK", help="Single modulation (when not --corpus)")
    args = parser.parse_args()

    if args.corpus:
        build_minimum_corpus(args.out)
    else:
        generate_test_case(args.mod, out_dir=args.out)
        p, gt = generate_test_case(args.mod, file_format="iq", out_dir=args.out)
        print(f"Wrote {p} ground-truth: modulation={gt['modulation']} fc={gt['center_freq_hz']}Hz")