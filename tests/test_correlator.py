"""Unit tests for Bitstream Cross-Correlation, Preamble Detection, and Frame Synchronization."""
import numpy as np
from ps26147_toolkit.correlator import (
    STANDARD_SYNC_WORDS,
    hex_to_bits,
    correlate_bitstream,
    find_sync_peaks,
    frame_synchronize,
    auto_discover_preamble,
)

def test_hex_conversion():
    bits = hex_to_bits("1ACFFC1D")
    expected = STANDARD_SYNC_WORDS["CCSDS-32"]
    assert np.array_equal(bits, expected), "hex_to_bits mismatch"
    print("[PASS] Hex to Bitstream Conversion")

def test_barker_correlation():
    barker13 = STANDARD_SYNC_WORDS["Barker-13"]
    rng = np.random.default_rng(42)
    noise = rng.integers(0, 2, size=100, dtype=np.uint8)

    # Insert Barker-13 at index 35
    stream = np.copy(noise)
    stream[35 : 35 + len(barker13)] = barker13

    corr, inverted = correlate_bitstream(stream, barker13)
    peak_idx = int(np.argmax(corr))
    assert peak_idx == 35, f"Expected peak at 35, got {peak_idx}"
    assert corr[peak_idx] == 1.0, f"Expected perfect correlation 1.0, got {corr[peak_idx]}"
    assert not inverted, "Detected false inversion"
    print("[PASS] Bipolar Barker-13 Cross-Correlation (Exact match at offset 35)")

def test_inverted_sync_correlation():
    ccsds = STANDARD_SYNC_WORDS["CCSDS-32"]
    rng = np.random.default_rng(99)
    noise = rng.integers(0, 2, size=120, dtype=np.uint8)

    # Insert 180° inverted sync marker at index 40
    stream = np.copy(noise)
    stream[40 : 40 + len(ccsds)] = 1 - ccsds

    corr, is_inverted = correlate_bitstream(stream, ccsds, tolerate_inverted=True)
    min_idx = int(np.argmin(corr))
    assert min_idx == 40, f"Expected inverted peak at 40, got {min_idx}"
    assert is_inverted, "Failed to identify 180° carrier phase inversion"
    print("[PASS] 180° Inverted Sync Word Detection & Auto-Uninversion")

def test_frame_synchronization():
    sync_word = STANDARD_SYNC_WORDS["Barker-11"]
    L_sync = len(sync_word)
    frame_len = 64
    num_frames = 5

    rng = np.random.default_rng(123)
    full_stream = []

    for f in range(num_frames):
        payload = rng.integers(0, 2, size=frame_len - L_sync, dtype=np.uint8)
        frame = np.concatenate([sync_word, payload])
        full_stream.extend(frame)

    full_stream = np.array(full_stream, dtype=np.uint8)
    # Add random preamble noise before first frame
    noise_prefix = rng.integers(0, 2, size=20, dtype=np.uint8)
    rx_stream = np.concatenate([noise_prefix, full_stream])

    res = frame_synchronize(rx_stream, sync_word, frame_length=frame_len, threshold=0.90)
    assert res["sync_found"], "Frame sync failed to find markers"
    assert res["num_frames"] == num_frames, f"Expected {num_frames} frames, found {res['num_frames']}"
    assert res["peak_indices"][0] == 20, f"Expected first frame at index 20, got {res['peak_indices'][0]}"
    print(f"[PASS] Frame Synchronization ({num_frames} frames locked and sliced at interval {frame_len})")

def test_auto_preamble_discovery():
    sync_word = STANDARD_SYNC_WORDS["DVB-S (0x47)"]
    frame_len = 188 * 8  # 188 byte MPEG frame in bits
    num_frames = 4

    rng = np.random.default_rng(555)
    stream = []
    for _ in range(num_frames):
        payload = rng.integers(0, 2, size=frame_len - len(sync_word), dtype=np.uint8)
        frame = np.concatenate([sync_word, payload])
        stream.extend(frame)

    rx_stream = np.array(stream, dtype=np.uint8)
    res = auto_discover_preamble(rx_stream)
    assert res["discovered"], "Preamble discovery failed"
    assert res["matched_standard_sync"] == "DVB-S (0x47)", f"Expected DVB-S match, got {res['matched_standard_sync']}"
    print(f"[PASS] Auto-Preamble Discovery (Detected {res['matched_standard_sync']} with confidence {res['standard_sync_confidence']:.2f})")

def test_auto_preamble_short_stream_returns_full_key_shape():
    """auto_discover_preamble on a short bitstream must return every key the
    normal path returns — not just 'discovered' + 'reason' — so callers using
    .get() don't hit a shape mismatch."""
    short_stream = np.zeros(4, dtype=np.uint8)
    res = auto_discover_preamble(short_stream)
    # Must carry the full key set
    for key in ("discovered", "estimated_frame_period", "periodicity_strength",
                "matched_standard_sync", "standard_sync_confidence",
                "candidate_preamble_bits", "candidate_preamble_hex"):
        assert key in res, f"Missing key '{key}' in early-return dict"
    # Sanity values
    assert res["discovered"] is False
    assert res["estimated_frame_period"] is None
    assert res["matched_standard_sync"] is None
    assert res["candidate_preamble_bits"] is None
    print("[PASS] Early-return dict carries full key shape")

if __name__ == "__main__":
    test_hex_conversion()
    test_barker_correlation()
    test_inverted_sync_correlation()
    test_frame_synchronization()
    test_auto_preamble_discovery()
    test_auto_preamble_short_stream_returns_full_key_shape()
    print("\n All Bitstream Correlation & Frame Synchronization Tests PASSED!")
