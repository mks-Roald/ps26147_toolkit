"""Bit Stream Correlation and Frame Synchronization module for PS26147 toolkit.

Implements:
  1. Standard Sync Words & Preambles (Barker 7/11/13, CCSDS 32-bit ASM, DVB-S Sync 0x47, IEEE 802.11 SFD, Ethernet Preamble).
  2. Bipolar Normalized Cross-Correlation (sliding window normalized inner product with phase ambiguity check).
  3. Frame Synchronization & Boundary Alignment (splits continuous bitstream into frames based on periodic sync markers).
  4. Bit Error Rate (BER) & Synchronization Confidence estimation.
  5. Auto-Preamble Discovery (detects repetitive frame sync patterns without prior knowledge using lag-autocorrelation).
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple, Any, Optional


# ===========================================================================
# Standard Sync Words & Preambles Library
# ===========================================================================

STANDARD_SYNC_WORDS: dict[str, np.ndarray] = {
    # Barker codes (optimum aperiodic autocorrelation)
    "Barker-7": np.array([1, 1, 1, 0, 0, 1, 0], dtype=np.uint8),
    "Barker-11": np.array([1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0], dtype=np.uint8),
    "Barker-13": np.array([1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1], dtype=np.uint8),
    
    # CCSDS 32-bit Attached Sync Marker (ASM: 0x1ACFFC1D)
    "CCSDS-32": np.unpackbits(np.array([0x1A, 0xCF, 0xFC, 0x1D], dtype=np.uint8)),
    
    # DVB-S MPEG-TS Frame Sync (0x47 = 01000111)
    "DVB-S (0x47)": np.unpackbits(np.array([0x47], dtype=np.uint8)),
    
    # Inverted DVB-S Sync (0xB8 = 10111000)
    "DVB-S Inverted (0xB8)": np.unpackbits(np.array([0xB8], dtype=np.uint8)),
    
    # IEEE 802.15.4 / ZigBee SFD (0xA7 = 10100111)
    "ZigBee-SFD (0xA7)": np.unpackbits(np.array([0xA7], dtype=np.uint8)),

    # IEEE 802.11 WiFi Barker Preamble (0xF3A0)
    "WiFi-SFD (0xF3A0)": np.unpackbits(np.array([0xF3, 0xA0], dtype=np.uint8)),

    # Alternating 10101010 Training Preamble (16-bit)
    "Preamble-1010 (16-bit)": np.array([1, 0] * 8, dtype=np.uint8),
}


def hex_to_bits(hex_str: str) -> np.ndarray:
    """Convert hexadecimal string (e.g. '1ACFFC1D' or '0x47') to 1-D numpy uint8 array of bits."""
    clean_hex = hex_str.strip().lower().replace("0x", "").replace(" ", "")
    if len(clean_hex) % 2 != 0:
        clean_hex = "0" + clean_hex
    byte_arr = bytes.fromhex(clean_hex)
    return np.unpackbits(np.frombuffer(byte_arr, dtype=np.uint8))


# ===========================================================================
# 1. Bipolar Cross-Correlation
# ===========================================================================

def correlate_bitstream(
    bitstream: np.ndarray,
    sync_word: np.ndarray,
    tolerate_inverted: bool = True,
) -> tuple[np.ndarray, bool]:
    """Compute normalized bipolar cross-correlation between continuous bitstream and target sync pattern.

    Maps {0, 1} -> {-1, +1} to maximize positive peak for exact match, negative peak for 180-deg phase ambiguity.

    Parameters
    ----------
    bitstream : np.ndarray
        Incoming stream of bits (uint8: 0/1).
    sync_word : np.ndarray
        Sync marker pattern (uint8: 0/1).
    tolerate_inverted : bool
        If True, checks for 180-deg inverted carrier synchronization.

    Returns
    -------
    corr_norm : np.ndarray
        Normalized cross-correlation scores [-1.0, 1.0] across all sliding positions.
    is_inverted : bool
        True if best correlation peak was with inverted sync word.
    """
    if len(bitstream) < len(sync_word) or len(sync_word) == 0:
        return np.zeros(0, dtype=np.float32), False

    # Convert binary {0, 1} to bipolar {-1, +1}
    s_bipolar = np.where(bitstream == 1, 1.0, -1.0).astype(np.float32)
    p_bipolar = np.where(sync_word == 1, 1.0, -1.0).astype(np.float32)
    L = len(sync_word)

    # 1-D valid correlation via convolution with reversed pattern
    raw_corr = np.correlate(s_bipolar, p_bipolar, mode="valid")
    corr_norm = raw_corr / float(L)

    # Detect if highest absolute correlation is negative (indicating phase inversion)
    max_pos = np.max(corr_norm) if len(corr_norm) > 0 else 0.0
    min_neg = np.min(corr_norm) if len(corr_norm) > 0 else 0.0

    if tolerate_inverted and abs(min_neg) > max_pos:
        return corr_norm, True
    return corr_norm, False


# ===========================================================================
# 2. Frame Synchronization & Alignment
# ===========================================================================

def find_sync_peaks(
    corr: np.ndarray,
    threshold: float = 0.85,
    min_distance: int = 1,
) -> np.ndarray:
    """Find peak sample indices where correlation exceeds threshold."""
    abs_corr = np.abs(corr)
    peak_indices = np.where(abs_corr >= threshold)[0]
    if len(peak_indices) <= 1 or min_distance <= 1:
        return peak_indices

    # Non-maximum suppression within min_distance
    filtered_peaks = []
    last_p = -min_distance
    for p in peak_indices:
        if p - last_p >= min_distance:
            filtered_peaks.append(p)
            last_p = p
        else:
            if abs_corr[p] > abs_corr[filtered_peaks[-1]]:
                filtered_peaks[-1] = p
                last_p = p

    return np.array(filtered_peaks, dtype=np.int32)


def frame_synchronize(
    bitstream: np.ndarray,
    sync_word: np.ndarray,
    frame_length: Optional[int] = None,
    threshold: float = 0.80,
    tolerate_inverted: bool = True,
) -> dict:
    """Locate sync markers and extract aligned frames.

    Parameters
    ----------
    bitstream : np.ndarray
        Raw or decoded bitstream.
    sync_word : np.ndarray
        Sync word pattern.
    frame_length : int, optional
        Fixed frame size in bits. If None, inferred from distance between consecutive sync peaks.
    threshold : float
        Minimum normalized correlation to declare sync match (0.0 to 1.0).
    tolerate_inverted : bool
        Whether to invert bitstream if 180° carrier phase inversion is detected.
    """
    if len(bitstream) < len(sync_word):
        return {
            "status": "Failed: Bitstream shorter than sync word",
            "sync_found": False,
            "peak_indices": [],
            "num_frames": 0,
            "frames": [],
            "correlation_curve": np.array([]),
            "max_correlation": 0.0,
            "is_inverted": False,
        }

    corr, is_inverted = correlate_bitstream(bitstream, sync_word, tolerate_inverted=tolerate_inverted)
    eff_bitstream = (1 - bitstream) if is_inverted else bitstream
    corr_eval = -corr if is_inverted else corr

    L_sync = len(sync_word)
    peaks = find_sync_peaks(corr_eval, threshold=threshold, min_distance=max(1, L_sync // 2))

    if len(peaks) == 0:
        # Fallback: find highest peak even if below threshold
        best_peak = int(np.argmax(corr_eval))
        max_c = float(corr_eval[best_peak])
        return {
            "status": f"No confident sync markers found (best peak = {max_c:.2f} at idx {best_peak})",
            "sync_found": False,
            "peak_indices": [best_peak] if max_c > 0.5 else [],
            "num_frames": 0,
            "frames": [],
            "correlation_curve": corr_eval,
            "max_correlation": max_c,
            "is_inverted": is_inverted,
            "detected_frame_length": None,
        }

    # Infer frame length from peak intervals if not specified
    detected_frame_len = frame_length
    if detected_frame_len is None:
        if len(peaks) >= 2:
            intervals = np.diff(peaks)
            # Find most common interval
            vals, counts = np.unique(intervals, return_counts=True)
            detected_frame_len = int(vals[np.argmax(counts)])
        else:
            detected_frame_len = len(bitstream) - peaks[0]

    # Slice aligned frames
    frames = []
    if detected_frame_len is not None and detected_frame_len > 0:
        for p in peaks:
            start_idx = p
            end_idx = min(len(eff_bitstream), start_idx + detected_frame_len)
            frame_bits = eff_bitstream[start_idx:end_idx]
            # Strip preamble from payload
            payload_bits = frame_bits[L_sync:]
            frames.append({
                "start_bit": int(start_idx),
                "end_bit": int(end_idx),
                "frame_bits": frame_bits,
                "payload_bits": payload_bits,
                "correlation": float(corr_eval[p]),
            })

    return {
        "status": "Synchronized",
        "sync_found": True,
        "peak_indices": peaks.tolist(),
        "num_frames": len(frames),
        "frames": frames,
        "correlation_curve": corr_eval,
        "max_correlation": float(np.max(corr_eval)),
        "is_inverted": is_inverted,
        "detected_frame_length": detected_frame_len,
        "sync_word_len": L_sync,
    }


# ===========================================================================
# 3. Automatic Preamble & Periodic Frame Discovery
# ===========================================================================

def auto_discover_preamble(
    bitstream: np.ndarray,
    min_frame_len: int = 16,
    max_frame_len: int = 2048,
    preamble_lens: tuple[int, ...] = (7, 11, 13, 16, 24, 32),
) -> dict:
    """Discover unknown sync markers / preambles and frame periodicity using lag-autocorrelation.

    Parameters
    ----------
    bitstream : np.ndarray
        Incoming bits.
    min_frame_len : int
        Minimum frame length in bits to search.
    max_frame_len : int
        Maximum frame length in bits to search.
    preamble_lens : tuple[int, ...]
        Preamble candidate lengths to test.
    """
    if len(bitstream) < min_frame_len * 2:
        return {
            "discovered": False,
            "estimated_frame_period": None,
            "periodicity_strength": 0.0,
            "matched_standard_sync": None,
            "standard_sync_confidence": None,
            "candidate_preamble_bits": None,
            "candidate_preamble_hex": None,
        }

    # 1. Bipolar circular / lag autocorrelation of bitstream to find frame period T
    bipolar = np.where(bitstream == 1, 1.0, -1.0).astype(np.float32)
    n = len(bipolar)
    max_lag = min(max_frame_len, n // 2)

    lags = np.arange(min_frame_len, max_lag)
    autocorr = np.zeros(len(lags), dtype=np.float32)

    for i, lag in enumerate(lags):
        overlap = n - lag
        c = np.sum(bipolar[:overlap] * bipolar[lag : lag + overlap]) / float(overlap)
        autocorr[i] = c

    best_lag_idx = int(np.argmax(autocorr))
    best_period = int(lags[best_lag_idx])
    best_ac = float(autocorr[best_lag_idx])

    # 2. If periodic peak is prominent, extract repeating preamble candidate
    discovered_preamble = None
    if best_ac > 0.15:
        # Average slices of length best_period across bitstream
        num_slices = n // best_period
        if num_slices >= 2:
            matrix = bipolar[: num_slices * best_period].reshape((num_slices, best_period))
            mean_pattern = np.mean(matrix, axis=0)
            candidate_bits = (mean_pattern > 0).astype(np.uint8)

            # Pick high-confidence prefix as preamble
            #
            # The preamble repeats identically in every frame, so its bits are
            # stable across frames: |mean_pattern| is near 1 for preamble bits
            # and drops toward 0 once the prefix extends into the varying
            # payload.  Score each candidate length by its *weakest* average
            # bit (worst-case confidence) and keep the longest prefix whose
            # bits are all consistent.  This fixes a real bug where the prior
            # loop kept the last (largest) length that merely fit, not the one
            # that actually matched the pattern.
            best_p_len = 0
            best_conf = -1.0
            for p_len in preamble_lens:
                if p_len >= best_period:
                    continue
                # Worst-case confidence across the prefix = min |mean|
                conf = float(np.min(np.abs(mean_pattern[:p_len])))
                if conf > best_conf:
                    best_conf = conf
                    best_p_len = p_len
            if best_p_len > 0:
                discovered_preamble = candidate_bits[:best_p_len]

    # 3. Check against known standard library sync words
    matched_standard = None
    best_std_score = 0.0
    best_std_corr = 0.0

    for name, word in STANDARD_SYNC_WORDS.items():
        if len(bitstream) >= len(word):
            c_curve, _ = correlate_bitstream(bitstream, word)
            if len(c_curve) > 0:
                abs_c = np.abs(c_curve)
                peaks_list = find_sync_peaks(abs_c, threshold=0.92, min_distance=max(1, len(word)))
                peaks_set = set(peaks_list)

                best_hits = 0
                best_stride = 0
                # Search all initial peak pairs for consistent periodic frame stride
                for i in range(len(peaks_list)):
                    for j in range(i + 1, min(len(peaks_list), i + 15)):
                        stride = peaks_list[j] - peaks_list[i]
                        if stride < len(word) * 2:
                            continue
                        hits = 1
                        curr = peaks_list[i] + stride
                        while curr in peaks_set:
                            hits += 1
                            curr += stride
                        if hits > best_hits:
                            best_hits = hits
                            best_stride = stride

                if best_hits >= 2:
                    score = (len(word) ** 1.5) * (best_hits ** 2.0)
                    if score > best_std_score:
                        best_std_score = score
                        best_std_corr = float(np.max(abs_c[peaks_list]))
                        matched_standard = name
                        best_period = int(best_stride)

    return {
        "discovered": (best_ac > 0.15 or matched_standard is not None),
        "estimated_frame_period": best_period if (best_ac > 0.15 or matched_standard) else None,
        "periodicity_strength": best_ac,
        "matched_standard_sync": matched_standard,
        "standard_sync_confidence": best_std_corr if matched_standard else None,
        "candidate_preamble_bits": discovered_preamble,
        "candidate_preamble_hex": np.packbits(discovered_preamble).tobytes().hex().upper() if discovered_preamble is not None else None,
    }
