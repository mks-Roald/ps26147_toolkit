"""De-interleaving algorithms for reversing bit/symbol interleaving.

Implements four standard de-interleaving methods required by PS26147:
  1. Block        – matrix transpose (write cols, read rows)
  2. Convolutional – Forney/Ramsey complementary shift-register delays
  3. Diagonal     – diagonal-fill matrix permutation inverse
  4. Pseudo-Random – PRBS-seeded permutation inverse

Each function accepts a 1-D ``np.ndarray`` (dtype ``uint8``, values 0/1) and
returns the de-interleaved bitstream of the same length plus any metadata.

An ``auto_detect_and_deinterleave`` helper tries all four methods with a grid of
common parameters and picks the one that yields the most structured output
(measured by byte-level entropy reduction and run-length statistics).
"""

from __future__ import annotations

import numpy as np
from typing import Literal

# Import ConvolutionalCodec for the new auto-detection function
from .fec_decoders import ConvolutionalCodec


# ---------------------------------------------------------------------------
# 1.  Block De-interleaver
# ---------------------------------------------------------------------------

def block_deinterleave(
    bits: np.ndarray,
    rows: int,
    cols: int,
) -> np.ndarray:
    """Reverse a block interleaver that wrote *row-wise* and read *column-wise*.

    The interleaver fills an (R × C) matrix row-by-row and transmits
    column-by-column.  The de-interleaver therefore writes column-by-column
    (i.e. fills columns first) and reads row-by-row.

    Parameters
    ----------
    bits : np.ndarray
        Interleaved bitstream (uint8, values 0/1).
    rows : int
        Number of rows in the interleaving matrix.
    cols : int
        Number of columns in the interleaving matrix.

    Returns
    -------
    np.ndarray
        De-interleaved bitstream (same length, padded / trimmed as needed).
    """
    block_size = rows * cols
    if block_size <= 0:
        return bits.copy()

    n = len(bits)
    # Pad to a multiple of block_size so reshape is clean
    pad_len = (block_size - n % block_size) % block_size
    padded = np.concatenate([bits, np.zeros(pad_len, dtype=bits.dtype)]) if pad_len else bits

    num_blocks = len(padded) // block_size
    output = np.empty_like(padded)

    for b in range(num_blocks):
        block = padded[b * block_size : (b + 1) * block_size]
        # Interleaver wrote row-wise, read column-wise  →  data arrives in
        # column-major order.  Reshape as (rows, cols) column-major and read
        # row-major to recover original order.
        matrix = block.reshape((rows, cols), order="F")
        output[b * block_size : (b + 1) * block_size] = matrix.ravel(order="C")

    return output[:n]


# ---------------------------------------------------------------------------
# 2.  Convolutional De-interleaver  (Forney / Ramsey)
# ---------------------------------------------------------------------------

def convolutional_deinterleave(
    bits: np.ndarray,
    num_branches: int,
    delay: int,
) -> np.ndarray:
    """Reverse a convolutional (Forney) interleaver.

    The *interleaver* has ``num_branches`` (N) branches with delays
    ``[0, D, 2D, …, (N-1)·D]``.  The complementary *de-interleaver* applies
    delays ``[(N-1)·D, (N-2)·D, …, D, 0]`` so that the total delay on every
    branch equals ``(N-1)·D``.

    Parameters
    ----------
    bits : np.ndarray
        Interleaved bitstream.
    num_branches : int
        Number of branches (N).
    delay : int
        Base unit delay (D) in bits.

    Returns
    -------
    np.ndarray
        De-interleaved bitstream.
    """
    if num_branches <= 1 or delay <= 0:
        return bits.copy()

    n = len(bits)
    N = num_branches
    D = delay

    # Each branch i (0-indexed) gets delay (N - 1 - i) * D
    # Initialise FIFO buffers for each branch (filled with zeros)
    buffers: list[list[int]] = [
        [0] * ((N - 1 - i) * D) for i in range(N)
    ]

    output = np.zeros(n, dtype=bits.dtype)
    for k in range(n):
        branch = k % N
        buffers[branch].append(int(bits[k]))
        output[k] = buffers[branch].pop(0)

    return output


# ---------------------------------------------------------------------------
# 3.  Diagonal De-interleaver
# ---------------------------------------------------------------------------

def diagonal_deinterleave(
    bits: np.ndarray,
    rows: int,
    cols: int,
) -> np.ndarray:
    """Reverse a diagonal interleaver.

    The *interleaver* fills an (R × C) matrix along successive diagonals
    (row + col = const) and reads row-by-row.  The de-interleaver inverts
    this mapping.

    Parameters
    ----------
    bits : np.ndarray
        Interleaved bitstream.
    rows : int
        Number of rows in the interleaving matrix.
    cols : int
        Number of columns in the interleaving matrix.

    Returns
    -------
    np.ndarray
        De-interleaved bitstream.
    """
    block_size = rows * cols
    if block_size <= 0:
        return bits.copy()

    # Build the diagonal write-order index mapping for one block:
    # diagonal d = r + c, iterate d = 0 … (R + C - 2)
    write_order: list[int] = []
    for d in range(rows + cols - 1):
        for r in range(rows):
            c = d - r
            if 0 <= c < cols:
                write_order.append(r * cols + c)

    # The interleaver wrote bits into the matrix in diagonal order (write_order)
    # and then read row-by-row (natural order).
    # interleaved[flat_idx] = original[write_order[flat_idx]]
    # So to de-interleave:  original[write_order[flat_idx]] = interleaved[flat_idx]
    perm = np.array(write_order, dtype=np.intp)

    n = len(bits)
    pad_len = (block_size - n % block_size) % block_size
    padded = np.concatenate([bits, np.zeros(pad_len, dtype=bits.dtype)]) if pad_len else bits

    num_blocks = len(padded) // block_size
    output = np.empty_like(padded)

    for b in range(num_blocks):
        blk = padded[b * block_size : (b + 1) * block_size]
        out_blk = np.empty(block_size, dtype=bits.dtype)
        out_blk[perm] = blk
        output[b * block_size : (b + 1) * block_size] = out_blk

    return output[:n]


# ---------------------------------------------------------------------------
# 4.  Pseudo-Random De-interleaver
# ---------------------------------------------------------------------------

def pseudorandom_deinterleave(
    bits: np.ndarray,
    block_size: int,
    seed: int = 42,
) -> np.ndarray:
    """Reverse a pseudo-random (PRBS-seeded) interleaver.

    The interleaver generates a random permutation of ``block_size`` indices
    (using ``seed``) and scatters the input bits accordingly.  The
    de-interleaver applies the *inverse* permutation.

    Parameters
    ----------
    bits : np.ndarray
        Interleaved bitstream.
    block_size : int
        Block size used by the interleaver permutation.
    seed : int
        PRBS seed (must match the interleaver setting).

    Returns
    -------
    np.ndarray
        De-interleaved bitstream.
    """
    if block_size <= 1:
        return bits.copy()

    rng = np.random.default_rng(seed)
    perm = rng.permutation(block_size)

    # Inverse permutation: inv_perm[perm[i]] = i
    inv_perm = np.empty_like(perm)
    inv_perm[perm] = np.arange(block_size)

    n = len(bits)
    pad_len = (block_size - n % block_size) % block_size
    padded = np.concatenate([bits, np.zeros(pad_len, dtype=bits.dtype)]) if pad_len else bits

    num_blocks = len(padded) // block_size
    output = np.empty_like(padded)

    for b in range(num_blocks):
        blk = padded[b * block_size : (b + 1) * block_size]
        output[b * block_size : (b + 1) * block_size] = blk[inv_perm]

    return output[:n]


# ---------------------------------------------------------------------------
# Quality metric  –  byte-level Shannon entropy
# ---------------------------------------------------------------------------

def _byte_entropy(bits: np.ndarray) -> float:
    """Compute Shannon entropy (bits/byte) of a bitstream packed into bytes.

    Lower entropy ⇒ more structured/repetitive data ⇒ likely correct
    de-interleaving.
    """
    if len(bits) < 8:
        return 8.0
    packed = np.packbits(bits)
    if len(packed) == 0:
        return 8.0
    _, counts = np.unique(packed, return_counts=True)
    probs = counts / counts.sum()
    return float(-np.sum(probs * np.log2(probs + 1e-12)))


def _run_length_score(bits: np.ndarray) -> float:
    """Score based on run-length distribution – structured data tends to have
    longer runs of identical bits than purely random data.

    Returns a positive score; higher is better (more structure).
    """
    if len(bits) < 4:
        return 0.0
    diffs = np.diff(bits.astype(np.int8))
    transitions = np.count_nonzero(diffs)
    # Ratio of transitions to total length (random ≈ 0.5, structured < 0.5)
    ratio = transitions / (len(bits) - 1)
    return max(0.0, 0.5 - ratio)


# ---------------------------------------------------------------------------
# Auto-detection helper (entropy-based)
# ---------------------------------------------------------------------------

# Common parameter grids to search
_BLOCK_SIZES = [(8, 8), (8, 16), (16, 8), (16, 16), (8, 32), (32, 8),
                (4, 16), (16, 4), (12, 12), (10, 10), (20, 20)]
_CONV_PARAMS = [(4, 4), (4, 8), (8, 4), (8, 8), (4, 16), (12, 17),
                (6, 8), (8, 16), (16, 8)]
_DIAG_SIZES = [(8, 8), (8, 16), (16, 8), (16, 16), (10, 10), (12, 12)]
_PR_BLOCK_SIZES = [64, 128, 256, 512, 1024]
_PR_SEEDS = [0, 1, 42, 123, 255]


DeinterleaverMethod = Literal["block", "convolutional", "diagonal", "pseudo-random", "none"]


def auto_detect_and_deinterleave(
    bits: np.ndarray,
    min_bits: int = 256,
    min_entropy_improvement: float = 0.2,
) -> dict:
    """Try all four de-interleaving methods with common parameter grids and
    return the result that yields the most structured output.

    Prompt 2 threshold logic
    ------------------------
    Two guards prevent false-positive detection on short or high-entropy
    (already near-random, e.g. viterbi-encoded) bitstreams:

    1. **Minimum signal length** (``min_bits``, default 256): fewer bits give
       unreliable byte-level entropy estimates because only a handful of byte
       values appear, making any permutation look like an improvement.  Signals
       shorter than this threshold are returned unchanged (method='none').

    2. **Minimum absolute entropy improvement** (``min_entropy_improvement``,
       default 1.5 bits/byte): a de-interleaving candidate is accepted only if
       its byte-level entropy is at least this many bits/byte lower than the
       baseline.  This avoids applying a permutation that happens to marginally
       win the heuristic score due to sampling noise — the improvement must be
       large enough to be meaningfully structural.

    Parameters
    ----------
    bits : np.ndarray
        Interleaved bitstream.
    min_bits : int
        Minimum number of bits required to attempt auto-detection.
    min_entropy_improvement : float
        Minimum absolute reduction in bits/byte entropy to accept a candidate.

    Returns
    -------
    dict
        ``method``          – detected method name (str)
        ``params``          – dict of best parameters
        ``bits``            – de-interleaved bitstream
        ``entropy``         – byte-level entropy of the output
        ``baseline_entropy``– byte-level entropy of the raw input
    """
    baseline_entropy = _byte_entropy(bits)
    baseline_rls = _run_length_score(bits)

    identity_result: dict = {
        "method": "none",
        "params": {},
        "bits": bits.copy(),
        "entropy": baseline_entropy,
        "baseline_entropy": baseline_entropy,
    }

    # Prompt 2 guard 1: bitstream too short for reliable entropy estimation.
    if len(bits) < min_bits:
        return identity_result

    best_score = 0.0   # improvement over baseline
    best_result: dict = identity_result.copy()

    def _eval(candidate: np.ndarray, method: str, params: dict):
        nonlocal best_score, best_result
        ent = _byte_entropy(candidate)
        rls = _run_length_score(candidate)
        # Combined improvement score (lower entropy + higher run-length)
        score = (baseline_entropy - ent) + 1.0 * (rls - baseline_rls)
        # Prompt 2 guard 2: require a minimum absolute entropy improvement to
        # avoid accepting candidates that only marginally beat the baseline.
        entropy_improvement = baseline_entropy - ent
        if score > best_score and entropy_improvement >= min_entropy_improvement:
            best_score = score
            best_result = {
                "method": method,
                "params": params,
                "bits": candidate,
                "entropy": ent,
                "baseline_entropy": baseline_entropy,
            }

    n = len(bits)

    # 1. Block
    for r, c in _BLOCK_SIZES:
        if r * c > n:
            continue
        out = block_deinterleave(bits, r, c)
        _eval(out, "block", {"rows": r, "cols": c})

    # 2. Convolutional
    for nb, d in _CONV_PARAMS:
        out = convolutional_deinterleave(bits, nb, d)
        _eval(out, "convolutional", {"num_branches": nb, "delay": d})

    # 3. Diagonal
    for r, c in _DIAG_SIZES:
        if r * c > n:
            continue
        out = diagonal_deinterleave(bits, r, c)
        _eval(out, "diagonal", {"rows": r, "cols": c})

    # 4. Pseudo-Random
    for bs in _PR_BLOCK_SIZES:
        if bs > n:
            continue
        for seed in _PR_SEEDS:
            out = pseudorandom_deinterleave(bits, bs, seed)
            _eval(out, "pseudo-random", {"block_size": bs, "seed": seed})

    return best_result


# ---------------------------------------------------------------------------
# New FEC-based auto-detection helper
# ---------------------------------------------------------------------------

def auto_detect_deinterleave_by_fec(bits: np.ndarray) -> dict:
    """Auto-detect deinterleaving by minimizing FEC mismatch.

    Candidates: "none" + block (_BLOCK_SIZES jahan r*c <= len(bits)) +
                convolutional (_CONV_PARAMS) + diagonal (_DIAG_SIZES) +
                pseudo-random (_PR_BLOCK_SIZES x _PR_SEEDS).

    For each candidate: c = deinterleave(bits)[:512];
                        dec = ConvolutionalCodec().decode(c);
                        re = ConvolutionalCodec().encode(dec, flush=True);
                        mismatch = mean(re[:n] != c[:n]), n=min(len(re),len(c)).
    Pick candidate with smallest mismatch. Accept only if:
        best_mismatch < 0.5 * none_mismatch AND best_mismatch < 0.5 * second_best_mismatch.
    Otherwise, return method="none", bits unchanged.

    Return dict shape matches auto_detect_and_deinterleave plus "mismatch" key.
    """
    # If bits are too short, return none
    if len(bits) < 2:
        return {
            "method": "none",
            "params": {},
            "bits": bits.copy(),
            "entropy": _byte_entropy(bits),
            "baseline_entropy": _byte_entropy(bits),
            "mismatch": 0.5,  # arbitrary high mismatch for short bits
        }

    # Helper to compute mismatch for a given deinterleaved candidate (first 512 bits)
    def compute_mismatch(candidate_bits: np.ndarray) -> float:
        # Take first 512 bits (or as many as available)
        c = candidate_bits[:512]
        if len(c) == 0:
            return 0.5
        # Decode with convolutional codec (Viterbi)
        try:
            dec = ConvolutionalCodec().decode(c)
            # Re-encode the decoded bits
            re = ConvolutionalCodec().encode(dec, flush=True)
            n = min(len(re), len(c))
            if n == 0:
                return 0.5
            mismatch = np.mean(re[:n] != c[:n])
            return float(mismatch)
        except Exception:
            # If decoding fails, return high mismatch
            return 0.5

    # Evaluate "none" candidate
    none_mismatch = compute_mismatch(bits)
    # We'll collect all candidates and their mismatches
    candidates = [("none", {}, bits, none_mismatch)]

    n = len(bits)

    # Block candidates
    for r, c in _BLOCK_SIZES:
        if r * c > n:
            continue
        out = block_deinterleave(bits, r, c)
        mismatch = compute_mismatch(out)
        candidates.append(("block", {"rows": r, "cols": c}, out, mismatch))

    # Convolutional candidates
    for nb, d in _CONV_PARAMS:
        out = convolutional_deinterleave(bits, nb, d)
        mismatch = compute_mismatch(out)
        candidates.append(("convolutional", {"num_branches": nb, "delay": d}, out, mismatch))

    # Diagonal candidates
    for r, c in _DIAG_SIZES:
        if r * c > n:
            continue
        out = diagonal_deinterleave(bits, r, c)
        mismatch = compute_mismatch(out)
        candidates.append(("diagonal", {"rows": r, "cols": c}, out, mismatch))

    # Pseudo-random candidates
    for bs in _PR_BLOCK_SIZES:
        if bs > n:
            continue
        for seed in _PR_SEEDS:
            out = pseudorandom_deinterleave(bits, bs, seed)
            mismatch = compute_mismatch(out)
            candidates.append(("pseudo-random", {"block_size": bs, "seed": seed}, out, mismatch))

    # Sort by mismatch (ascending)
    candidates.sort(key=lambda x: x[3])

    # Extract best, second best, and none (which is first in original list but we have sorted)
    best_method, best_params, best_bits, best_mismatch = candidates[0]
    # Find second best (skip if same as best? we'll take next distinct)
    second_best_mismatch = None
    for cand in candidates[1:]:
        if cand[3] != best_mismatch:
            second_best_mismatch = cand[3]
            break
    if second_best_mismatch is None:
        # All candidates have same mismatch? then use none_mismatch as second best for safety
        second_best_mismatch = none_mismatch

    # Acceptance condition: best_mismatch < 0.5 * none_mismatch AND best_mismatch < 0.5 * second_best_mismatch
    if best_mismatch < 0.5 * none_mismatch and best_mismatch < 0.5 * second_best_mismatch:
        # Use the best candidate
        selected_method, selected_params, selected_bits = best_method, best_params, best_bits
        selected_mismatch = best_mismatch
    else:
        # Fall back to none
        selected_method, selected_params, selected_bits = "none", {}, bits.copy()
        selected_mismatch = none_mismatch

    # Compute entropy and baseline entropy for the selected bits
    entropy = _byte_entropy(selected_bits)
    baseline_entropy = _byte_entropy(bits)

    return {
        "method": selected_method,
        "params": selected_params,
        "bits": selected_bits,
        "entropy": entropy,
        "baseline_entropy": baseline_entropy,
        "mismatch": selected_mismatch,
    }


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def deinterleave(
    bits: np.ndarray,
    method: DeinterleaverMethod = "block",
    *,
    rows: int = 8,
    cols: int = 8,
    num_branches: int = 4,
    delay: int = 8,
    block_size: int = 128,
    seed: int = 42,
) -> dict:
    """Apply a specific de-interleaving method with explicit parameters.

    Parameters
    ----------
    bits : np.ndarray
        Input interleaved bitstream.
    method : str
        One of ``"block"``, ``"convolutional"``, ``"diagonal"``,
        ``"pseudo-random"``, or ``"none"``.
    rows, cols : int
        Matrix dimensions for block / diagonal methods.
    num_branches, delay : int
        Branch count and unit delay for convolutional method.
    block_size, seed : int
        Block size and PRBS seed for pseudo-random method.

    Returns
    -------
    dict
        ``method``  – method name
        ``params``  – parameters used
        ``bits``    – de-interleaved bitstream
        ``entropy`` – byte-level entropy of the output
    """
    if method == "block":
        out = block_deinterleave(bits, rows, cols)
        params = {"rows": rows, "cols": cols}
    elif method == "convolutional":
        out = convolutional_deinterleave(bits, num_branches, delay)
        params = {"num_branches": num_branches, "delay": delay}
    elif method == "diagonal":
        out = diagonal_deinterleave(bits, rows, cols)
        params = {"rows": rows, "cols": cols}
    elif method == "pseudo-random":
        out = pseudorandom_deinterleave(bits, block_size, seed)
        params = {"block_size": block_size, "seed": seed}
    else:
        out = bits.copy()
        params = {}

    return {
        "method": method,
        "params": params,
        "bits": out,
        "entropy": _byte_entropy(out),
    }