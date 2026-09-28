import numpy as np
from scipy.signal import hilbert, resample_poly
from functools import reduce
from math import gcd


# Ideal constellation templates for EVM calculation & reference
CONSTELLATIONS = {
    "BPSK": np.array([-1.0, 1.0], dtype=np.complex64),
    "QPSK": np.array([-1-1j, -1+1j, 1-1j, 1+1j], dtype=np.complex64) / np.sqrt(2),
    "8PSK": np.exp(1j * np.arange(8) * (2 * np.pi / 8)).astype(np.complex64),
    "16QAM": (
        np.tile(np.array([-3, -1, 1, 3]), 4) + 1j * np.repeat(np.array([-3, -1, 1, 3]), 4)
    ).astype(np.complex64) / np.sqrt(10),
    "64QAM": (
        np.tile(np.arange(-7, 8, 2), 8) + 1j * np.repeat(np.arange(-7, 8, 2), 8)
    ).astype(np.complex64) / np.sqrt(42),
}


def costas_carrier_recovery(
    sig: np.ndarray,
    order: int = 4,
    loop_bw: float = 0.01,
) -> np.ndarray:
    """Decision-directed / Costas PLL for carrier frequency and phase tracking.

    Order: 2 for BPSK, 4 for QPSK/QAM, 8 for 8PSK.

    Phase 4 Enhancement:
    - Proper phase wrapping inside [-π/N, +π/N] to prevent accumulator drift
    - Dual-stage loop filter (proportional + integral) for stable lock tracking
    """
    n = len(sig)
    out_sig = np.zeros(n, dtype=np.complex64)

    phase = 0.0
    freq = 0.0
    alpha = loop_bw  # Proportional gain
    beta = (alpha ** 2) / 4.0  # Integral gain

    for i in range(n):
        # Rotate input by current phase estimate
        sample = sig[i] * np.exp(-1j * phase)
        out_sig[i] = sample

        # Phase error detector (order-adaptive)
        if order == 2:
            # BPSK error: sign(real) * imag
            error = np.sign(sample.real) * sample.imag
        elif order == 4:
            # QPSK / QAM error: sign(real)*imag - sign(imag)*real
            error = np.sign(sample.real) * sample.imag - np.sign(sample.imag) * sample.real
        elif order == 8:
            # 8PSK error: Use 8-th power method with proper wrapping
            angle = np.angle(sample)
            error_angle = 8 * angle
            # Wrap phase error inside [-π, +π] then scale by 1/N
            error = np.arctan2(np.sin(error_angle), np.cos(error_angle)) / 8.0
        else:
            error = sample.imag

        # Clamp error to prevent loop runaway
        error = np.clip(error, -2.0, 2.0)

        # Dual-stage loop filter: proportional + integral
        freq += beta * error  # Integral path (accumulates frequency offset)
        phase += freq + alpha * error  # Proportional path

        # Critical Phase 4 fix: Wrap phase accumulator to prevent drift
        # Keep phase bounded in [-π, +π]
        if phase > np.pi:
            phase -= 2 * np.pi
        elif phase < -np.pi:
            phase += 2 * np.pi

    return out_sig


def _cubic_interp(sig: np.ndarray, pos: float):
    b = int(np.floor(pos))
    f = pos - b
    if b < 1 or b + 2 >= len(sig):
        return None
    c0 = -f * (f - 1) * (f - 2) / 6.0
    c1 = (f + 1) * (f - 1) * (f - 2) / 2.0
    c2 = -(f + 1) * f * (f - 2) / 2.0
    c3 = (f + 1) * f * (f - 1) / 6.0
    return c0 * sig[b - 1] + c1 * sig[b] + c2 * sig[b + 1] + c3 * sig[b + 2]


def _acquire_symbol_phase(sig: np.ndarray, sps: float, max_syms: int = 400) -> float:
    """Find a symbol-centre sampling phase from the eye opening or transitions."""
    n = len(sig)
    L = int(round(sps))
    K = min(max_syms, (n - 2) // max(L, 1) - 1)
    if K < 8 or L < 4:
        return sps / 2.0
    m = (K + 1) * L + 1
    ang = 2 * np.pi * np.arange(L) / L

    E = np.array([np.mean(np.abs(sig[o:m:L][:K])) for o in range(L)])
    if (E.max() - E.min()) > 0.15 * np.mean(E):
        w = np.clip(E - E.min(), 0, None) ** 2
        return (np.angle(np.sum(w * np.exp(1j * ang))) % (2 * np.pi)) / (2 * np.pi) * L

    d = np.abs(np.diff(sig[:m])) ** 2
    T = np.array([np.mean(d[o::L][:K]) for o in range(L)])
    if (T.max() - T.min()) < 0.5 * np.mean(T) + 1e-12:
        return sps / 2.0
    w = np.clip(T - T.min(), 0, None)
    boundary = (np.angle(np.sum(w * np.exp(1j * ang))) % (2 * np.pi)) / (2 * np.pi) * L
    return (boundary + 1.0 + sps / 2.0) % sps


def gardner_timing_recovery(sig: np.ndarray, sps: float, loop_bw: float = 0.03) -> np.ndarray:
    """Gardner TED, 2nd-order loop with acquired phase and cubic interpolation."""
    if sps <= 1.0:
        return sig
    n = len(sig)
    alpha = loop_bw
    beta = (alpha ** 2) / 4.0
    fmax = 0.02 * sps

    pos = _acquire_symbol_phase(sig, sps)
    while pos < 2.0:
        pos += sps

    freq = 0.0
    prev = None
    out = []
    while pos < n - sps - 2:
        x = _cubic_interp(sig, pos)
        if x is None:
            pos += 1.0
            continue
        out.append(x)
        step = sps
        if prev is not None:
            m1 = _cubic_interp(sig, pos - sps / 2.0 - 0.5)
            m2 = _cubic_interp(sig, pos - sps / 2.0 + 0.5)
            if m1 is not None and m2 is not None:
                mid = 0.5 * (m1 + m2)
                pw = 0.5 * (abs(x) ** 2 + abs(prev) ** 2) + 1e-12
                err = float(((x - prev) * np.conj(mid)).real / pw)
                err = max(-2.0, min(2.0, err))
                freq = max(-fmax, min(fmax, freq + beta * err))
                step = sps - alpha * err - freq
        pos += step
        prev = x
    return np.array(out, dtype=np.complex64)


def run_length_baud_factor(bits, min_runs: int = 40, max_k: int = 8) -> int:
    """Detect integer baud overestimates from the GCD of complete bit runs."""
    b = np.asarray(bits).astype(np.int8)
    if len(b) < 64:
        return 1
    runs = np.diff(np.flatnonzero(np.diff(b) != 0))
    if len(runs) < min_runs:
        return 1
    g = int(reduce(gcd, [int(x) for x in runs]))
    return g if 2 <= g <= max_k else 1


def symbol_timing_recovery(
    sig: np.ndarray,
    fs: float,
    baud_rate: float,
    method: str = "gardner",
) -> np.ndarray:
    """Symbol timing recovery wrapper supporting multiple methods.

    Phase 4 Enhancement: Replaced basic integer decimation with adaptive
    fractional symbol clock tracking.

    Args:
        sig: Complex baseband signal
        fs: Sampling rate (Hz)
        baud_rate: Symbol rate (baud)
        method: Recovery method - "gardner" (default) or "simple" (legacy)

    Returns:
        Array of recovered symbols
    """
    if baud_rate <= 0 or fs <= 0:
        return sig

    sps = fs / baud_rate
    if sps <= 1.0:
        return sig

    if method == "gardner":
        return gardner_timing_recovery(sig, sps)

    # Legacy simple method (fallback)
    int_sps = int(np.round(sps))
    if int_sps <= 1:
        return sig

    # Find the optimal sampling offset (0 .. int_sps-1) maximizing constellation variance/energy
    best_offset = 0
    best_metric = -1.0

    for offset in range(min(int_sps, len(sig))):
        decimated = sig[offset::int_sps]
        if len(decimated) < 10:
            continue
        # Metric: variance of envelope power or real/imag kurtosis
        metric = np.var(np.abs(decimated)) + np.var(decimated.real)
        if metric > best_metric:
            best_metric = metric
            best_offset = offset

    return sig[best_offset::int_sps]


def compute_soft_llr(
    symbols: np.ndarray,
    modulation: str,
    noise_variance: float = 0.1,
) -> np.ndarray:
    """Compute Log-Likelihood Ratios (LLR) for soft FEC decoding.

    Phase 4 Enhancement: Produces soft-decision metrics for each bit based on
    Euclidean distance to constellation points.

    LLR definition: LLR = log(P(bit=0|y) / P(bit=1|y))
    Positive LLR → bit likely 0, Negative LLR → bit likely 1

    Args:
        symbols: Received complex symbols (normalized)
        modulation: Modulation scheme
        noise_variance: Estimated noise variance (σ²) for scaling

    Returns:
        Array of LLRs (one per bit)
    """
    mod_upper = modulation.upper()

    # Special case for BPSK: simple real-part based LLR
    if "BPSK" in mod_upper:
        # BPSK: real >= 0 → bit 1, real < 0 → bit 0
        # LLR positive → bit 0, LLR negative → bit 1
        # So LLR = -real / noise_variance (negated to match convention)
        llrs = -symbols.real / noise_variance
        return np.clip(llrs, -20.0, 20.0).astype(np.float32)

    if "FSK" in mod_upper:
        # FSK: bits live in instantaneous frequency, not the static complex
        # constellation.  Derive LLR from the signed, np.diff'd instantaneous
        # phase -- the same signal slice_symbols_to_bits() uses for the hard
        # decision:
        #   dev = np.diff(np.unwrap(np.angle(symbols))),  bit 1 when dev >= 0.
        # LLR convention (positive -> bit 0, negative -> bit 1) gives:
        #   llr = -dev / disc_noise_var
        #
        # disc_noise_var is estimated from the spread of `dev` itself (the
        # actual deviation-domain noise), NOT from the constellation-domain
        # noise_variance argument, which is irrelevant for FSK.
        dev = np.diff(np.unwrap(np.angle(symbols)))  # length == len(symbols)-1

        # Estimate discriminator noise variance from deviation signal residuals.
        # Noise proxy: var(dev - sign(dev)*mean_abs_dev).
        # Falls back to noise_variance when dev is too short.
        if len(dev) >= 2:
            mean_abs_dev = float(np.mean(np.abs(dev)))
            disc_noise_var = float(np.var(dev - np.sign(dev) * mean_abs_dev))
            disc_noise_var = max(disc_noise_var, 1e-6)
        else:
            disc_noise_var = max(float(noise_variance), 1e-6)

        if "4FSK" in mod_upper:
            # 4FSK maps each dev sample to 2 bits via population quantiles
            # q1 < q2 < q3 (see slice_symbols_to_bits).
            # Returned length: 2*(len(symbols)-1) — matches 4FSK hard-bit count.
            if len(dev) == 0:
                llrs = np.array([], dtype=np.float32)
            else:
                q1, q2, q3 = np.percentile(dev, [25, 50, 75])
                out: list[float] = []
                for d in dev:
                    # MSB (bit 0): boundary q2 splits lower two levels from upper two.
                    out.append(float(np.clip(-(d - q2) / disc_noise_var, -20.0, 20.0)))
                    # LSB (bit 1): boundary q1 in lower half, q3 in upper half.
                    # Gray mapping: d<q1 -> 0, q1<=d<q2 -> 1, q2<=d<q3 -> 1, d>=q3 -> 0
                    if d < q2:
                        llr_b1 = -(d - q1) / disc_noise_var
                    else:
                        llr_b1 = (d - q3) / disc_noise_var
                    out.append(float(np.clip(llr_b1, -20.0, 20.0)))
                llrs = np.array(out, dtype=np.float32)
            expected_bit_count = 2 * (len(symbols) - 1)
            assert len(llrs) == expected_bit_count, (
                f"compute_soft_llr length mismatch: got {len(llrs)} LLRs, "
                f"expected {expected_bit_count} for {modulation} "
                f"(len(symbols)={len(symbols)})"
            )
            return llrs

        # 2FSK / generic FSK: 1 LLR per dev sample → exactly len(symbols)-1 LLRs.
        # This matches the hard-bit count from slice_symbols_to_bits() for FSK.
        llrs = np.clip(-dev / disc_noise_var, -20.0, 20.0).astype(np.float32)
        expected_bit_count = len(symbols) - 1
        assert len(llrs) == expected_bit_count, (
            f"compute_soft_llr length mismatch: got {len(llrs)} LLRs, "
            f"expected {expected_bit_count} for {modulation} "
            f"(len(symbols)={len(symbols)})"
        )
        return llrs

    constellation = CONSTELLATIONS.get(mod_upper.replace("-", ""))

    if constellation is None:
        # Fallback: treat as BPSK-like
        llrs = -symbols.real / noise_variance
        return np.clip(llrs, -20.0, 20.0).astype(np.float32)

    llrs = []
    bits_per_symbol = int(np.log2(len(constellation)))

    for sym in symbols:
        # Compute distances to all constellation points
        distances = np.abs(sym - constellation) ** 2

        # For each bit position, compute LLR
        for bit_pos in range(bits_per_symbol):
            # Indices where bit_pos is 0
            mask_0 = np.array([(i >> (bits_per_symbol - 1 - bit_pos)) & 1 == 0
                              for i in range(len(constellation))])
            # Indices where bit_pos is 1
            mask_1 = ~mask_0

            # Min distance among symbols with bit=0
            if np.any(mask_0):
                min_dist_0 = np.min(distances[mask_0])
            else:
                min_dist_0 = 1e6

            # Min distance among symbols with bit=1
            if np.any(mask_1):
                min_dist_1 = np.min(distances[mask_1])
            else:
                min_dist_1 = 1e6

            # LLR = (d1 - d0) / (2 * σ²)
            # Positive → bit=0 more likely, Negative → bit=1 more likely
            llr = (min_dist_1 - min_dist_0) / (2.0 * noise_variance)

            # Clamp to prevent overflow in FEC decoders
            llr = np.clip(llr, -20.0, 20.0)
            llrs.append(llr)

    llrs = np.array(llrs, dtype=np.float32)
    expected_bit_count = len(symbols) * bits_per_symbol
    assert len(llrs) == expected_bit_count, (
        f"compute_soft_llr length mismatch: got {len(llrs)} LLRs, "
        f"expected {expected_bit_count} for {modulation} "
        f"(len(symbols)={len(symbols)}, bits_per_symbol={bits_per_symbol})"
    )
    return llrs


def slice_symbols_to_bits(symbols: np.ndarray, modulation: str) -> tuple[np.ndarray, np.ndarray]:
    """Slice normalized complex symbols to nearest constellation points and extract bitstream.

    Returns (demodulated_bits, ideal_reference_symbols).
    """
    mod_upper = modulation.upper()
    bits_list = []
    ref_symbols = []

    if "BPSK" in mod_upper:
        # 1 bit per symbol
        for s in symbols:
            bit = 1 if s.real >= 0 else 0
            bits_list.append(bit)
            ref_symbols.append(1.0 if bit == 1 else -1.0)

    # NOTE: match QPSK/4QAM by *equality* rather than substring. "64QAM" and
    # "16QAM" both contain "4QAM" as a substring, which would otherwise route
    # them here (2 bits/symbol) and make the specific QAM branches unreachable.
    elif "QPSK" in mod_upper or mod_upper in {"4QAM", "4-QAM"}:
        # Gray-coded QPSK (2 bits per symbol)
        for s in symbols:
            b0 = 0 if s.real >= 0 else 1
            b1 = 0 if s.imag >= 0 else 1
            bits_list.extend([b0, b1])
            ref_re = (1.0 if b0 == 0 else -1.0) / np.sqrt(2)
            ref_im = (1.0 if b1 == 0 else -1.0) / np.sqrt(2)
            ref_symbols.append(ref_re + 1j * ref_im)

    elif "8PSK" in mod_upper:
        # Gray-coded 8PSK (3 bits per symbol)
        angles = np.angle(symbols) % (2 * np.pi)
        sector = (np.round(angles / (np.pi / 4)) % 8).astype(int)
        gray_map = {
            0: [0, 0, 0], 1: [0, 0, 1], 2: [0, 1, 1], 3: [0, 1, 0],
            4: [1, 1, 0], 5: [1, 1, 1], 6: [1, 0, 1], 7: [1, 0, 0],
        }
        for sec in sector:
            bits_list.extend(gray_map[sec])
            ref_symbols.append(np.exp(1j * sec * (np.pi / 4)))

    elif "16QAM" in mod_upper or "16-QAM" in mod_upper:
        # Gray-coded 16QAM (4 bits per symbol: 2 for I, 2 for Q)
        levels = np.array([-3, -1, 1, 3]) / np.sqrt(10)
        level_bits = {0: [0, 0], 1: [0, 1], 2: [1, 1], 3: [1, 0]}
        for s in symbols:
            idx_i = int(np.argmin(np.abs(s.real - levels)))
            idx_q = int(np.argmin(np.abs(s.imag - levels)))
            bits_list.extend(level_bits[idx_i] + level_bits[idx_q])
            ref_symbols.append(levels[idx_i] + 1j * levels[idx_q])

    elif "64QAM" in mod_upper or "64-QAM" in mod_upper:
        # 64QAM (6 bits per symbol: 3 for I, 3 for Q)
        levels = np.arange(-7, 8, 2) / np.sqrt(42)
        for s in symbols:
            idx_i = int(np.argmin(np.abs(s.real - levels)))
            idx_q = int(np.argmin(np.abs(s.imag - levels)))
            bits_i = [(idx_i >> 2) & 1, (idx_i >> 1) & 1, idx_i & 1]
            bits_q = [(idx_q >> 2) & 1, (idx_q >> 1) & 1, idx_q & 1]
            bits_list.extend(bits_i + bits_q)
            ref_symbols.append(levels[idx_i] + 1j * levels[idx_q])

    # NOTE: 4FSK must be matched *before* the generic "FSK" branch. "4FSK"
    # contains "FSK", so ordering the 2FSK/FSK branch first would route every
    # 4FSK symbol through the 1-bit slicer and make this branch unreachable.
    elif "4FSK" in mod_upper:
        diff = np.diff(np.unwrap(np.angle(symbols)))
        q1, q2, q3 = np.percentile(diff, [25, 50, 75])
        for d in diff:
            if d < q1:
                b = [0, 0]
            elif d < q2:
                b = [0, 1]
            elif d < q3:
                b = [1, 1]
            else:
                b = [1, 0]
            bits_list.extend(b)
            ref_symbols.append(1.0)

    elif "FSK" in mod_upper or "2FSK" in mod_upper:
        # Instantaneous frequency / phase slope slicing (2FSK & generic FSK)
        diff = np.diff(np.unwrap(np.angle(symbols)))
        for d in diff:
            b = 1 if d >= 0 else 0
            bits_list.append(b)
            ref_symbols.append(1.0 if b == 1 else -1.0)

    else:
        # Default binary envelope slicer
        env = np.abs(symbols)
        thresh = np.median(env)
        for e in env:
            bit = 1 if e >= thresh else 0
            bits_list.append(bit)
            ref_symbols.append(1.0 if bit == 1 else 0.0)

    return np.array(bits_list, dtype=np.uint8), np.array(ref_symbols, dtype=np.complex64)


def compute_evm(symbols: np.ndarray, ref_symbols: np.ndarray) -> dict:
    """Calculate Error Vector Magnitude (EVM) in dB and percentage.

    Phase 4 Enhancement: Returns both dB and percentage RMS EVM metrics.

    EVM_RMS = sqrt(mean(|s_rx - s_ref|²) / mean(|s_ref|²))

    Args:
        symbols: Received symbols
        ref_symbols: Ideal reference constellation points

    Returns:
        Dictionary with 'evm_db', 'evm_percent', and 'noise_variance'
    """
    if len(symbols) == 0 or len(ref_symbols) == 0:
        return {"evm_db": 0.0, "evm_percent": 0.0, "noise_variance": 0.1}

    min_len = min(len(symbols), len(ref_symbols))
    s = symbols[:min_len]
    r = ref_symbols[:min_len]

    error = s - r
    p_error = np.mean(np.abs(error) ** 2)
    p_ref = np.mean(np.abs(r) ** 2)

    if p_ref <= 1e-12:
        return {"evm_db": 0.0, "evm_percent": 0.0, "noise_variance": 0.1}

    evm_rms = np.sqrt(p_error / p_ref)
    evm_db = float(20.0 * np.log10(max(evm_rms, 1e-6)))
    evm_percent = float(evm_rms * 100.0)

    # Estimate noise variance for LLR calculation
    noise_var = float(p_error)

    return {
        "evm_db": round(evm_db, 2),
        "evm_percent": round(evm_percent, 2),
        "noise_variance": max(noise_var, 1e-6),
    }


def compute_fsk_evm(symbols: np.ndarray, modulation: str) -> dict:
    """Deviation-domain EVM for FSK (Phase 6 §1.6).

    FSK encodes bits in instantaneous *frequency*, not in a static complex
    constellation, so the constellation-domain ``compute_evm()`` measures phase
    rotation rather than demodulation error — a genuinely noiseless 2FSK file
    used to read 2.9–3.6 dB EVM when there was no actual error to measure.

    This computes the FSK analogue of EVM by:
      1. estimating the per-symbol instantaneous-frequency deviation
         ``dev = diff(unwrap(angle(symbols)))``  [rad/sample],
      2. normalizing by an estimated deviation magnitude so ideal levels sit
         at ±1,
      3. measuring the RMS distance of each sample to the decision-derived
         ideal (sign of ``dev``), normalized by ideal reference power.

    Returns the same shape as ``compute_evm()``: ``evm_db``, ``evm_percent``,
    ``noise_variance`` (the discriminator/deviation-domain noise — the scaling
    the FSK LLR branch uses).
    """
    mod_upper = modulation.upper()
    dev = np.diff(np.unwrap(np.angle(symbols)))
    n = len(dev)
    if n < 2:
        return {"evm_db": 0.0, "evm_percent": 0.0, "noise_variance": 0.1}

    delta = float(np.mean(np.abs(dev)))
    if delta <= 1e-12:
        return {"evm_db": 0.0, "evm_percent": 0.0, "noise_variance": 0.1}

    dev_norm = dev / delta
    ref = np.where(dev >= 0, 1.0, -1.0)          # decision-derived ideal ±1
    error = dev_norm - ref
    p_error = float(np.mean(error ** 2))
    p_ref = 1.0                                   # mean(ref**2) with ref=±1
    evm_rms = np.sqrt(p_error / p_ref)
    evm_db = float(20.0 * np.log10(max(evm_rms, 1e-6)))

    return {
        "evm_db": round(evm_db, 2),
        "evm_percent": round(evm_rms * 100.0, 2),
        "noise_variance": max(p_error, 1e-6),
    }


def demodulate_signal(
    signal: np.ndarray,
    fs: float,
    modulation: str,
    center_freq: float = 0.0,
    baud_rate: float = None,
    timing_method: str = "gardner",
    _baud_checked: bool = False,
) -> dict:
    """Complete demodulation pipeline with Phase 4 enhancements.

    Pipeline stages:
    1. Baseband downconversion & analytic conversion
    2. Symbol timing recovery (Gardner TED with fractional interpolation)
    3. Carrier phase synchronization (Costas Loop with drift prevention)
    4. Constellation normalization & slicing to bits
    5. EVM calculation and soft LLR generation

    Args:
        signal: Input IQ signal (complex or real)
        fs: Sampling rate (Hz)
        modulation: Modulation scheme string
        center_freq: Center frequency offset (Hz) for downconversion
        baud_rate: Symbol rate (baud) for timing recovery
        timing_method: "gardner" (default) or "simple" for timing recovery

    Returns:
        Dictionary containing:
        - symbols: Normalized constellation symbols
        - bits: Hard-decision bit stream
        - llr: Soft-decision Log-Likelihood Ratios
        - bit_string_preview: First 512 bits as string
        - hex_preview: First 64 bytes as hex string
        - num_bits: Total number of bits
        - evm_db: Error Vector Magnitude in dB
        - evm_percent: Error Vector Magnitude in %
        - modulation: Modulation scheme used
    """
    if not np.iscomplexobj(signal):
        sig = hilbert(signal)
    else:
        sig = np.copy(signal)

    # 1. Baseband frequency shift (downconversion)
    if abs(center_freq) > 0.01:
        t = np.arange(len(sig)) / fs
        sig_bb = sig * np.exp(-1j * 2 * np.pi * center_freq * t)
    else:
        sig_bb = sig

    # 2. Symbol timing recovery (Phase 4: Gardner TED)
    if baud_rate is not None and baud_rate > 0:
        symbols_raw = symbol_timing_recovery(sig_bb, fs, baud_rate, method=timing_method)
    else:
        symbols_raw = sig_bb

    # Cap to reasonable number of symbols for performance & plotting
    max_symbols = 4096
    if len(symbols_raw) > max_symbols:
        symbols_raw = symbols_raw[:max_symbols]

    # 3. Carrier PLL / Phase Tracking (Phase 4: with drift prevention)
    mod_upper = modulation.upper()
    if "FSK" in mod_upper:
        # FSK: information is in instantaneous frequency; carrier recovery would
        # remove the modulation. Skip carrier loop for FSK.
        symbols_tracked = symbols_raw
    else:
        order = 2 if "BPSK" in mod_upper else (8 if "8PSK" in mod_upper else 4)
        symbols_tracked = costas_carrier_recovery(symbols_raw, order=order)

    # 4. Energy normalization
    p_avg = np.mean(np.abs(symbols_tracked) ** 2)
    if p_avg > 1e-12:
        symbols_norm = symbols_tracked / np.sqrt(p_avg)
    else:
        symbols_norm = symbols_tracked

    # 5. Slicing to bits & reference symbols
    bits, ref_symbols = slice_symbols_to_bits(symbols_norm, modulation)

    if not _baud_checked and baud_rate is not None:
        k = run_length_baud_factor(bits)
        if k > 1:
            return demodulate_signal(
                signal, fs, modulation, center_freq, baud_rate / k,
                timing_method, _baud_checked=True,
            )

    # 6. EVM calculation.  FSK uses the deviation-domain metric (Phase 6 §1.6):
    #    the constellation-domain compute_evm() measures phase rotation, not
    #    demodulation error, for FSK.  Non-FSK keeps the constellation metric.
    if "FSK" in mod_upper:
        evm_results = compute_fsk_evm(symbols_norm, modulation)
    else:
        evm_results = compute_evm(symbols_norm, ref_symbols)

    # 7. Soft LLR generation (Phase 4: for soft FEC decoding)
    llr = compute_soft_llr(symbols_norm, modulation, evm_results["noise_variance"])

    # Format bit string & hex string preview
    bit_str = "".join(str(b) for b in bits[:512])
    # Convert bits to bytes
    byte_array = np.packbits(bits)
    hex_str = byte_array[:64].tobytes().hex().upper()

    return {
        "symbols": symbols_norm,
        "bits": bits,
        "llr": llr,
        "bit_string_preview": bit_str,
        "hex_preview": hex_str,
        "num_bits": len(bits),
        "evm_db": evm_results["evm_db"],
        "evm_percent": evm_results["evm_percent"],
        "modulation": modulation,
        "baud_rate_used": baud_rate,
    }
