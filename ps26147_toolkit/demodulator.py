import numpy as np
from scipy.signal import hilbert, resample_poly


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


def gardner_timing_recovery(
    sig: np.ndarray,
    sps: float,
    loop_bw: float = 0.01,
) -> np.ndarray:
    """Gardner Timing Error Detector (TED) with fractional interpolation.

    Phase 4 Enhancement: Adaptive symbol clock tracking that samples at optimal
    eye-diagram opening using fractional cubic interpolation.

    Args:
        sig: Complex baseband signal
        sps: Samples per symbol (float, can be fractional)
        loop_bw: Loop bandwidth for timing recovery loop filter

    Returns:
        Array of recovered symbols (1 sample per symbol)
    """
    if sps <= 1.0:
        return sig

    n = len(sig)
    mu = 0.0  # Fractional timing offset [0, 1)
    mu_samples = []
    symbols = []

    # Loop filter gains
    alpha = loop_bw
    beta = (alpha ** 2) / 4.0

    # Tracking variables
    sample_idx = sps  # Start after first symbol period
    prev_sample = sig[0]
    mid_sample = sig[0]

    while sample_idx < n - sps:
        # Fractional interpolation using cubic (4-point) interpolation
        base_idx = int(np.floor(sample_idx))
        frac = sample_idx - base_idx

        # Ensure we have enough samples for 4-point interpolation
        if base_idx < 1 or base_idx + 2 >= n:
            break

        # Cubic interpolation: y(mu) = y[-1]*c0 + y[0]*c1 + y[1]*c2 + y[2]*c3
        c0 = -frac * (frac - 1) * (frac - 2) / 6.0
        c1 = (frac + 1) * (frac - 1) * (frac - 2) / 2.0
        c2 = -(frac + 1) * frac * (frac - 2) / 2.0
        c3 = (frac + 1) * frac * (frac - 1) / 6.0

        interpolated = (
            c0 * sig[base_idx - 1]
            + c1 * sig[base_idx]
            + c2 * sig[base_idx + 1]
            + c3 * sig[base_idx + 2]
        )

        symbols.append(interpolated)

        # Gardner TED: error = real[(x[n] - x[n-2]) * conj(x[n-1])]
        # Uses current, previous, and midpoint samples
        mid_idx = int(np.floor(sample_idx - sps / 2.0))
        if mid_idx >= 0 and mid_idx < n:
            mid_sample = sig[mid_idx]

        # Gardner error detector (works for most modulations)
        error = ((interpolated - prev_sample) * np.conj(mid_sample)).real

        # Update timing with loop filter
        mu += beta * error
        sample_idx += sps + alpha * error + mu

        # Clamp mu to prevent runaway
        if mu > 0.5:
            mu -= 1.0
            sample_idx -= 1.0
        elif mu < -0.5:
            mu += 1.0
            sample_idx += 1.0

        prev_sample = interpolated

    return np.array(symbols, dtype=np.complex64)


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
        # FSK (Phase 6 §1.6): bits live in instantaneous frequency, not the
        # static complex constellation, so the generic distant-based LLR is
        # meaningless here.  Derive LLR from the signed, diff'd instantaneous
        # frequency estimate -- the same signal slice_symbols_to_bits uses:
        #   dev = diff(unwrap(angle(symbols))),  bit 1 when dev >= 0.
        # LLR convention (positive -> bit 0, negative -> bit 1) then gives
        #   llr = -dev / noise_variance,  scaled by discriminator noise var.
        dev = np.diff(np.unwrap(np.angle(symbols)))
        if "4FSK" in mod_upper:
            # 4FSK maps a sample of dev to 2 bits via the population quantiles
            # q1 < q2 < q3  (see slice_symbols_to_bits).  For each of the two
            # bit positions we soft-metric the distance of dev to its decision
            # boundary, scaled by noise_variance.  This is approximate (not a
            # full Gray-map), but strictly better than the -symbols.real fallback.
            if len(dev) == 0:
                return np.array([], dtype=np.float32)
            q1, q2, q3 = np.percentile(dev, [25, 50, 75])
            out: list[float] = []
            for d in dev:
                # MSB (bit 0): boundary q2 splits the lower two levels from the
                # upper two; dev < q2 -> bit 0.
                out.append(float(np.clip(-(d - q2) / noise_variance, -20.0, 20.0)))
                # LSB (bit 1): boundary q1 within the lower half, q3 within the
                # upper half.
                b = q1 if d < q2 else q3
                out.append(float(np.clip(-(d - b) / noise_variance, -20.0, 20.0)))
            return np.array(out, dtype=np.float32)
        # 2FSK / generic FSK: 1 bit per deviation sample
        llrs = -dev / noise_variance
        return np.clip(llrs, -20.0, 20.0).astype(np.float32)

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

    return np.array(llrs, dtype=np.float32)


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
    }
