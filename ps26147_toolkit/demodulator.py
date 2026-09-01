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
    """
    n = len(sig)
    out_sig = np.zeros(n, dtype=np.complex64)

    phase = 0.0
    freq = 0.0
    alpha = loop_bw
    beta = (alpha ** 2) / 4.0

    for i in range(n):
        # Rotate input by current phase estimate
        sample = sig[i] * np.exp(-1j * phase)
        out_sig[i] = sample

        # Phase error detector
        if order == 2:
            # BPSK error: sign(real) * imag
            error = np.sign(sample.real) * sample.imag
        elif order == 4:
            # QPSK / QAM error: sign(real)*imag - sign(imag)*real
            error = np.sign(sample.real) * sample.imag - np.sign(sample.imag) * sample.real
        elif order == 8:
            # 8PSK error: 8-th power phase
            angle = np.angle(sample)
            error = np.sin(8 * angle) / 8.0
        else:
            error = sample.imag

        # Clamp error to prevent loop runaway
        error = np.clip(error, -2.0, 2.0)

        # Loop filter
        freq += beta * error
        phase += freq + alpha * error

    return out_sig


def symbol_timing_recovery(
    sig: np.ndarray,
    fs: float,
    baud_rate: float,
) -> np.ndarray:
    """Downsample and align baseband signal to 1 sample per symbol at optimal eye opening."""
    if baud_rate <= 0 or fs <= 0:
        return sig

    sps = fs / baud_rate
    if sps <= 1.0:
        return sig

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

    elif "QPSK" in mod_upper or "4-QAM" in mod_upper or "4QAM" in mod_upper:
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

    elif "2FSK" in mod_upper or "FSK" in mod_upper:
        # Instantaneous frequency / phase slope slicing
        diff = np.diff(np.unwrap(np.angle(symbols)))
        for d in diff:
            b = 1 if d >= 0 else 0
            bits_list.append(b)
            ref_symbols.append(1.0 if b == 1 else -1.0)

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

    else:
        # Default binary envelope slicer
        env = np.abs(symbols)
        thresh = np.median(env)
        for e in env:
            bit = 1 if e >= thresh else 0
            bits_list.append(bit)
            ref_symbols.append(1.0 if bit == 1 else 0.0)

    return np.array(bits_list, dtype=np.uint8), np.array(ref_symbols, dtype=np.complex64)


def compute_evm(symbols: np.ndarray, ref_symbols: np.ndarray) -> float:
    """Calculate Error Vector Magnitude (EVM) in dB."""
    if len(symbols) == 0 or len(ref_symbols) == 0:
        return 0.0
    min_len = min(len(symbols), len(ref_symbols))
    s = symbols[:min_len]
    r = ref_symbols[:min_len]

    error = s - r
    p_error = np.mean(np.abs(error) ** 2)
    p_ref = np.mean(np.abs(r) ** 2)

    if p_ref <= 1e-12:
        return 0.0
    evm_rms = np.sqrt(p_error / p_ref)
    return float(20.0 * np.log10(max(evm_rms, 1e-6)))


def demodulate_signal(
    signal: np.ndarray,
    fs: float,
    modulation: str,
    center_freq: float = 0.0,
    baud_rate: float = None,
) -> dict:
    """Complete demodulation pipeline:
    1. Baseband downconversion & analytic conversion
    2. Symbol timing recovery
    3. Carrier phase synchronization (Costas Loop)
    4. Constellation normalization & slicing to bits
    5. EVM and bitstream formatting
    """
    if not np.iscomplexobj(signal):
        sig = hilbert(signal)
    else:
        sig = np.copy(signal)

    # 1. Baseband frequency shift
    if abs(center_freq) > 0.01:
        t = np.arange(len(sig)) / fs
        sig_bb = sig * np.exp(-1j * 2 * np.pi * center_freq * t)
    else:
        sig_bb = sig

    # 2. Timing recovery
    if baud_rate is not None and baud_rate > 0:
        symbols_raw = symbol_timing_recovery(sig_bb, fs, baud_rate)
    else:
        symbols_raw = sig_bb

    # Cap to reasonable number of symbols for performance & plotting
    max_symbols = 4096
    if len(symbols_raw) > max_symbols:
        symbols_raw = symbols_raw[:max_symbols]

    # 3. Carrier PLL / Phase Tracking
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
    evm_db = compute_evm(symbols_norm, ref_symbols)

    # Format bit string & hex string preview
    bit_str = "".join(str(b) for b in bits[:512])
    # Convert bits to bytes
    byte_array = np.packbits(bits)
    hex_str = byte_array[:64].tobytes().hex().upper()

    return {
        "symbols": symbols_norm,
        "bits": bits,
        "bit_string_preview": bit_str,
        "hex_preview": hex_str,
        "num_bits": len(bits),
        "evm_db": round(float(evm_db), 2),
        "modulation": modulation,
    }
