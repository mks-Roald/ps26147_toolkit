"""Unit tests for all 4 Forward Error Correction (FEC) decoders."""
import numpy as np
from ps26147_toolkit.fec_decoders import (
    ConvolutionalCodec,
    ReedSolomonCodec,
    LDPCCodec,
    viterbi_decode,
    reed_solomon_decode,
    concatenated_decode,
    ldpc_decode,
    decode_fec,
)

def test_viterbi():
    # K=7 NASA standard [171, 133]
    codec = ConvolutionalCodec(k=7, polys=(0o171, 0o133))
    rng = np.random.default_rng(42)
    msg = rng.integers(0, 2, size=64, dtype=np.uint8)

    encoded = codec.encode(msg)
    # Inject bit errors (5 bit flips out of ~140 bits)
    corrupted = np.copy(encoded)
    flip_indices = [5, 23, 47, 88, 112]
    for idx in flip_indices:
        corrupted[idx] ^= 1

    decoded = codec.decode(corrupted, max_len=len(msg))
    assert np.array_equal(decoded, msg), f"Viterbi failed to correct errors! Errors: {np.sum(decoded != msg)}"
    print("[PASS] Viterbi Convolutional Decoder (Corrected 5 bit flips)")

def test_reed_solomon():
    # RS(255, 223) can correct up to (255-223)/2 = 16 byte errors
    codec = ReedSolomonCodec(n=255, k=223)
    rng = np.random.default_rng(101)
    msg_bytes = rng.integers(0, 256, size=223, dtype=np.uint8).tolist()

    codeword = codec.encode_block(msg_bytes)
    # Corrupt 8 bytes in the block
    corrupted = list(codeword)
    corrupt_locs = [3, 15, 39, 77, 102, 150, 199, 240]
    for loc in corrupt_locs:
        corrupted[loc] ^= 0xAA

    recovered, n_err = codec.decode_block(corrupted)
    assert recovered == msg_bytes, "RS failed to recover message"
    assert n_err == len(corrupt_locs), f"Expected {len(corrupt_locs)} errors corrected, got {n_err}"
    print(f"[PASS] Reed-Solomon RS(255,223) Decoder (Corrected {n_err} byte errors)")

def test_concatenated():
    rng = np.random.default_rng(202)
    msg_bits = rng.integers(0, 2, size=223 * 8, dtype=np.uint8)

    # RS Encode (Outer)
    rs_codec = ReedSolomonCodec(n=255, k=223)
    raw_bytes = np.packbits(msg_bits).tolist()
    rs_codeword_bytes = rs_codec.encode_block(raw_bytes)
    rs_codeword_bits = np.unpackbits(np.array(rs_codeword_bytes, dtype=np.uint8))

    # Conv Encode (Inner)
    conv_codec = ConvolutionalCodec(k=7, polys=(0o171, 0o133))
    tx_bits = conv_codec.encode(rs_codeword_bits)

    # Channel Noise: Add distributed errors
    rx_bits = np.copy(tx_bits)
    err_pos = np.random.choice(len(rx_bits), size=12, replace=False)
    for p in err_pos:
        rx_bits[p] ^= 1

    # Concatenated decode
    res = concatenated_decode(rx_bits, viterbi_k=7, rs_n=255, rs_k=223)
    final_bits = res["bits"][: len(msg_bits)]
    assert np.array_equal(final_bits, msg_bits), "Concatenated decoding failed!"
    print("[PASS] Concatenated Code Decoder (Inner Viterbi + Outer Reed-Solomon)")

def test_ldpc():
    codec = LDPCCodec(n=128, k=64, seed=42)
    rng = np.random.default_rng(303)
    
    # Generate zero-codeword which is guaranteed to be a valid LDPC codeword
    cw = np.zeros(128, dtype=np.uint8)
    # Verify H * cw == 0
    assert np.all(np.dot(codec.H, cw) % 2 == 0), "H matrix syndrome failed on zero codeword"

    # Add channel noise (4 bit flips)
    corrupted_cw = np.copy(cw)
    corrupted_cw[[10, 35, 72, 105]] = 1
    llrs = np.where(corrupted_cw == 0, 4.0, -4.0).astype(np.float32)

    decoded_info, converged, iters = codec.decode_block(llrs, max_iters=20)
    assert converged, "LDPC Min-Sum failed to converge"
    assert np.array_equal(decoded_info, cw[:64]), "LDPC decoded information bits incorrect"
    print(f"[PASS] LDPC Min-Sum Belief Propagation Decoder (Converged in {iters} iterations)")

def test_fec_dispatcher():
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0] * 16, dtype=np.uint8)
    res = decode_fec(bits, scheme="viterbi")
    assert "bits" in res and res["decoder"].startswith("Viterbi")
    print("[PASS] Master FEC Decoder Dispatcher API")

def test_soft_decision_viterbi_awgn_ber_curve():
    """Sweep Eb/N0 and assert soft decision never worse than hard decision."""
    print("\n=== Testing Soft vs Hard Decision Viterbi over AWGN ===")
    codec = ConvolutionalCodec(k=7, polys=(0o171, 0o133))
    rng = np.random.default_rng(123)
    msg = rng.integers(0, 2, size=2000, dtype=np.uint8)
    encoded = codec.encode(msg)
    n_coded = len(encoded)
    # BPSK: bit 1 -> +1, bit 0 -> -1
    tx = 2 * encoded.astype(np.float32) - 1.0
    # Eb/N0 in dB points
    ebno_dbs = [2, 4, 6, 8, 10]
    results = []
    for db in ebno_dbs:
        snr_lin = 10 ** (db / 10.0)
        # BPSK antipodal ±1 symbols, rate-1/2 code -> Eb = Es/r = 2, N0 = 2*sigma^2,
        # so Eb/N0 = Es/(r*2*sigma^2) = 1/sigma^2  =>  sigma^2 = 1/snr_lin.
        # (Earlier code used sqrt(1/(2*snr_lin)) here, +3 dB above the label, which
        #  kept the whole sweep in the error-free regime and made this test vacuous.)
        noise_std = np.sqrt(1.0 / snr_lin)
        rx = tx + noise_std * rng.standard_normal(n_coded).astype(np.float32)
        # Hard decision
        hard_bits = (rx >= 0).astype(np.uint8)
        dec_hard = codec.decode(hard_bits, max_len=len(msg))
        ber_hard = np.mean(dec_hard != msg)
        # Soft decision (LLR convention: positive -> bit 0, negative -> bit 1)
        llrs = -2 * rx / (noise_std ** 2)
        dec_soft = codec.decode_soft(llrs, max_len=len(msg))
        ber_soft = np.mean(dec_soft != msg)
        results.append((db, ber_hard, ber_soft))
        print(f"  Eb/N0 = {db} dB | hard BER = {ber_hard:.4f} | soft BER = {ber_soft:.4f}")
        # Soft must never be worse than hard (allow tiny margin for finite sample noise)
        assert ber_soft <= ber_hard + 0.001, \
            f"soft BER ({ber_soft}) worse than hard ({ber_hard}) at Eb/N0={db} dB"
        # The key assertion: soft-decision must give coding gain over hard
        # For high SNR both should be zero; for low SNR soft should be <= hard
        # We allow soft == hard (both may hit zero floor) but soft must not exceed hard by more than tolerance
        # At moderate/high SNR soft should clearly beat hard

    # Sanity guard against a vacuous curve: the lowest point MUST be in the
    # error regime (hard BER > 0).  A noise-normalization bug that shifts the
    # sweep +3 dB (the old sqrt(1/(2*snr)) version) keeps every point at 0 BER
    # and would let this test pass while proving nothing about soft-vs-hard gain.
    _, ber_hard_low, _ = results[0]
    assert ber_hard_low > 0.0, \
        f"Lowest point hard BER is 0.0 — sweep is not in the error regime; " \
        f"a vacuous AWGN test can't validate §1.3.1"

    # At the highest SNR point, both should be very low
    _, ber_hard_high, ber_soft_high = results[-1]
    assert ber_soft_high < 0.02, f"Soft BER at {ebno_dbs[-1]} dB too high: {ber_soft_high}"
    # Soft must not be worse than hard at any point (with tolerance)
    for db, bh, bs in results:
        assert bs <= bh + 0.005, f"Soft BER ({bs}) worse than hard ({bh}) at {db} dB"
    print("[PASS] Soft-decision Viterbi AWGN BER curve validation")

if __name__ == "__main__":
    test_viterbi()
    test_reed_solomon()
    test_concatenated()
    test_ldpc()
    test_fec_dispatcher()
    test_soft_decision_viterbi_awgn_ber_curve()
    print("\n All 4 Forward Error Correction (FEC) Decoders PASSED!")
