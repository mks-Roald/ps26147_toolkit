"""Unit test suite for all de-interleaving algorithms."""
import numpy as np
from ps26147_toolkit.deinterleaver import (
    block_deinterleave,
    convolutional_deinterleave,
    diagonal_deinterleave,
    pseudorandom_deinterleave,
    auto_detect_and_deinterleave,
    deinterleave,
)

def test_all():
    rng = np.random.default_rng(99)
    original = rng.integers(0, 2, size=256, dtype=np.uint8)

    # 1. Block round-trip
    R, C = 8, 16
    blk = original[: R * C].reshape((R, C), order="C").ravel(order="F")
    recovered_blk = block_deinterleave(blk, R, C)
    assert np.array_equal(recovered_blk, original[: R * C]), "Block de-interleaver failed"
    print("[PASS] Block De-interleaver")

    # 2. Convolutional round-trip
    N, D = 4, 4
    buffers_fwd = [[0] * (i * D) for i in range(N)]
    interleaved_conv = np.zeros_like(original)
    for k in range(len(original)):
        br = k % N
        buffers_fwd[br].append(int(original[k]))
        interleaved_conv[k] = buffers_fwd[br].pop(0)

    recovered_conv = convolutional_deinterleave(interleaved_conv, N, D)
    # Total delay is (N - 1) * D * N bits
    total_bit_delay = (N - 1) * D * N
    tail = len(original) - total_bit_delay
    assert np.array_equal(recovered_conv[total_bit_delay:], original[:tail]), "Conv de-interleaver failed"
    print("[PASS] Convolutional De-interleaver")

    # 3. Diagonal round-trip
    R2, C2 = 8, 8
    bs = R2 * C2
    data_diag = original[:bs]
    write_order = []
    for d in range(R2 + C2 - 1):
        for r in range(R2):
            c = d - r
            if 0 <= c < C2:
                write_order.append(r * C2 + c)
    perm_diag = np.array(write_order)
    interleaved_diag = np.empty(bs, dtype=np.uint8)
    interleaved_diag[np.arange(bs)] = data_diag[perm_diag]
    recovered_diag = diagonal_deinterleave(interleaved_diag, R2, C2)
    assert np.array_equal(recovered_diag, data_diag), "Diagonal de-interleaver failed"
    print("[PASS] Diagonal De-interleaver")

    # 4. Pseudo-Random round-trip
    seed = 42
    bs_pr = 64
    data_pr = original[:bs_pr]
    prng = np.random.default_rng(seed)
    perm_pr = prng.permutation(bs_pr)
    interleaved_pr = data_pr[perm_pr]
    recovered_pr = pseudorandom_deinterleave(interleaved_pr, bs_pr, seed)
    assert np.array_equal(recovered_pr, data_pr), "Pseudo-Random de-interleaver failed"
    print("[PASS] Pseudo-Random De-interleaver")

    # 5. Convenience wrapper test
    wrapped = deinterleave(blk, method="block", rows=R, cols=C)
    assert np.array_equal(wrapped["bits"], original[: R * C]), "Wrapper failed"
    print("[PASS] De-interleave Wrapper API")

    # 6. Auto-detect smoke test
    # Create structured pattern that is block interleaved
    pattern = np.array([1, 1, 1, 1, 0, 0, 0, 0] * 16, dtype=np.uint8)
    interleaved_pat = pattern.reshape((8, 16), order="C").ravel(order="F")
    res = auto_detect_and_deinterleave(interleaved_pat)
    print(f"[PASS] Auto-detect heuristic completed: Best Method = {res['method']}, Params = {res['params']}")

    print("\n All 4 De-interleaver algorithms passed verification tests.")

if __name__ == "__main__":
    test_all()
