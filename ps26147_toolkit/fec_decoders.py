"""Forward Error Correction (FEC) decoders for PS26147 toolkit.

Implements the four required decoding schemes:
  1. Viterbi Decoder (Convolutional codes: e.g., Rate 1/2, Constraint Length K=7 standard NASA/ESA polynomial [171, 133] octal)
  2. Reed-Solomon Decoder (GF(2^8) Berlekamp-Massey / Chien search / Forney algorithm)
  3. Concatenated Code Decoder (Inner Convolutional/Viterbi + Outer Reed-Solomon)
  4. LDPC Decoder (Belief Propagation / Log-Domain Min-Sum algorithm on Tanner graph)

All decoders operate on 1-D numpy arrays of bits (uint8) and return decoded data bits plus diagnostic metadata.
"""

from __future__ import annotations
import numpy as np
from typing import Literal, Tuple, Dict, Any, List


# ===========================================================================
# 1. Viterbi Decoder for Convolutional Codes
# ===========================================================================

class ConvolutionalCodec:
    """Standard Convolutional Encoder and Soft/Hard Viterbi Decoder.
    Default: NASA/ESA standard K=7, Rate 1/2, Polynomials G1=171 (0o171), G2=133 (0o133).
    """

    def __init__(self, k: int = 7, polys: tuple[int, int] = (0o171, 0o133)):
        self.k = k
        self.num_states = 1 << (k - 1)
        self.polys = polys
        self.rate_inv = len(polys)

        # Precompute state transitions and expected outputs
        # state is represented by (k-1) previous input bits
        self.next_state = np.zeros((self.num_states, 2), dtype=np.int32)
        self.outputs = np.zeros((self.num_states, 2, self.rate_inv), dtype=np.uint8)

        for state in range(self.num_states):
            for bit in (0, 1):
                # Combined register: new input bit is MSB, old state shifted right
                reg = (bit << (k - 1)) | state
                next_st = reg >> 1
                self.next_state[state, bit] = next_st

                for p_idx, poly in enumerate(polys):
                    # Parity of bitwise AND between register and generator polynomial
                    val = reg & poly
                    # Popcount mod 2
                    parity = bin(val).count("1") % 2
                    self.outputs[state, bit, p_idx] = parity

        # Precompute predecessor states for Viterbi trellis recursion
        # prev_states[next_state] = [(prev_state, input_bit, expected_output), ...]
        self.prev_transitions: list[list[tuple[int, int, np.ndarray]]] = [
            [] for _ in range(self.num_states)
        ]
        for st in range(self.num_states):
            for bit in (0, 1):
                nxt = self.next_state[st, bit]
                out = self.outputs[st, bit]
                self.prev_transitions[nxt].append((st, bit, out))

    def encode(self, bits: np.ndarray, flush: bool = True) -> np.ndarray:
        """Encode input bitstream with zero-tail flushing."""
        input_bits = list(bits)
        if flush:
            input_bits.extend([0] * (self.k - 1))

        encoded_bits = []
        state = 0
        for b in input_bits:
            out = self.outputs[state, b]
            encoded_bits.extend(out)
            state = self.next_state[state, b]

        return np.array(encoded_bits, dtype=np.uint8)

    def decode(self, rx_bits: np.ndarray, max_len: int = None) -> np.ndarray:
        """Hard-decision Viterbi trellis decoding (Hamming distance path metric)."""
        n_symbols = len(rx_bits) // self.rate_inv
        if n_symbols == 0:
            return np.array([], dtype=np.uint8)

        rx_symbols = rx_bits[: n_symbols * self.rate_inv].reshape((n_symbols, self.rate_inv))

        # Trellis path metrics and survivor history
        INF = 1e9
        path_metrics = np.full(self.num_states, INF, dtype=np.float32)
        path_metrics[0] = 0.0  # Initial state is 0

        # history[t][state] = (best_prev_state, input_bit)
        history: list[dict[int, tuple[int, int]]] = []

        for t in range(n_symbols):
            rx_sym = rx_symbols[t]
            new_metrics = np.full(self.num_states, INF, dtype=np.float32)
            step_history = {}

            for curr_state in range(self.num_states):
                best_metric = INF
                best_prev = 0
                best_bit = 0

                for prev_state, in_bit, exp_out in self.prev_transitions[curr_state]:
                    # Hamming distance branch metric
                    bm = int(np.sum(rx_sym != exp_out))
                    pm = path_metrics[prev_state] + bm

                    if pm < best_metric:
                        best_metric = pm
                        best_prev = prev_state
                        best_bit = in_bit

                new_metrics[curr_state] = best_metric
                step_history[curr_state] = (best_prev, best_bit)

            path_metrics = new_metrics
            history.append(step_history)

        # Traceback from minimum metric state (or state 0 if flushed)
        best_end_state = int(np.argmin(path_metrics))
        curr_state = best_end_state
        decoded = []

        for t in reversed(range(n_symbols)):
            prev_st, in_bit = history[t][curr_state]
            decoded.append(in_bit)
            curr_state = prev_st

        decoded.reverse()
        decoded_arr = np.array(decoded, dtype=np.uint8)

        # Discard zero-tail bits (k-1) if present
        if len(decoded_arr) > (self.k - 1):
            decoded_arr = decoded_arr[: -(self.k - 1)]

        if max_len is not None and len(decoded_arr) > max_len:
            decoded_arr = decoded_arr[:max_len]

        return decoded_arr


def viterbi_decode(
    bits: np.ndarray,
    constraint_length: int = 7,
    polys: tuple[int, int] = (0o171, 0o133),
) -> dict:
    """Viterbi Decoder convenience wrapper."""
    codec = ConvolutionalCodec(k=constraint_length, polys=polys)
    decoded = codec.decode(bits)
    return {
        "decoder": "Viterbi (Convolutional)",
        "rate": f"1/{len(polys)}",
        "constraint_length": constraint_length,
        "polynomials": [oct(p) for p in polys],
        "input_bits": len(bits),
        "output_bits": len(decoded),
        "bits": decoded,
    }


# ===========================================================================
# 2. Reed-Solomon (RS) Codec over GF(2^8)
# ===========================================================================

class GF256:
    """Galois Field GF(2^8) arithmetic using primitive polynomial 0x11D (x^8 + x^4 + x^3 + x^2 + 1)."""

    def __init__(self, prim_poly: int = 0x11D):
        self.exp = [0] * 512
        self.log = [0] * 256
        x = 1
        for i in range(255):
            self.exp[i] = x
            self.exp[i + 255] = x
            self.log[x] = i
            x <<= 1
            if x & 0x100:
                x ^= prim_poly

    def mul(self, a: int, b: int) -> int:
        if a == 0 or b == 0:
            return 0
        return self.exp[self.log[a] + self.log[b]]

    def div(self, a: int, b: int) -> int:
        if a == 0:
            return 0
        if b == 0:
            raise ZeroDivisionError("GF(256) division by zero")
        return self.exp[(self.log[a] - self.log[b]) % 255]

    def inv(self, a: int) -> int:
        if a == 0:
            raise ZeroDivisionError("GF(256) inverse of zero")
        return self.exp[255 - self.log[a]]

    def poly_mul(self, p: list[int], q: list[int]) -> list[int]:
        r = [0] * (len(p) + len(q) - 1)
        for i, a in enumerate(p):
            for j, b in enumerate(q):
                r[i + j] ^= self.mul(a, b)
        return r

    def poly_eval(self, p: list[int], x: int) -> int:
        """Evaluate polynomial using Horner's method."""
        val = 0
        for coef in p:
            val = self.mul(val, x) ^ coef
        return val


class ReedSolomonCodec:
    """Reed-Solomon (n, k) Codec over GF(2^8) with Berlekamp-Massey & Forney."""

    def __init__(self, n: int = 255, k: int = 223, prim_poly: int = 0x11D):
        self.n = n
        self.k = k
        self.two_t = n - k
        self.gf = GF256(prim_poly)

        # Generator polynomial: g(x) = (x - alpha^0)(x - alpha^1)...(x - alpha^(2t-1))
        self.gen = [1]
        for i in range(self.two_t):
            self.gen = self.gf.poly_mul(self.gen, [1, self.gf.exp[i]])

    def encode_block(self, msg: list[int]) -> list[int]:
        """Systematic RS encoding."""
        if len(msg) > self.k:
            msg = msg[: self.k]
        elif len(msg) < self.k:
            msg = msg + [0] * (self.k - len(msg))

        out = list(msg) + [0] * self.two_t
        for i in range(self.k):
            coef = out[i]
            if coef != 0:
                for j in range(1, len(self.gen)):
                    out[i + j] ^= self.gf.mul(self.gen[j], coef)
        return list(msg) + out[self.k :]

    def decode_block(self, r: list[int]) -> tuple[list[int], int]:
        """Berlekamp-Massey Syndrome decoding."""
        if len(r) != self.n:
            if len(r) < self.n:
                r = list(r) + [0] * (self.n - len(r))
            else:
                r = r[: self.n]

        # 1. Compute syndromes S_0 .. S_{2t-1}
        syn = [self.gf.poly_eval(r, self.gf.exp[i]) for i in range(self.two_t)]
        if max(syn) == 0:
            return r[: self.k], 0  # No errors detected

        # 2. Berlekamp-Massey algorithm to find error locator lambda(x)
        C = [1]
        B = [1]
        L = 0
        m = 1
        b = 1

        for i in range(self.two_t):
            # Discrepancy delta
            delta = syn[i]
            for j in range(1, L + 1):
                if j < len(C):
                    delta ^= self.gf.mul(C[j], syn[i - j])

            if delta == 0:
                m += 1
            else:
                T = list(C)
                scale = self.gf.div(delta, b)
                shifted_B = [0] * m + [self.gf.mul(coef, scale) for coef in B]

                # C = C + shifted_B
                new_len = max(len(C), len(shifted_B))
                C_padded = C + [0] * (new_len - len(C))
                B_padded = shifted_B + [0] * (new_len - len(shifted_B))
                C = [c ^ b_val for c, b_val in zip(C_padded, B_padded)]

                if 2 * L <= i:
                    L = i + 1 - L
                    B = T
                    b = delta
                    m = 1
                else:
                    m += 1

        # 3. Chien search: find roots of error locator polynomial Lambda(x)
        err_pos = []
        for pos in range(self.n):
            x_inv = self.gf.exp[(255 - (self.n - 1 - pos)) % 255]
            val = 0
            for deg, coef in enumerate(C):
                if coef != 0:
                    term = coef if deg == 0 else self.gf.mul(coef, self.gf.exp[(self.gf.log[x_inv] * deg) % 255])
                    val ^= term
            if val == 0:
                err_pos.append(pos)

        if len(err_pos) != L or L == 0:
            return r[: self.k], -1

        # 4. Error magnitude solver via linear syndrome equations:
        # Sum_{j=0}^{L-1} E_j * (X_j)^i = S_i
        X = [self.gf.exp[(self.n - 1 - p) % 255] for p in err_pos]
        M = [[self.gf.exp[(self.gf.log[X[j]] * i) % 255] for j in range(L)] for i in range(L)]
        b_vec = syn[:L]

        # Gaussian elimination in GF(2^8)
        for i in range(L):
            if M[i][i] == 0:
                for row in range(i + 1, L):
                    if M[row][i] != 0:
                        M[i], M[row] = M[row], M[i]
                        b_vec[i], b_vec[row] = b_vec[row], b_vec[i]
                        break
            if M[i][i] == 0:
                return r[: self.k], -1

            inv_p = self.gf.inv(M[i][i])
            for col in range(i, L):
                M[i][col] = self.gf.mul(M[i][col], inv_p)
            b_vec[i] = self.gf.mul(b_vec[i], inv_p)

            for row in range(L):
                if row != i and M[row][i] != 0:
                    factor = M[row][i]
                    for col in range(i, L):
                        M[row][col] ^= self.gf.mul(factor, M[i][col])
                    b_vec[row] ^= self.gf.mul(factor, b_vec[i])

        corrected = list(r)
        for idx, pos in enumerate(err_pos):
            corrected[pos] ^= b_vec[idx]

        return corrected[: self.k], len(err_pos)


def reed_solomon_decode(
    bits: np.ndarray,
    n: int = 255,
    k: int = 223,
) -> dict:
    """Reed-Solomon block decoder."""
    # Convert bitstream to bytes
    n_bytes = len(bits) // 8
    if n_bytes == 0:
        return {
            "decoder": "Reed-Solomon",
            "n": n, "k": k,
            "bits": np.array([], dtype=np.uint8),
            "errors_corrected": 0,
        }

    raw_bytes = np.packbits(bits[: n_bytes * 8]).tolist()
    codec = ReedSolomonCodec(n=n, k=k)

    decoded_bytes = []
    total_corrected = 0

    # Process block by block
    for i in range(0, len(raw_bytes), n):
        blk = raw_bytes[i : i + n]
        dec_blk, n_err = codec.decode_block(blk)
        if n_err > 0:
            total_corrected += n_err
        decoded_bytes.extend(dec_blk)

    decoded_bits = np.unpackbits(np.array(decoded_bytes, dtype=np.uint8))
    return {
        "decoder": "Reed-Solomon",
        "n": n,
        "k": k,
        "code_rate": f"{k}/{n}",
        "input_bits": len(bits),
        "output_bits": len(decoded_bits),
        "errors_corrected": total_corrected,
        "bits": decoded_bits,
    }


# ===========================================================================
# 3. Concatenated Code Decoder (Inner Viterbi + Outer Reed-Solomon)
# ===========================================================================

def concatenated_decode(
    bits: np.ndarray,
    viterbi_k: int = 7,
    viterbi_polys: tuple[int, int] = (0o171, 0o133),
    rs_n: int = 255,
    rs_k: int = 223,
) -> dict:
    """Standard DVB/CCSDS Concatenated FEC Decoder:
    Stage 1: Inner Viterbi Convolutional Decoder.
    Stage 2: Outer Reed-Solomon (255, 223) Algebraic Decoder.
    """
    # Stage 1: Viterbi decoding
    vit_res = viterbi_decode(bits, constraint_length=viterbi_k, polys=viterbi_polys)
    vit_bits = vit_res["bits"]

    # Stage 2: Outer Reed-Solomon decoding
    rs_res = reed_solomon_decode(vit_bits, n=rs_n, k=rs_k)

    return {
        "decoder": "Concatenated (Inner Viterbi + Outer Reed-Solomon)",
        "inner_scheme": f"Convolutional (K={viterbi_k}, Rate 1/{len(viterbi_polys)})",
        "outer_scheme": f"Reed-Solomon RS({rs_n}, {rs_k})",
        "input_bits": len(bits),
        "inner_output_bits": len(vit_bits),
        "final_output_bits": len(rs_res["bits"]),
        "rs_errors_corrected": rs_res["errors_corrected"],
        "bits": rs_res["bits"],
    }


# ===========================================================================
# 4. LDPC Decoder (Log-Domain Min-Sum Algorithm)
# ===========================================================================

class LDPCCodec:
    """Low-Density Parity-Check (LDPC) Codec with Log-Domain Min-Sum Message Passing."""

    def __init__(self, n: int = 128, k: int = 64, dv: int = 3, dc: int = 6, seed: int = 42):
        self.n = n
        self.k = k
        self.m = n - k
        self.dv = dv
        self.dc = dc

        # Build regular Gallager LDPC parity check matrix H (m x n)
        rng = np.random.default_rng(seed)
        H = np.zeros((self.m, self.n), dtype=np.uint8)

        # Place dc ones in each row, dv ones in each column
        sub_m = max(1, self.m // dv)
        for i in range(dv):
            perm = rng.permutation(self.n)
            for j in range(sub_m):
                row_idx = i * sub_m + j
                if row_idx < self.m:
                    ones_idx = np.arange(j * dc, (j + 1) * dc) % self.n
                    H[row_idx, perm[ones_idx]] = 1

        # Fill any remaining rows
        for r in range(dv * sub_m, self.m):
            ones_idx = rng.choice(self.n, size=min(dc, self.n), replace=False)
            H[r, ones_idx] = 1

        self.H = H
        self.check_adj = [np.where(self.H[c, :] == 1)[0] for c in range(self.m)]
        self.var_adj = [np.where(self.H[:, v] == 1)[0] for v in range(self.n)]

    def decode_block(
        self,
        llr: np.ndarray,
        max_iters: int = 25,
        alpha: float = 0.8,
    ) -> tuple[np.ndarray, bool, int]:
        """Normalized Min-Sum Algorithm for LDPC Decoding.

        Parameters
        ----------
        llr : np.ndarray
            Channel Log-Likelihood Ratios for n variable nodes. (Positive -> bit 0, Negative -> bit 1).
        max_iters : int
            Maximum belief propagation iterations.
        alpha : float
            Min-sum normalization scaling factor (default: 0.8).
        """
        # Variable-to-Check messages (m x n)
        v2c = np.zeros((self.m, self.n), dtype=np.float32)
        # Initialize with channel LLRs
        for v in range(self.n):
            for c in self.var_adj[v]:
                v2c[c, v] = llr[v]

        c2v = np.zeros((self.m, self.n), dtype=np.float32)

        for iteration in range(max_iters):
            # 1. Check Node Update (Min-Sum approximation)
            for c in range(self.m):
                var_nodes = self.check_adj[c]
                msgs = v2c[c, var_nodes]
                signs = np.sign(msgs)
                signs[signs == 0] = 1
                mags = np.abs(msgs)

                # Total parity sign
                prod_sign = np.prod(signs)

                for idx, v in enumerate(var_nodes):
                    # Exclude current variable
                    other_mags = np.delete(mags, idx)
                    min_val = np.min(other_mags) if len(other_mags) > 0 else 0.0
                    c2v[c, v] = alpha * (prod_sign * signs[idx]) * min_val

            # 2. Variable Node Update & Marginal LLR calculation
            total_llr = np.copy(llr)
            for v in range(self.n):
                total_llr[v] += np.sum(c2v[self.var_adj[v], v])

            # Hard decision candidate codeword
            est_bits = (total_llr < 0).astype(np.uint8)

            # 3. Syndrome check: H * x^T == 0 (mod 2)
            syndrome = np.dot(self.H, est_bits) % 2
            if np.all(syndrome == 0):
                return est_bits[: self.k], True, iteration + 1

            # Update Variable-to-Check messages for next iteration
            for v in range(self.n):
                for c in self.var_adj[v]:
                    v2c[c, v] = total_llr[v] - c2v[c, v]

        est_bits = (total_llr < 0).astype(np.uint8)
        return est_bits[: self.k], False, max_iters


def ldpc_decode(
    bits: np.ndarray,
    n: int = 128,
    k: int = 64,
    max_iters: int = 25,
) -> dict:
    """LDPC Block Decoder wrapper."""
    codec = LDPCCodec(n=n, k=k)
    # Convert hard bits (0/1) to pseudo-channel LLRs (+6.0 for 0, -6.0 for 1)
    llrs = np.where(bits == 0, 6.0, -6.0).astype(np.float32)

    decoded_bits = []
    blocks_converged = 0
    num_blocks = len(bits) // n

    for i in range(num_blocks):
        blk_llr = llrs[i * n : (i + 1) * n]
        dec_k, converged, _ = codec.decode_block(blk_llr, max_iters=max_iters)
        if converged:
            blocks_converged += 1
        decoded_bits.extend(dec_k)

    # Handle remaining partial bits
    if len(bits) % n != 0 and num_blocks == 0:
        pad_len = n - len(bits)
        padded_llr = np.pad(llrs, (0, pad_len), constant_values=0.0)
        dec_k, converged, _ = codec.decode_block(padded_llr, max_iters=max_iters)
        if converged:
            blocks_converged += 1
        decoded_bits.extend(dec_k)

    out_arr = np.array(decoded_bits, dtype=np.uint8)
    return {
        "decoder": "LDPC (Min-Sum)",
        "block_size_n": n,
        "info_bits_k": k,
        "code_rate": f"{k}/{n}",
        "input_bits": len(bits),
        "output_bits": len(out_arr),
        "blocks_converged": blocks_converged,
        "total_blocks": max(1, num_blocks),
        "bits": out_arr,
    }


# ===========================================================================
# 5. Master FEC Decoder Dispatcher & Auto-Detection
# ===========================================================================

FECScheme = Literal["viterbi", "reed-solomon", "concatenated", "ldpc", "none"]


def decode_fec(
    bits: np.ndarray,
    scheme: FECScheme = "viterbi",
    **kwargs: Any,
) -> dict:
    """Dispatch decoding to the requested Forward Error Correction (FEC) scheme."""
    if len(bits) == 0 or scheme == "none":
        return {
            "decoder": "None (Raw Pass-Through)",
            "input_bits": len(bits),
            "output_bits": len(bits),
            "bits": bits.copy(),
        }

    s = scheme.lower().replace("_", "-")
    if "vit" in s or "conv" in s:
        k_val = kwargs.get("constraint_length", 7)
        return viterbi_decode(bits, constraint_length=k_val)
    elif "rs" in s or "reed" in s or "solomon" in s:
        n_val = kwargs.get("n", 255)
        k_val = kwargs.get("k", 223)
        return reed_solomon_decode(bits, n=n_val, k=k_val)
    elif "concat" in s:
        return concatenated_decode(bits)
    elif "ldpc" in s:
        n_val = kwargs.get("n", 128)
        k_val = kwargs.get("k", 64)
        return ldpc_decode(bits, n=n_val, k=k_val)
    else:
        return {
            "decoder": "Raw",
            "input_bits": len(bits),
            "output_bits": len(bits),
            "bits": bits.copy(),
        }
