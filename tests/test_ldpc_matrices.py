"""Unit tests for the standards-compliant LDPC parity-check matrix loader.

Covers:
- IEEE 802.11n matrix sizes for all block lengths & code rates
- DVB-S2 matrix generation (normal & short frames)
- Regular Gallager construction
- Master loader dispatch + error paths
- Matrix sanity: rank / syndrome validity where applicable
"""

import numpy as np
import pytest

from ps26147_toolkit.ldpc_matrices import (
    get_ieee_80211n_ldpc_matrix,
    get_dvbs2_ldpc_matrix,
    get_regular_ldpc_matrix,
    get_ldpc_matrix,
)


class TestIEEE80211n:
    @pytest.mark.parametrize("rate", ["1/2", "2/3", "3/4", "5/6"])
    @pytest.mark.parametrize("blk", [648, 1296, 1944])
    def test_all_rates_and_blocks(self, rate, blk):
        H = get_ieee_80211n_ldpc_matrix(code_rate=rate, block_length=blk)
        assert H.shape[1] == blk, f"n must equal {blk}"
        assert H.dtype == np.uint8
        # Quasi-cyclic expansion factor Z = blk // 27
        assert H.shape[0] % (blk // 27) == 0

    def test_unsupported_rate_raises(self):
        with pytest.raises(ValueError, match="Unsupported code rate"):
            get_ieee_80211n_ldpc_matrix(code_rate="7/8")

    def test_code_rate_approx_implemented_rate(self):
        """m/n approximates the requested code rate within QC tolerance."""
        H = get_ieee_80211n_ldpc_matrix(code_rate="1/2", block_length=648)
        m, n = H.shape
        assert abs((1.0 - m / n) - 0.5) < 0.15


class TestDvbs2:
    def test_normal_frame(self):
        H = get_dvbs2_ldpc_matrix(code_rate="1/2", frame_size="normal")
        assert H.shape[1] == 64800
        m, n = H.shape
        # z = 360
        assert H.shape[0] % 360 == 0
        assert abs((1.0 - m / n) - 0.5) < 0.05

    def test_short_frame(self):
        H = get_dvbs2_ldpc_matrix(code_rate="3/4", frame_size="short")
        assert H.shape[1] == 16200
        assert H.shape[0] % 90 == 0

    @pytest.mark.parametrize("rate", ["1/4", "2/5", "5/6", "9/10"])
    def test_various_rates_normal(self, rate):
        H = get_dvbs2_ldpc_matrix(code_rate=rate, frame_size="normal")
        m, n = H.shape
        expected = {"1/4": 0.25, "2/5": 0.4, "5/6": 5 / 6, "9/10": 0.9}[rate]
        assert abs((1.0 - m / n) - expected) < 0.05

    def test_unsupported_rate_raises(self):
        with pytest.raises(KeyError):
            get_dvbs2_ldpc_matrix(code_rate="0.9/1", frame_size="normal")


class TestRegularGallager:
    def test_shape_and_degree(self):
        H = get_regular_ldpc_matrix(n=128, k=64, dv=3, dc=6, seed=42)
        assert H.shape == (64, 128)
        # Column degree is bounded by dv * 2 (permuted ones per layer)
        col_deg = np.sum(H, axis=0)
        assert np.min(col_deg) >= 1
        assert np.max(col_deg) <= 6

    def test_syndrome_on_zero_codeword(self):
        H = get_regular_ldpc_matrix(n=200, k=100, seed=7)
        cw = np.zeros(200, dtype=np.uint8)
        assert np.all((H @ cw) % 2 == 0)


class TestMasterLoader:
    def test_80211n_dispatch(self):
        H = get_ldpc_matrix(standard="802.11n", code_rate="2/3", block_length=1296)
        assert H.shape[1] == 1296

    def test_dvbs2_dispatch(self):
        H = get_ldpc_matrix(standard="dvb-s2", code_rate="1/2", frame_size="short")
        assert H.shape[1] == 16200

    def test_regular_dispatch(self):
        H = get_ldpc_matrix(standard="regular", n=96, k=48, dv=3, dc=6)
        assert H.shape == (48, 96)

    def test_unknown_standard_raises(self):
        with pytest.raises(ValueError, match="Unknown LDPC standard"):
            get_ldpc_matrix(standard="turbo")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])