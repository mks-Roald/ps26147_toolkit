"""Test spectrogram/waterfall data extraction in API routes."""

import numpy as np
import pytest
from api.routes.process import _process_signal_core
from api.schemas import WaterfallData, ProcessResponse


def test_process_signal_waterfall_data_structure():
    """Verify waterfall_data is populated with downsampled ~80x80 bins."""
    fs = 1_000_000.0
    num_samples = 4096
    t = np.arange(num_samples) / fs
    # Synthesize a simple chirp / multi-tone signal
    sig = (np.exp(1j * 2 * np.pi * 50_000 * t) + np.exp(1j * 2 * np.pi * (-100_000) * t)).astype(np.complex64)
    raw_bytes = sig.tobytes()

    res = _process_signal_core(raw_bytes, "synthetic_chirp.iq", fs)

    assert isinstance(res, ProcessResponse)
    assert res.waterfall_data is not None
    assert isinstance(res.waterfall_data, WaterfallData)

    # Verify downsampled bounds (<= 80x80)
    assert len(res.waterfall_data.time) <= 80
    assert len(res.waterfall_data.frequency) <= 80
    assert len(res.waterfall_data.power_db) == len(res.waterfall_data.frequency)
    assert len(res.waterfall_data.power_db[0]) == len(res.waterfall_data.time)

    # Verify types and validity
    assert all(isinstance(val, float) for val in res.waterfall_data.time)
    assert all(isinstance(val, float) for val in res.waterfall_data.frequency)
    assert all(isinstance(val, float) for row in res.waterfall_data.power_db for val in row)


def test_process_signal_waterfall_large_signal():
    """Verify large signal downsamples to exactly 80x80 bins."""
    fs = 2_000_000.0
    num_samples = 65536
    sig = (np.random.randn(num_samples) + 1j * np.random.randn(num_samples)).astype(np.complex64)
    raw_bytes = sig.tobytes()

    res = _process_signal_core(raw_bytes, "large_signal.iq", fs)

    assert res.waterfall_data is not None
    assert len(res.waterfall_data.time) == 80
    assert len(res.waterfall_data.frequency) == 80
    assert len(res.waterfall_data.power_db) == 80
    assert len(res.waterfall_data.power_db[0]) == 80
