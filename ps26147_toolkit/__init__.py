__version__ = "0.2.0"

from .preprocess import (
    SignalMetadata,
    load_iq,
    load_wav,
    load_sigmf_meta,
    load_iq_with_sigmf,
    segment_signal,
)
from .filters import (
    remove_dc_offset,
    correct_iq_imbalance,
    bandpass_filter,
    spectral_denoise,
    median_filter_signal,
    clean_signal,
)
from .feature_extractor import compute_psd, compute_spectrogram, plot_spectrogram, plot_constellation
from .parameter_extractor import (
    estimate_center_frequency,
    estimate_bandwidth,
    estimate_bandwidth_all,
    estimate_snr,
    estimate_snr_m2m4,
    estimate_baud_rate,
    extract_signal_parameters,
)
from .classifier import ModulationClassifier, compute_cumulants, extract_features, rule_based_classify
from .demodulator import demodulate_signal, slice_symbols_to_bits, compute_evm
from .deinterleaver import (
    block_deinterleave,
    convolutional_deinterleave,
    diagonal_deinterleave,
    pseudorandom_deinterleave,
    auto_detect_and_deinterleave,
    deinterleave,
)
from .fec_decoders import (
    viterbi_decode,
    reed_solomon_decode,
    concatenated_decode,
    ldpc_decode,
    decode_fec,
)
from .correlator import (
    STANDARD_SYNC_WORDS,
    hex_to_bits,
    correlate_bitstream,
    find_sync_peaks,
    frame_synchronize,
    auto_discover_preamble,
)
