"""Single source of truth for the experimental model's class and feature order."""

CLASS_NAMES = ("BPSK", "QPSK", "8PSK", "16QAM", "64QAM", "2FSK", "4FSK")
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}
INPUT_LENGTH = 4096
FEATURE_NAMES = (
    "abs_C20", "abs_C40", "abs_C41", "Re_C42", "abs_C60", "Re_C63",
    "gamma_max", "sigma_aa", "sigma_dp", "sigma_af", "spec_entropy",
    "kurtosis_env", "fsk_persistence", "qpsk_metric", "psk8_metric", "PAPR",
    "radial_q90_over_q50", "radial_q50_over_q10", "radial_moment4_rmsnorm",
    "radial_moment6_rmsnorm", "radial_hist_entropy_8",
    "radial_midband_fraction_075_125",
)
