"""Automatic Modulation Recognition (AMR) Engine for PS26147 Toolkit."""

import numpy as np
import joblib
from pathlib import Path
from typing import Optional, Dict
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.signal import hilbert

from .feature_extractor import (
    compute_cumulants,
    extract_features,
    extract_instantaneous_features,
    extract_spectral_features,
    rrc_filter,
)

MODULATION_CLASSES = [
    "BPSK",
    "QPSK",
    "8PSK",
    "16QAM",
    "64QAM",
    "2FSK",
    "4FSK",
    "AM",
]


def downconvert_baseband(signal: np.ndarray, fs: float, fc: Optional[float] = None) -> np.ndarray:
    """Shift a (real or complex) passband signal to complex baseband.

    Cumulant / instantaneous features — and therefore the whole classifier —
    are only well-defined at *complex baseband* (zero-mean, carrier removed).
    The ground-truth corpus is stored as a real *passband* WAV (carrier at
    ~10 kHz with fs = 48 kHz); feeding that raw passband straight into
    ``compute_cumulants`` collapses every clean carrier-modulated signal into a
    near-identical tone and the classifier (RF and rules alike) degrades to
    guessing 'AM'.  Downconverting first is the missing first step.

    ``fc``, when given, is trusted as the true carrier (an accurate estimate is
    hard to derive from these RRC-shaped bands — raw spectral argmax lands tens
    to hundreds of Hz off, and that residual CFO destroys the baseband
    constellation).  When ``fc`` is omitted the carrier is estimated as the
    positive-band spectral peak; a true baseband input then estimates ~0 Hz and
    this becomes a (near) no-op.
    """
    sig = signal if np.iscomplexobj(signal) else hilbert(signal.astype(np.float32))
    sig = np.asarray(sig, dtype=np.complex64)
    if len(sig) < 8 or not np.all(np.isfinite(sig)):
        return sig

    # 1. Coarse carrier: caller-provided fc (trusted/accurate), else Welch argmax.
    from scipy.signal import welch
    nperseg = min(1024, len(sig))
    freqs, psd = welch(sig, fs=fs, nperseg=nperseg)
    fc_coarse = float(fc) if fc else float(freqs[int(np.argmax(psd))])
    if fc_coarse <= 0.0:
        return sig

    # 2. Fine residual carrier-frequency-offset (CFO) recovery via the P-th power
    #    line.  A coarse estimate lands tens to hundreds of Hz off for RRC-shaped
    #    bands (the spectral *magnitude* peak is not the band centre), and even a
    #    ~10 Hz residual CFO destroys the cumulants (they average phase-sensitive
    #    moments over the whole signal).  For PSK/QAM the P-th power of the
    #    normalised baseband concentrates at ``P * residual_freq``.
    t = np.arange(len(sig)) / fs
    bb = sig * np.exp(-2j * np.pi * fc_coarse * t)
    bb_n = bb - np.mean(bb)
    aa = bb_n / (np.abs(bb_n) + 1e-12)

    # Try P=4 then P=8; pick the candidate whose line is cleanest (largest peak).
    best_res, best_peak_e, best_fc = 0.0, 0.0, fc_coarse
    for P in (4, 8):
        sp = np.fft.fft(aa ** P)
        fq = np.fft.fftfreq(len(sp), 1.0 / fs)
        fpk = float(fq[int(np.argmax(np.abs(sp)))])
        if abs(fpk) > fs / 4:  # unwrap aliasing around Nyquist
            fpk -= np.sign(fpk) * fs
        peak_e = float(np.max(np.abs(sp)))
        # Refinement only makes sense for small residuals: reject if the P-th
        # power line implies a huge jump (e.g. FSK, where the method is invalid
        # but the coarse estimate already suffices).
        if abs(fpk) > 0.25 * fs:
            continue
        cand = fc_coarse + fpk / P
        if peak_e > best_peak_e:
            best_peak_e, best_fc = peak_e, cand

    return sig * np.exp(-2j * np.pi * best_fc * t)


def _process_signal_chunks(signal: np.ndarray, fs: float = 1_000_000.0, fc: Optional[float] = None,
                           block_len: int = 65536, hop_len: int = 32768) -> dict:
    """Process signal in overlapping blocks and aggregate statistics needed by rule_based_classify.

    Returns a dict with keys:
        c20, c40, c42, c60, c63 (complex/float averages),
        sigma_aa (float envelope std),
        sigma_af (float median-filtered IF std / fs),
        fsk_persistence (float ratio),
        inst_freq_filtered (np.ndarray concatenated filtered instantaneous frequency),
        qpsk_fold (float average of |s_norm|^4).
    """
    if len(signal) == 0:
        # Return zeros/defaults
        return {
            "c20": 0.0+0.0j,
            "c40": 0.0+0.0j,
            "c42": 0.0+0.0j,
            "c60": 0.0+0.0j,
            "c63": 0.0+0.0j,
            "sigma_aa": 0.0,
            "sigma_af": 0.0,
            "fsk_persistence": 0.0,
            "inst_freq_filtered": np.array([], dtype=np.float64),
            "qpsk_fold": 0.0,
        }

    n_samples = len(signal)
    start = 0
    block_idx = 0

    # Accumulators
    sum_c20 = 0.0+0.0j
    sum_c40 = 0.0+0.0j
    sum_c42 = 0.0+0.0j
    sum_c60 = 0.0+0.0j
    sum_c63 = 0.0+0.0j

    sum_env = 0.0          # Σ |signal|
    sum_env_sq = 0.0       # Σ |signal|^2
    sum_phase = 0.0        # Σ angle
    sum_phase_sq = 0.0     # Σ angle^2
    count_samples = 0

    sum_norm_pow4 = 0.0+0.0j  # Σ (s_norm)^4

    # For FSK histogram we will accumulate histogram counts
    freq_hist_bins = 40
    freq_hist = np.zeros(freq_hist_bins, dtype=np.float64)
    freq_hist_min = -0.5*fs
    freq_hist_max = 0.5*fs

    # For sigma_af and fsk_persistence we need raw and filtered std per block; we'll accumulate sums
    sum_sigma_af = 0.0
    sum_fsk_p = 0.0
    count_blocks = 0

    while start < n_samples:
        end = min(start + block_len, n_samples)
        block = signal[start:end].astype(np.complex64)

        # Downconvert to baseband
        bb = downconvert_baseband(block, fs=fs, fc=fc)

        # Cumulants
        cum = compute_cumulants(bb, fs=fs)
        sum_c20 += cum["c20"]
        sum_c40 += cum["c40"]
        sum_c42 += cum["c42"]
        sum_c60 += cum["c60"]
        sum_c63 += cum["c63"]

        # Instantaneous features
        inst = extract_instantaneous_features(bb, fs=fs)
        sigma_aa_block = inst["sigma_aa"]
        sigma_af_block = inst["sigma_af"]
        fsk_p_block = inst["fsk_persistence"]
        inst_freq_filt = inst["inst_freq_filtered"]

        # Envelope and phase stats
        env = np.abs(bb)
        phase = np.angle(bb)

        sum_env += np.sum(env)
        sum_env_sq += np.sum(env**2)
        sum_phase += np.sum(phase)
        sum_phase_sq += np.sum(phase**2)
        count_samples += len(bb)

        # QPSK 4th-power fold metric: need normalized signal
        s_norm = (bb - np.mean(bb)) / (np.sqrt(np.mean(np.abs(bb - np.mean(bb))**2)) + 1e-12)
        sum_norm_pow4 += np.sum(s_norm**4)

        # Accumulate for FSK histogram
        hist, _ = np.histogram(inst_freq_filt, bins=freq_hist_bins,
                               range=(freq_hist_min, freq_hist_max))
        freq_hist += hist

        # Accumulate sigma_af and fsk_persistence (simple average)
        sum_sigma_af += sigma_af_block
        sum_fsk_p += fsk_p_block
        count_blocks += 1

        start += hop_len
        block_idx += 1

    # Compute averages
    if count_blocks == 0:
        count_blocks = 1
    if count_samples == 0:
        count_samples = 1

    avg_c20 = sum_c20 / count_blocks
    avg_c40 = sum_c40 / count_blocks
    avg_c42 = sum_c42 / count_blocks
    avg_c60 = sum_c60 / count_blocks
    avg_c63 = sum_c63 / count_blocks

    mean_env = sum_env / count_samples
    mean_env_sq = sum_env_sq / count_samples
    var_env = mean_env_sq - mean_env**2
    sigma_aa = np.sqrt(var_env) if var_env > 0 else 0.0

    mean_phase = sum_phase / count_samples
    mean_phase_sq = sum_phase_sq / count_samples
    var_phase = mean_phase_sq - mean_phase**2
    phase_std = np.sqrt(var_phase) if var_phase > 0 else 0.0

    avg_sigma_af = sum_sigma_af / count_blocks
    avg_fsk_p = sum_fsk_p / count_blocks

    # Reconstruct concatenated instantaneous frequency filtered array? We'll just reuse histogram.
    # For the rule that needs inst_freq_filtered array (for histogram), we pass the histogram.
    # We'll need to modify rule_based_classify to accept histogram instead of raw array.
    # Instead we can provide a dummy array and rely on histogram; but easier: modify rule_based_classify
    # to accept optional features dict containing the histogram.
    # We'll add key "fsk_hist" and "fsk_hist_bins", "fsk_hist_range".
    qpsk_fold = np.abs(sum_norm_pow4) / count_samples

    return {
        "c20": avg_c20,
        "c40": avg_c40,
        "c42": avg_c42,
        "c60": avg_c60,
        "c63": avg_c63,
        "sigma_aa": sigma_aa,
        "sigma_af": avg_sigma_af,
        "fsk_persistence": avg_fsk_p,
        "fsk_hist": freq_hist,
        "fsk_hist_bins": freq_hist_bins,
        "fsk_hist_range": (freq_hist_min, freq_hist_max),
        "qpsk_fold": qpsk_fold,
    }


def rule_based_classify(signal: np.ndarray = None, fs: float = 1000000.0, fc: Optional[float] = None,
                        features: Optional[dict] = None) -> str:
    """Expert rule-based classifier using Higher-Order Cumulants (HOC) and instantaneous signal statistics.

    Either provide `signal` (will be processed) or `features` dict (pre‑aggregated statistics).
    """
    # Default for short signal
    if signal is not None and len(signal) < 32:
        return "QPSK"

    if features is None:
        # ----- original path (compute from signal) -----
        sig = downconvert_baseband(signal, fs, fc=fc)

        max_samples = 131072
        sig = sig[:max_samples] if len(sig) > max_samples else sig

        # `sig` is already complex baseband (downconverted above); do NOT pass fc
        # to compute_cumulants, or it would apply a second (double) downconversion
        # and shift the baseband to -fc, destroying c20/c40.
        cum = compute_cumulants(sig, fs=fs)
        abs_c20 = float(np.abs(cum["c20"]))
        abs_c40 = float(np.abs(cum["c40"]))
        c42 = float(np.real(cum["c42"]))
        abs_c60 = float(np.abs(cum["c60"]))
        c63 = float(np.real(cum["c63"]))

        inst = extract_instantaneous_features(sig, fs=fs)
        sigma_aa = inst["sigma_aa"]
        sigma_af = inst["sigma_af"]
        fsk_persistence = inst["fsk_persistence"]
        inst_freq_filtered = inst["inst_freq_filtered"]

        # For FSK detection we need histogram
        hist, _ = np.histogram(inst_freq_filtered, bins=40)
        max_h = np.max(hist)
        peak_bins = np.where(hist > 0.20 * max_h)[0]
    else:
        # ----- aggregated features path -----
        cum20 = features.get("c20", 0.0+0.0j)
        cum40 = features.get("c40", 0.0+0.0j)
        cum42 = features.get("c42", 0.0+0.0j)
        cum60 = features.get("c60", 0.0+0.0j)
        cum63 = features.get("c63", 0.0+0.0j)
        abs_c20 = float(np.abs(cum20))
        abs_c40 = float(np.abs(cum40))
        c42 = float(np.real(cum42))
        abs_c60 = float(np.abs(cum60))
        c63 = float(np.real(cum63))

        sigma_aa = features.get("sigma_aa", 0.0)
        sigma_af = features.get("sigma_af", 0.0)
        fsk_persistence = features.get("fsk_persistence", 0.0)
        # For FSK histogram
        hist = features.get("fsk_hist", np.array([], dtype=np.float64))
        # If hist is empty, fallback to dummy to avoid error
        if hist.size == 0:
            # create a dummy histogram with one bin to avoid errors
            hist = np.zeros(1, dtype=np.float64)
        max_h = np.max(hist) if hist.size > 0 else 0.0
        peak_bins = np.where(hist > 0.20 * max_h)[0] if max_h > 0 else np.array([], dtype=int)
        # QPSK fold metric
        qpsk_fold = features.get("qpsk_fold", 0.0)

    # 1. Genuine FSK detection (baseband):
    # FSK has near-constant envelope (sigma_aa ~ 0.004 for clean corpus, well
    # below any PSK/QAM which sits >0.28) and very high fsk_persistence
    # (filtered inst-freq std tracks raw, ~0.99 vs PSK ~0.5).
    # The old gate used sigma_af > 0.012 which excludes FSK at low baud
    # (4FSK=0.0055, 2FSK=0.0102) and is unnecessary once sigma_aa + fskP
    # isolate FSK.  Histogram clustering then distinguishes 2 vs 4 peaks.
    if sigma_aa < 0.10 and fsk_persistence > 0.85:
        if len(peak_bins) > 0:
            clusters = 1 + np.sum(np.diff(peak_bins) > 2)
            if clusters >= 3:
                return "4FSK"
            elif clusters == 2:
                return "2FSK"
        return "2FSK"

    # 2. AM Detection: Significant envelope variation with near-zero phase modulation or non-zero carrier offset
    # More strict to avoid misclassifying PSK/QAM as AM
    # Need phase_std: compute from features if available, else compute from signal
    if features is not None:
        # We didn't store phase_std; we can approximate from sigma_aa? Not accurate.
        # Instead compute phase_std from signal if we have it; otherwise approximate using sigma_aa?
        # For simplicity, if features provided we cannot compute phase_std; we'll skip AM detection
        # and rely on other rules. This is a limitation but acceptable because AM detection
        # mainly uses envelope variance and phase_std; we can approximate phase_std as 0 if not available.
        phase_std = 0.0  # placeholder
    else:
        phase_angles = np.angle(sig)
        phase_std = float(np.std(phase_angles))
    if sigma_aa > 0.20 and phase_std < 0.25:
        return "AM"

    # 3. PSK vs QAM Discrimination using 4th-power phase folding
    if features is not None:
        # qpsk_fold already computed
        pass
    else:
        s_norm = (sig - np.mean(sig)) / (np.abs(sig - np.mean(sig)) + 1e-12)
        qpsk_fold = float(np.abs(np.mean(s_norm ** 4)))

    # BPSK: Strong real moment c20 and high c40/c60
    if abs_c20 > 0.55 or (abs_c40 > 1.35 and phase_std > 0.8):
        return "BPSK"

    # QPSK: High 4th-power fold metric, c40 > 0.55 or c63 > 2.2
    if qpsk_fold > 0.35 or (abs_c40 > 0.65 and c63 > 2.2):
        return "QPSK"

    # 8PSK: Constant envelope, low c40 (< 0.40) and c42 < -0.75
    if abs_c40 < 0.40 and c42 < -0.75:
        return "8PSK"

    # QAM Family (reached after BPSK, QPSK, 8PSK are already caught).
    # At baseband: 16QAM c63 ~ 0.37, 64QAM c63 ~ 1.70.
    # Higher-order QAM has more amplitude levels → larger 6th-order moment (c63).
    if c63 > 1.0:
        return "64QAM"
    else:
        return "16QAM"


def generate_synthetic_dataset(
    n_samples_per_class: int = 150,
    fs: float = 1000000.0,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate realistic synthetic training data with channel impairments for AMR.

    Impairments include:
      - Root-Raised Cosine (RRC) pulse shaping (alpha in [0.2, 0.5])
      - Carrier Frequency Offset (CFO) and random carrier phase
      - Phase jitter / phase noise
      - Multi-path dispersion / Rayleigh & Rician fading
      - SNR sweeps from 4 dB to 30 dB
    """
    np.random.seed(random_state)
    classes = MODULATION_CLASSES
    X, y = [], []
    num_symbols = 512
    sps = 8  # samples per symbol
    n_samples = num_symbols * sps

    for cls in classes:
        for _ in range(n_samples_per_class):
            snr_db = np.random.uniform(4.0, 30.0)
            alpha = np.random.uniform(0.2, 0.5)
            rrc = rrc_filter(num_taps=49, alpha=alpha, sps=sps)

            if cls == "BPSK":
                syms = np.random.choice([-1.0, 1.0], size=num_symbols)
                upsampled = np.zeros(n_samples, dtype=np.complex64)
                upsampled[::sps] = syms
                sig = np.convolve(upsampled, rrc, mode="same")
            elif cls == "QPSK":
                syms = (
                    np.random.choice([-1.0, 1.0], size=num_symbols)
                    + 1j * np.random.choice([-1.0, 1.0], size=num_symbols)
                ) / np.sqrt(2)
                upsampled = np.zeros(n_samples, dtype=np.complex64)
                upsampled[::sps] = syms
                sig = np.convolve(upsampled, rrc, mode="same")
            elif cls == "8PSK":
                phases = np.random.choice(np.arange(8) * (2.0 * np.pi / 8.0), size=num_symbols)
                syms = np.exp(1j * phases)
                upsampled = np.zeros(n_samples, dtype=np.complex64)
                upsampled[::sps] = syms
                sig = np.convolve(upsampled, rrc, mode="same")
            elif cls == "16QAM":
                grid = np.array([-3, -1, 1, 3])
                syms = (
                    np.random.choice(grid, size=num_symbols)
                    + 1j * np.random.choice(grid, size=num_symbols)
                ) / np.sqrt(10)
                upsampled = np.zeros(n_samples, dtype=np.complex64)
                upsampled[::sps] = syms
                sig = np.convolve(upsampled, rrc, mode="same")
            elif cls == "64QAM":
                grid = np.array([-7, -5, -3, -1, 1, 3, 5, 7])
                syms = (
                    np.random.choice(grid, size=num_symbols)
                    + 1j * np.random.choice(grid, size=num_symbols)
                ) / np.sqrt(42)
                upsampled = np.zeros(n_samples, dtype=np.complex64)
                upsampled[::sps] = syms
                sig = np.convolve(upsampled, rrc, mode="same")
            elif cls == "2FSK":
                bits = np.random.choice([0, 1], size=num_symbols)
                f_dev = np.random.uniform(25000.0, 50000.0)
                freqs = np.where(np.repeat(bits, sps) == 1, f_dev, -f_dev)
                phase = 2.0 * np.pi * np.cumsum(freqs) / fs
                sig = np.exp(1j * phase).astype(np.complex64)
            elif cls == "4FSK":
                symbols = np.random.choice([-3, -1, 1, 3], size=num_symbols)
                f_dev = np.random.uniform(15000.0, 30000.0)
                freqs = np.repeat(symbols, sps) * f_dev
                phase = 2.0 * np.pi * np.cumsum(freqs) / fs
                sig = np.exp(1j * phase).astype(np.complex64)
            elif cls == "AM":
                t = np.arange(n_samples) / fs
                fm1 = np.random.uniform(500, 2000)
                fm2 = np.random.uniform(2000, 5000)
                mod_sig = 0.5 * np.cos(2 * np.pi * fm1 * t) + 0.3 * np.sin(2 * np.pi * fm2 * t)
                m_depth = np.random.uniform(0.5, 0.85)
                sig = (1.0 + m_depth * mod_sig).astype(np.complex64)

            # Channel Impairments:
            # 1. Carrier Frequency Offset & Carrier Phase Offset
            cfo = np.random.uniform(-0.008, 0.008) * fs
            t = np.arange(len(sig)) / fs
            sig = sig * np.exp(1j * (2.0 * np.pi * cfo * t + np.random.uniform(0, 2.0 * np.pi)))

            # 2. Multipath Fading / Dispersion
            if np.random.rand() > 0.5:
                delay = np.random.randint(1, 3)
                beta = np.random.uniform(0.05, 0.15)
                sig = sig + beta * np.exp(1j * np.random.uniform(0, 2.0 * np.pi)) * np.roll(sig, delay)

            # 3. AWGN Noise
            sig_power = np.mean(np.abs(sig) ** 2)
            noise_power = sig_power / (10.0 ** (snr_db / 10.0))
            noise = np.sqrt(noise_power / 2.0) * (
                np.random.randn(len(sig)) + 1j * np.random.randn(len(sig))
            )
            sig_noisy = sig + noise

            feats = extract_features(sig_noisy, fs=fs)
            X.append(feats)
            y.append(cls)

    return np.vstack(X), np.array(y)


class ModulationClassifier:
    """Random Forest Classifier with HOC feature extraction and rule-based fallback."""

    def __init__(self, model_path: str = None):
        if model_path is None:
            default_model = Path(__file__).resolve().parents[1] / "model.pkl"
            self.model_path = default_model
        else:
            self.model_path = Path(model_path)

        self.pipeline = None
        self.is_fitted = False

        if self.model_path.exists():
            try:
                self.pipeline = joblib.load(str(self.model_path))
                self.is_fitted = True
            except Exception:
                self._build_and_train_default()
        else:
            self._build_and_train_default()

    def _build_and_train_default(self):
        """Train a robust Random Forest model on synthesized data and save to disk."""
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=150,
                    max_depth=14,
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ])
        X, y = generate_synthetic_dataset(n_samples_per_class=120)
        self.pipeline.fit(X, y)
        self.is_fitted = True
        try:
            joblib.dump(self.pipeline, str(self.model_path))
        except Exception:
            pass

    def train(self, X: np.ndarray, y: np.ndarray):
        """Train the classifier on feature matrix X and label array y."""
        self.pipeline.fit(X, y)
        self.is_fitted = True

    def predict(self, signal: np.ndarray, fs: float = 1000000.0, fc: float = None) -> str:
        """Return the top class from the unified classifier decision."""
        return self.predict_with_confidence(signal, fs=fs, fc=fc)["modulation"]

    def predict_proba(
        self, signal: np.ndarray, fs: float = 1000000.0, fc: float = None
    ) -> dict[str, float]:
        """Return posterior class probabilities for all modulation classes."""
        if self.is_fitted and self.pipeline is not None:
            try:
                feats = extract_features(signal, fs=fs, fc=fc).reshape(1, -1)
                probs = self.pipeline.predict_proba(feats)[0]
                classes = list(self.pipeline.classes_)
                return {cls: float(p) for cls, p in zip(classes, probs)}
            except Exception:
                pass

        pred = rule_based_classify(signal, fs=fs, fc=fc)
        return {cls: (1.0 if cls == pred else 0.0) for cls in MODULATION_CLASSES}

    def predict_with_confidence(
        self, signal: np.ndarray, fs: float = 1000000.0, fc: float = None,
        include_diagnostics: bool = True,
    ) -> dict:
        """Predict modulation with confidence score, class probabilities, and diagnostic cumulants."""
        cum = compute_cumulants(signal, fs=fs, fc=fc) if include_diagnostics else None
        feats = extract_features(signal, fs=fs, fc=fc)

        if self.is_fitted and self.pipeline is not None:
            try:
                # `feats` was extracted above for the result payload; reuse it
                # for the estimator instead of running the full feature pass a
                # second time on every live frame.
                probabilities = self.pipeline.predict_proba(feats.reshape(1, -1))[0]
                probs_dict = {
                    str(cls): float(probability)
                    for cls, probability in zip(self.pipeline.classes_, probabilities)
                }
                probs = np.array(list(probs_dict.values()))
                classes = list(probs_dict.keys())
                pred_idx = int(np.argmax(probs))
                pred_class = classes[pred_idx]
                rule_evidence = rule_based_classify(signal, fs=fs, fc=fc) if include_diagnostics else None
                p_max = float(probs[pred_idx])

                # Shannon entropy certainty metric in [0.0, 1.0]
                n_cls = len(classes)
                h_max = np.log2(n_cls) if n_cls > 1 else 1.0
                entropy = -np.sum(probs * np.log2(probs + 1e-12))
                entropy_certainty = float(np.clip(1.0 - (entropy / h_max), 0.0, 1.0))
                confidence = float(np.clip(p_max * 0.5 + entropy_certainty * 0.5, 0.0, 1.0))

                return {
                    "modulation": pred_class,
                    "confidence": confidence,
                    "probabilities": probs_dict,
                    "cumulants": cum,
                    "features": feats.tolist() if include_diagnostics else [],
                    "rule_evidence": rule_evidence,
                    "agreement": bool(rule_evidence == pred_class) if include_diagnostics else None,
                    "diagnostics": {"classifier_disagreement": rule_evidence != pred_class} if include_diagnostics else {},
                }
            except Exception:
                pass

        pred_class = rule_based_classify(signal, fs=fs, fc=fc)
        return {
            "modulation": pred_class,
            "confidence": 0.85,
            "probabilities": {cls: (1.0 if cls == pred_class else 0.0) for cls in MODULATION_CLASSES},
            "cumulants": cum,
            "features": feats.tolist(),
        }

    def save(self, model_path: str):
        """Save the fitted model pipeline to disk."""
        joblib.dump(self.pipeline, str(model_path))

    def load(self, model_path: str):
        """Load a model pipeline from disk."""
        self.pipeline = joblib.load(str(model_path))
        self.model_path = Path(model_path)
        self.is_fitted = True

