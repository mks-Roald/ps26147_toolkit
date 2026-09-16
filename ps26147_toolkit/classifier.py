"""Automatic Modulation Recognition (AMR) Engine for PS26147 Toolkit."""

import numpy as np
import joblib
from pathlib import Path
from typing import Optional
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


def rule_based_classify(signal: np.ndarray, fs: float = 1000000.0, fc: float = None) -> str:
    """Expert rule-based classifier using Higher-Order Cumulants (HOC) and instantaneous signal statistics.

    Eliminates the BPSK/QPSK -> FSK misclassification bug by using median-filtered
    instantaneous frequency and phase constellation analysis.

    The signal (real passband or complex) is downconverted to complex baseband
    first so the cumulant thresholds below are meaningful.  ``fc`` is the carrier
    used for downconversion; pass it when you have an accurate estimate (or a
    ground-truth value) so no residual carrier frequency offset corrupts the
    features.
    """
    if len(signal) < 32:
        return "QPSK"

    sig = downconvert_baseband(signal, fs, fc=fc)

    max_samples = 32768
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

    # 1. Genuine FSK detection (baseband):
    # FSK has near-constant envelope (sigma_aa ~ 0.004 for clean corpus, well
    # below any PSK/QAM which sits >0.28) and very high fsk_persistence
    # (filtered inst-freq std tracks raw, ~0.99 vs PSK ~0.5).
    # The old gate used sigma_af > 0.012 which excludes FSK at low baud
    # (4FSK=0.0055, 2FSK=0.0102) and is unnecessary once sigma_aa + fskP
    # isolate FSK.  Histogram clustering then distinguishes 2 vs 4 peaks.
    if sigma_aa < 0.10 and fsk_persistence > 0.85:
        hist, _ = np.histogram(inst_freq_filtered, bins=40)
        max_h = np.max(hist)
        peak_bins = np.where(hist > 0.20 * max_h)[0]
        if len(peak_bins) > 0:
            clusters = 1 + np.sum(np.diff(peak_bins) > 2)
            if clusters >= 3:
                return "4FSK"
            elif clusters == 2:
                return "2FSK"
        return "2FSK"

    # 2. AM Detection: Significant envelope variation with near-zero phase modulation or non-zero carrier offset
    # More strict to avoid misclassifying PSK/QAM as AM
    phase_angles = np.angle(sig)
    phase_std = float(np.std(phase_angles))
    if sigma_aa > 0.35 and phase_std < 0.25:
        return "AM"

    # 3. PSK vs QAM Discrimination using 4th-power phase folding
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
        """Predict modulation using the baseband rule-based classifier.

        The deterministic rule-based engine is authoritative because it
        downconverts to complex baseband before computing cumulant/instantaneous
        features — the only regime where those thresholds are meaningful.  The
        RF model is retained for optional use (``predict_proba``) but is not
        used for the modulation decision.
        """
        return rule_based_classify(signal, fs=fs, fc=fc)

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
        self, signal: np.ndarray, fs: float = 1000000.0, fc: float = None
    ) -> dict:
        """Predict modulation with confidence score, class probabilities, and diagnostic cumulants."""
        cum = compute_cumulants(signal, fs=fs, fc=fc)
        feats = extract_features(signal, fs=fs, fc=fc)

        if self.is_fitted and self.pipeline is not None:
            try:
                probs_dict = self.predict_proba(signal, fs=fs, fc=fc)
                probs = np.array(list(probs_dict.values()))
                classes = list(probs_dict.keys())
                pred_idx = int(np.argmax(probs))
                pred_class = classes[pred_idx]
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
                    "features": feats.tolist(),
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
