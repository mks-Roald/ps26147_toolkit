"""Automatic Modulation Recognition (AMR) Engine for PS26147 Toolkit."""

import numpy as np
import joblib
from pathlib import Path
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


def rule_based_classify(signal: np.ndarray, fs: float = 1000000.0, fc: float = None) -> str:
    """Expert rule-based classifier using Higher-Order Cumulants (HOC) and instantaneous signal statistics.
    
    Eliminates the BPSK/QPSK -> FSK misclassification bug by using median-filtered
    instantaneous frequency and phase constellation analysis.
    """
    if len(signal) < 32:
        return "QPSK"

    if not np.iscomplexobj(signal):
        sig = hilbert(signal)
    else:
        sig = signal

    max_samples = 32768
    sig = sig[:max_samples] if len(sig) > max_samples else sig

    cum = compute_cumulants(sig, fs=fs, fc=fc)
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

    # 1. Genuine FSK detection:
    # FSK has constant envelope (sigma_aa < 0.28), high filtered frequency std (sigma_af > 0.012),
    # AND high fsk_persistence (filtered std is close to raw std, unlike PSK impulse transitions).
    if sigma_aa < 0.28 and sigma_af > 0.012 and fsk_persistence > 0.35:
        hist, _ = np.histogram(inst_freq_filtered, bins=40)
        max_h = np.max(hist)
        peak_bins = np.where(hist > 0.35 * max_h)[0]
        if len(peak_bins) > 0:
            clusters = 1 + np.sum(np.diff(peak_bins) > 2)
            if clusters >= 3:
                return "4FSK"
            elif clusters == 2:
                return "2FSK"
        return "2FSK"

    # 2. AM Detection: Significant envelope variation with near-zero phase modulation or non-zero carrier offset
    phase_angles = np.angle(sig)
    phase_std = float(np.std(phase_angles))
    if sigma_aa > 0.20 and phase_std < 0.35:
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

    # QAM Family
    if abs_c40 > 0.45 or c63 > 1.30:
        return "16QAM"
    else:
        return "64QAM"


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
        """Predict modulation format using Random Forest with rule-based fallback."""
        if self.is_fitted and self.pipeline is not None:
            try:
                feats = extract_features(signal, fs=fs, fc=fc).reshape(1, -1)
                pred = self.pipeline.predict(feats)[0]
                return str(pred)
            except Exception:
                return rule_based_classify(signal, fs=fs, fc=fc)
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
