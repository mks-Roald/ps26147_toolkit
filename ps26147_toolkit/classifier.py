import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.signal import hilbert, welch


def compute_cumulants(signal: np.ndarray) -> dict:
    """Compute 2nd, 4th, and 6th order cumulants of a normalized complex baseband signal.
    Cumulants are invariant to phase rotation and provide distinctive signatures
    for BPSK, QPSK, 8PSK, 16QAM, 64QAM, FSK, and AM/FM.
    """
    if len(signal) < 32:
        return {
            "c20": 0.0, "c21": 1.0, "c40": 0.0, "c41": 0.0,
            "c42": -1.0, "c60": 0.0, "c63": 4.0,
        }

    # Zero-mean and unit-variance normalization
    s = signal - np.mean(signal)
    var = np.mean(np.abs(s) ** 2)
    if var > 1e-12:
        s = s / np.sqrt(var)
    else:
        return {
            "c20": 0.0, "c21": 1.0, "c40": 0.0, "c41": 0.0,
            "c42": -1.0, "c60": 0.0, "c63": 4.0,
        }

    # Moments
    s_conj = np.conj(s)
    m20 = np.mean(s ** 2)
    m21 = np.mean(np.abs(s) ** 2)
    m40 = np.mean(s ** 4)
    m41 = np.mean((s ** 3) * s_conj)
    m42 = np.mean(np.abs(s) ** 4)
    m60 = np.mean(s ** 6)
    m63 = np.mean(np.abs(s) ** 6)

    # Cumulants
    c20 = m20
    c21 = m21
    c40 = m40 - 3 * (m20 ** 2)
    c41 = m41 - 3 * m20 * m21
    c42 = m42 - np.abs(m20) ** 2 - 2 * (m21 ** 2)
    c60 = m60 - 15 * m40 * m20 + 30 * (m20 ** 3)
    c63 = m63 - 9 * c42 * c21 - np.abs(c40) ** 2 - 6 * (c21 ** 3)

    return {
        "c20": complex(c20),
        "c21": float(np.real(c21)),
        "c40": complex(c40),
        "c41": complex(c41),
        "c42": float(np.real(c42)),
        "c60": complex(c60),
        "c63": float(np.real(c63)),
    }


def extract_features(signal: np.ndarray, fs: float) -> np.ndarray:
    """Extract a comprehensive feature vector including Higher-Order Cumulants (HOC),
    instantaneous amplitude/phase/frequency statistics, and PSD descriptors.
    """
    if not np.iscomplexobj(signal):
        sig_c = hilbert(signal)
    else:
        sig_c = signal

    max_samples = 32768
    sig = sig_c[:max_samples] if len(sig_c) > max_samples else sig_c

    # 1. Cumulants
    cum = compute_cumulants(sig)
    abs_c20 = np.abs(cum["c20"])
    abs_c40 = np.abs(cum["c40"])
    abs_c41 = np.abs(cum["c41"])
    c42 = cum["c42"]
    abs_c60 = np.abs(cum["c60"])
    c63 = cum["c63"]

    # 2. Instantaneous features
    env = np.abs(sig)
    env_norm = env / (np.mean(env) + 1e-12)
    gamma_max = np.max(env_norm ** 2) if len(env_norm) > 0 else 1.0
    sigma_aa = np.std(env_norm)

    # Instantaneous phase & frequency
    phase = np.unwrap(np.angle(sig))
    phase_diff = np.diff(phase)
    inst_freq = phase_diff / (2 * np.pi) * fs
    sigma_af = np.std(inst_freq) / (fs + 1e-12)
    sigma_dp = np.std(np.abs(np.angle(sig)))

    # 3. Spectral features
    nperseg = min(512, max(16, len(sig)))
    _, psd = welch(sig, fs=fs, nperseg=nperseg)
    psd_norm = psd / (np.sum(psd) + 1e-12)
    spec_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))
    kurtosis_env = (np.mean((env - np.mean(env)) ** 4) / (np.std(env) ** 4 + 1e-12)) - 3.0

    feats = [
        abs_c20,
        abs_c40,
        abs_c41,
        c42,
        abs_c60,
        c63,
        gamma_max,
        sigma_aa,
        sigma_dp,
        sigma_af,
        spec_entropy,
        kurtosis_env,
    ]
    return np.array(feats, dtype=np.float32)


def rule_based_classify(signal: np.ndarray, fs: float) -> str:
    """Expert rule-based classifier based on Higher-Order Cumulants (HOC)
    and instantaneous frequency/envelope distributions.
    """
    if not np.iscomplexobj(signal):
        sig = hilbert(signal)
    else:
        sig = signal

    cum = compute_cumulants(sig)
    abs_c20 = np.abs(cum["c20"])
    abs_c40 = np.abs(cum["c40"])
    c42 = cum["c42"]
    abs_c41 = np.abs(cum["c41"])

    env = np.abs(sig)
    sigma_aa = np.std(env / (np.mean(env) + 1e-12))

    # Instantaneous frequency analysis
    inst_freq = np.diff(np.unwrap(np.angle(sig))) / (2 * np.pi) * fs
    freq_std = np.std(inst_freq)

    # Check for FSK (distinct frequency shifts, low amplitude variation)
    if sigma_aa < 0.3 and freq_std > fs * 0.02:
        # Check number of distinct frequency clusters
        hist, _ = np.histogram(inst_freq, bins=30)
        peaks = np.sum(hist > (np.max(hist) * 0.4))
        if peaks >= 3:
            return "4FSK"
        return "2FSK"

    # Constant modulus signals: BPSK, QPSK, 8PSK, FM
    if c42 < -0.85:
        if abs_c20 > 0.65 or abs_c40 > 1.4:
            return "BPSK"
        elif abs_c40 > 0.65:
            return "QPSK"
        elif abs_c40 < 0.4:
            return "8PSK"
        return "QPSK"

    # Non-constant modulus: QAM & AM
    if -0.85 <= c42 <= -0.4:
        if abs_c40 > 0.45:
            return "16QAM"
        else:
            return "64QAM"

    if sigma_aa > 0.45:
        return "AM"

    if abs_c40 > 1.2:
        return "BPSK"
    elif abs_c40 > 0.5:
        return "QPSK"

    return "QPSK"


def generate_synthetic_dataset(n_samples_per_class: int = 100, fs: float = 1000000.0) -> tuple[np.ndarray, np.ndarray]:
    """Generate clean & noisy synthetic training data for multiple modulations."""
    classes = ["BPSK", "QPSK", "8PSK", "16QAM", "64QAM", "2FSK", "4FSK", "AM"]
    X, y = [], []
    num_symbols = 512
    sps = 8  # samples per symbol
    n_samples = num_symbols * sps

    for cls in classes:
        for _ in range(n_samples_per_class):
            snr_db = np.random.uniform(5, 30)
            if cls == "BPSK":
                syms = np.random.choice([-1, 1], size=num_symbols)
                sig = np.repeat(syms, sps).astype(np.complex64)
            elif cls == "QPSK":
                syms = np.random.choice([-1-1j, -1+1j, 1-1j, 1+1j] / np.sqrt(2), size=num_symbols)
                sig = np.repeat(syms, sps).astype(np.complex64)
            elif cls == "8PSK":
                phases = np.random.choice(np.arange(8) * (2 * np.pi / 8), size=num_symbols)
                syms = np.exp(1j * phases)
                sig = np.repeat(syms, sps).astype(np.complex64)
            elif cls == "16QAM":
                grid = np.array([-3, -1, 1, 3])
                syms = np.random.choice(grid, size=num_symbols) + 1j * np.random.choice(grid, size=num_symbols)
                syms = syms / np.sqrt(10)
                sig = np.repeat(syms, sps).astype(np.complex64)
            elif cls == "64QAM":
                grid = np.array([-7, -5, -3, -1, 1, 3, 5, 7])
                syms = np.random.choice(grid, size=num_symbols) + 1j * np.random.choice(grid, size=num_symbols)
                syms = syms / np.sqrt(42)
                sig = np.repeat(syms, sps).astype(np.complex64)
            elif cls == "2FSK":
                bits = np.random.choice([0, 1], size=num_symbols)
                f_dev = 25000.0
                freqs = np.where(np.repeat(bits, sps) == 1, f_dev, -f_dev)
                phase = 2 * np.pi * np.cumsum(freqs) / fs
                sig = np.exp(1j * phase)
            elif cls == "4FSK":
                symbols = np.random.choice([-3, -1, 1, 3], size=num_symbols)
                f_dev = 15000.0
                freqs = np.repeat(symbols, sps) * f_dev
                phase = 2 * np.pi * np.cumsum(freqs) / fs
                sig = np.exp(1j * phase)
            elif cls == "AM":
                t = np.arange(n_samples) / fs
                mod_sig = 0.5 * np.cos(2 * np.pi * 1000 * t) + 0.3 * np.sin(2 * np.pi * 3000 * t)
                sig = (1.0 + 0.8 * mod_sig).astype(np.complex64)

            # Add AWGN noise
            sig_power = np.mean(np.abs(sig) ** 2)
            noise_power = sig_power / (10 ** (snr_db / 10.0))
            noise = np.sqrt(noise_power / 2) * (np.random.randn(len(sig)) + 1j * np.random.randn(len(sig)))
            sig_noisy = sig + noise

            feats = extract_features(sig_noisy, fs)
            X.append(feats)
            y.append(cls)

    return np.vstack(X), np.array(y)


class ModulationClassifier:
    def __init__(self, model_path: str = None):
        if model_path is None:
            default_model = Path(__file__).resolve().parents[1] / "model.pkl"
            self.model_path = default_model
        else:
            self.model_path = Path(model_path)

        if self.model_path.exists():
            try:
                self.pipeline = joblib.load(str(self.model_path))
                self.is_fitted = True
            except Exception:
                self._build_and_train_default()
        else:
            self._build_and_train_default()

    def _build_and_train_default(self):
        """Train a lightweight Random Forest model on synthesized data and save to disk."""
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("rf", RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42)),
        ])
        X, y = generate_synthetic_dataset(n_samples_per_class=60)
        self.pipeline.fit(X, y)
        self.is_fitted = True
        try:
            joblib.dump(self.pipeline, str(self.model_path))
        except Exception:
            pass

    def train(self, X: np.ndarray, y: np.ndarray):
        """Train the classifier on custom feature matrix X and label array y."""
        self.pipeline.fit(X, y)
        self.is_fitted = True

    def predict(self, signal: np.ndarray, fs: float) -> str:
        """Predict modulation format using Random Forest with Higher-Order Cumulants fallback."""
        try:
            feats = extract_features(signal, fs).reshape(1, -1)
            pred = self.pipeline.predict(feats)[0]
            return str(pred)
        except Exception:
            return rule_based_classify(signal, fs)

    def save(self, model_path: str):
        joblib.dump(self.pipeline, model_path)
