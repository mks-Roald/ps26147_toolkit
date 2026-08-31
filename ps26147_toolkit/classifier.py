import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.signal import welch

# Simple feature extractor: use PSD summary statistics

def extract_features(signal: np.ndarray, fs: float) -> np.ndarray:
    """Extract a short feature vector from a signal.
    Uses Welch PSD summary statistics.
    """
    nperseg = min(1024, max(2, signal.size))
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)
    feats = [np.mean(psd), np.std(psd), np.max(psd), np.min(psd), np.median(psd), np.percentile(psd, 75) - np.percentile(psd, 25)]
    return np.array(feats)

from sklearn.exceptions import NotFittedError

class ModulationClassifier:
    def __init__(self, model_path: str = None):
        if model_path is None:
            default_model = Path(__file__).resolve().parents[1] / "model.pkl"
            if default_model.exists():
                model_path = str(default_model)

        if model_path and Path(model_path).exists():
            self.pipeline = joblib.load(model_path)
            self.is_fitted = True
        else:
            self.pipeline = Pipeline([('scaler', StandardScaler()), ('rf', RandomForestClassifier(n_estimators=200, random_state=42))])
            self.is_fitted = False

    def train(self, X: np.ndarray, y: np.ndarray):
        """Train the classifier.
        X: (n_samples, n_features)
        y: (n_samples,) string labels (e.g., 'BPSK').
        """
        self.pipeline.fit(X, y)
        self.is_fitted = True

    def predict(self, signal: np.ndarray, fs: float) -> str:
        feats = extract_features(signal, fs).reshape(1, -1)
        try:
            return str(self.pipeline.predict(feats)[0])
        except NotFittedError:
            return "Untrained Model (Run train_classifier.py)"

    def save(self, model_path: str):
        joblib.dump(self.pipeline, model_path)
