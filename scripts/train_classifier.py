import os
import joblib
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from scipy.io import loadmat

# Import the feature extractor from the toolkit
from ..ps26147_toolkit.classifier import extract_features, ModulationClassifier

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
MODEL_PATH = Path(__file__).resolve().parents[1] / "model.pkl"

def load_radio_ml_samples():
    """Load samples from RadioML2016.10a.
    The dataset contains .mat files where each entry has:
        - X: complex IQ array (shape: [samples, 2] interleaved I/Q)
        - Y: modulation label (string)
        - snr: signal‑to‑noise ratio (dB)
    This function extracts the IQ as a 1‑D complex numpy array and returns
    a list of (signal, label, fs) tuples.
    """
    samples = []
    # The dataset extracts to a folder named 'RadioML2016.10a' with many .mat files
    radio_ml_root = DATA_DIR / "RadioML2016.10a"
    if not radio_ml_root.exists():
        raise FileNotFoundError(f"RadioML data not found in {radio_ml_root}. Run download_datasets.py first.")
    # Recursively find .mat files
    for mat_path in radio_ml_root.rglob("*.mat"):
        mat = loadmat(str(mat_path))
        # Expected keys: 'X', 'Y', 'snr'
        X = mat.get("X")
        Y = mat.get("Y")
        if X is None or Y is None:
            continue
        # X is shape (samples, 2) with I/Q columns; convert to complex
        for i in range(X.shape[0]):
            iq = X[i]
            # Some versions store as (N, 2) float32, others as (N, 1) complex128
            if iq.ndim == 2 and iq.shape[1] == 2:
                signal = iq[:, 0] + 1j * iq[:, 1]
            else:
                signal = iq.squeeze()
            label = str(Y[i][0]) if isinstance(Y[i], np.ndarray) else str(Y[i])
            # RadioML uses a nominal sample rate of 1 Msps for all files
            fs = 1e6
            samples.append((signal, label, fs))
    return samples

def main():
    print("Loading RadioML samples …")
    samples = load_radio_ml_samples()
    print(f"Loaded {len(samples)} samples.")

    # Build feature matrix and label vector
    X_feats = []
    y_labels = []
    for signal, label, fs in samples:
        feats = extract_features(signal, fs)
        X_feats.append(feats)
        y_labels.append(label)
    X = np.vstack(X_feats)
    y = np.array(y_labels)

    # Train / test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Initialise the classifier (its internal pipeline is a RandomForest)
    clf = ModulationClassifier()
    print("Training RandomForest classifier …")
    clf.train(X_train, y_train)

    # Evaluation
    y_pred = clf.pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc * 100:.2f}%")
    print("Classification report:\n")
    print(classification_report(y_test, y_pred))

    # Save the trained model for later use by the CLI
    print(f"Saving trained model to {MODEL_PATH}")
    clf.save(str(MODEL_PATH))
    print("Done.")

if __name__ == "__main__":
    main()
