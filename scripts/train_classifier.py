"""Training script for ModulationClassifier with RadioML and synthetic data fallbacks."""

import os
import joblib
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from scipy.io import loadmat

from ps26147_toolkit.classifier import (
    ModulationClassifier,
    extract_features,
    generate_synthetic_dataset,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
MODEL_PATH = Path(__file__).resolve().parents[1] / "model.pkl"


def load_radio_ml_samples():
    """Load samples from RadioML2016.10a if present."""
    samples = []
    radio_ml_root = DATA_DIR / "RadioML2016.10a"
    if not radio_ml_root.exists():
        return None

    for mat_path in radio_ml_root.rglob("*.mat"):
        try:
            mat = loadmat(str(mat_path))
            X = mat.get("X")
            Y = mat.get("Y")
            if X is None or Y is None:
                continue
            for i in range(X.shape[0]):
                iq = X[i]
                if iq.ndim == 2 and iq.shape[1] == 2:
                    signal = iq[:, 0] + 1j * iq[:, 1]
                else:
                    signal = iq.squeeze()
                label = str(Y[i][0]) if isinstance(Y[i], np.ndarray) else str(Y[i])
                fs = 1e6
                samples.append((signal, label, fs))
        except Exception:
            continue
    return samples if samples else None


def main():
    print("Checking for RadioML dataset...")
    samples = load_radio_ml_samples()

    if samples:
        print(f"Loaded {len(samples)} samples from RadioML.")
        X_feats = []
        y_labels = []
        for signal, label, fs in samples:
            feats = extract_features(signal, fs)
            X_feats.append(feats)
            y_labels.append(label)
        X = np.vstack(X_feats)
        y = np.array(y_labels)
    else:
        print("RadioML dataset not found. Generating realistic synthetic dataset with impairments...")
        X, y = generate_synthetic_dataset(n_samples_per_class=150)
        print(f"Generated synthetic dataset: {X.shape[0]} samples across {len(np.unique(y))} classes.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    clf = ModulationClassifier(model_path=str(MODEL_PATH))
    print("Training Random Forest classifier pipeline...")
    clf.train(X_train, y_train)

    y_pred = clf.pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {acc * 100:.2f}%\n")
    print(classification_report(y_test, y_pred))

    print(f"Saving trained model to {MODEL_PATH}")
    clf.save(str(MODEL_PATH))
    print("Training complete.")


if __name__ == "__main__":
    main()
