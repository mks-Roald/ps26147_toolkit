import os
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from ps26147_toolkit.classifier import ModulationClassifier, extract_features, generate_synthetic_dataset, downconvert_baseband, rule_based_classify
from ps26147_toolkit.preprocess import load_iq, load_wav


class raw_fs(list):
    """List-like container for validated sampling frequencies."""

    @staticmethod
    def _validate(value):
        value = float(value)
        if not np.isfinite(value) or value <= 0:
            raise ValueError("Sampling frequency must be a finite positive number")
        return value

    def append(self, value):
        super().append(self._validate(value))

    def extend(self, values):
        for value in values:
            self.append(value)

    def insert(self, index, value):
        super().insert(index, self._validate(value))

    def __setitem__(self, index, value):
        if isinstance(index, slice):
            value = [self._validate(item) for item in value]
        else:
            value = self._validate(value)
        super().__setitem__(index, value)


def train_on_custom_data(data_dir: str):
    data_path = Path(data_dir)
    raw_signals = []
    raw_fs_values = raw_fs()
    iq_dir = Path(r"C:\Users\amity\Desktop\SIH_toolkit\ps26147_toolkit\scripts\dataset_out_iq")
    wav_dir = Path(r"C:\Users\amity\Desktop\SIH_toolkit\ps26147_toolkit\scripts\dataset_out_wav")
    # iq_dir = data_path / "dataset_out_iq"
    # wav_dir = data_path / "dataset_out_wav"
    
    print(f"Loading custom data from {data_path}...")
    X_feats = []
    y_labels = []

    if iq_dir.exists():
        for iq_file in iq_dir.glob("*.iq"):
            try:
                label = iq_file.name.split('_')[0]
                fs = 2_048_000.0 
                signal, _ = load_iq(iq_file, fs=fs)
                signal = downconvert_baseband(signal, fs, fc=500.0)
                feats = extract_features(signal, fs=fs)
                raw_signals.append(signal)
                raw_fs_values.append(fs)
                X_feats.append(feats)
                y_labels.append(label)
            except Exception as e:
                print(f"Failed to process {iq_file.name}: {e}")

    if wav_dir.exists():
        for wav_file in wav_dir.glob("*.wav"):
            try:
                label = wav_file.name.split('_')[0]
                signal, meta = load_wav(wav_file)
                signal = downconvert_baseband(signal, meta.fs, fc=500.0)
                feats = extract_features(signal, fs=meta.fs)
                raw_signals.append(signal)
                raw_fs_values.append(meta.fs)
                X_feats.append(feats)
                y_labels.append(label)
            except Exception as e:
                print(f"Failed to process {wav_file.name}: {e}")

    if not X_feats:
        print("No valid custom data found.")
        return

    X_custom = np.vstack(X_feats)
    y_custom = np.array(y_labels)

    print(f"Successfully loaded {len(X_custom)} custom samples across {len(np.unique(y_custom))} classes: {np.unique(y_custom)}")

    print("Generating standard synthetic dataset to retain old knowledge...")
    X_synthetic, y_synthetic = generate_synthetic_dataset(n_samples_per_class=120)
    print(f"Generated {len(X_synthetic)} synthetic samples.")

    raw_signals_all = [None] * len(X_synthetic) + raw_signals
    raw_fs_all = [None] * len(X_synthetic) + raw_fs_values
    
    # Combine custom data with old synthetic data
    X_combined = np.vstack([X_synthetic, X_custom])
    y_combined = np.concatenate([y_synthetic, y_custom])

    print(f"Total dataset size: {len(X_combined)} samples.")

    # Split into train and test sets
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X_combined, y_combined, test_size=0.20, random_state=42, stratify=y_combined
    # )
    indices = np.arange(len(X_combined))
    idx_train, idx_test = train_test_split(
    indices, test_size=0.20, random_state=42, stratify=y_combined
    )
    X_train, X_test = X_combined[idx_train], X_combined[idx_test]
    y_train, y_test = y_combined[idx_train], y_combined[idx_test]

    # Load existing classifier model to overwrite or update
    model_file = Path(__file__).resolve().parents[1] / "model.pkl"
    clf = ModulationClassifier(model_path=str(model_file))
    
    print("Training Random Forest classifier on custom data...")
    # This will train the model entirely on your custom data. 
    # If you want to merge it with synthetic data, you can append the data sets before this step.
    clf.train(X_train, y_train)

    print("Evaluating model...")
    y_pred = clf.pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\nTest Accuracy: {acc * 100:.2f}%\n")
    print(classification_report(y_test, y_pred))

    from sklearn.metrics import confusion_matrix
    labels_sorted = sorted(np.unique(y_combined))
    cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)
    print("\nConfusion Matrix (rows=true, cols=predicted):")
    print("Labels:", labels_sorted)
    print(cm)

    print("\n--- Benchmarking rule-only vs RF-only vs hybrid (real data only) ---")
    rule_preds, rf_preds, hybrid_preds, y_real_only = [], [], [], []

    for idx in idx_test:
        if raw_signals_all[idx] is None:
            continue
        sig, fs_i = raw_signals_all[idx], raw_fs_all[idx]
        rule_preds.append(rule_based_classify(sig, fs=fs_i))
        hybrid_preds.append(clf.hybrid_predict(sig, fs=fs_i, confidence_threshold=0.55))
        y_real_only.append(y_combined[idx])

    print("Rule-only accuracy:  ", accuracy_score(y_real_only, rule_preds))
    print("Hybrid accuracy:     ", accuracy_score(y_real_only, hybrid_preds))
    print("RF-only accuracy:    ", acc)  # your existing RF accuracy from earlier in the script

    print("\n--- Threshold sweep ---")
    for thresh in [0.3, 0.4, 0.5, 0.55, 0.6, 0.7, 0.8]:
        preds = [
            clf.hybrid_predict(raw_signals_all[idx], fs=raw_fs_all[idx], confidence_threshold=thresh)
            for idx in idx_test if raw_signals_all[idx] is not None
        ]
    acc_t = accuracy_score(y_real_only, preds)
    print(f"threshold={thresh}: accuracy={acc_t:.4f}")

    print(f"Saving newly trained model to {model_file}")
    clf.save(str(model_file))
    print("Done!")

if __name__ == "__main__":
    custom_dataset_path = r"C:\Users\amity\Desktop\SIH_toolkit\ps26147_toolkit\scripts\SIH_DataSet"
    train_on_custom_data(custom_dataset_path)

