"""Regression checks for hybrid feature scaling and inference preprocessing."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import scipy.io.wavfile as wav
import torch

from ps26147_toolkit.dataset import IQModulationDataset, fit_feature_scaler, load_signal
from ps26147_toolkit.hybrid_model import HybridModulationClassifier


def _write_iq(path: Path, samples: np.ndarray, fs: float) -> None:
    np.asarray(samples, dtype=np.complex64).tofile(path)
    path.with_name(path.name + ".sigmf-meta").write_text(json.dumps({
        "global": {"core:sample_rate": fs, "core:datatype": "cf32_le"}
    }), encoding="utf-8")


class HybridPreprocessingTests(unittest.TestCase):
    def test_iq_loader_uses_sigmf_rate_and_wav_keeps_file_rate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            iq_path = root / "BPSK_capture.iq"
            samples = np.exp(1j * np.arange(5000, dtype=np.float32) * 0.02).astype(np.complex64)
            _write_iq(iq_path, samples, 500_000.0)
            loaded, fs = load_signal(iq_path)
            self.assertEqual(fs, 500_000.0)
            np.testing.assert_allclose(loaded, samples)

            legacy_path = root / "QPSK_legacy.iq"
            np.asarray(samples, dtype=np.complex64).tofile(legacy_path)
            _, fallback_fs = load_signal(legacy_path)
            self.assertEqual(fallback_fs, 2_048_000.0)

            invalid_path = root / "8PSK_invalid.iq"
            _write_iq(invalid_path, samples, 0.0)
            with self.assertRaisesRegex(ValueError, "Invalid core:sample_rate"):
                load_signal(invalid_path)

            wav_path = root / "AM_capture.wav"
            wav.write(wav_path, 44_100, np.zeros(5000, dtype=np.int16))
            _, wav_fs = load_signal(wav_path)
            self.assertEqual(wav_fs, 44_100.0)

    def test_scaler_is_train_only_frozen_and_checkpoint_round_trips(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = []
            rng = np.random.default_rng(12)
            for i in range(6):
                path = root / f"BPSK_train_{i}.iq"
                samples = (0.1 * rng.normal(size=8192)
                           + 0.1j * rng.normal(size=8192)).astype(np.complex64)
                _write_iq(path, samples, 1_000_000.0)
                records.append({"path": str(path), "label": "BPSK"})
            scaler = fit_feature_scaler(records[:4])
            mean_before = scaler.mean_.copy()
            self.assertEqual(scaler.n_samples_seen_, 4)
            calibration_matrix = np.stack([
                IQModulationDataset([record], mode="val")[0][1].numpy()
                for record in records[:4]
            ])
            standardized = scaler.transform(calibration_matrix)
            np.testing.assert_allclose(standardized.mean(axis=0), 0.0, atol=1e-4)
            active = scaler.var_ > 1e-12
            np.testing.assert_allclose(standardized.var(axis=0)[active], 1.0, atol=1e-5)
            for mode in ("val", "test"):
                heldout_ds = IQModulationDataset(records[4:], mode=mode, feature_scaler=scaler)
                for index in range(len(heldout_ds)):
                    _, features, _ = heldout_ds[index]
                    self.assertEqual(tuple(features.shape), (16,))
            np.testing.assert_array_equal(scaler.mean_, mean_before)

            checkpoint = root / "hybrid.pt"
            classifier = HybridModulationClassifier(checkpoint, device="cpu", load_existing=False)
            classifier.feature_scaler = {
                "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
                "var": scaler.var_.tolist(), "n_samples_seen": int(np.max(scaler.n_samples_seen_)),
                "n_features_in": int(scaler.n_features_in_),
            }
            classifier.is_fitted = True
            classifier.save()
            restored = HybridModulationClassifier(checkpoint, device="cpu")
            np.testing.assert_allclose(restored.feature_scaler["mean"], scaler.mean_)
            np.testing.assert_allclose(restored.feature_scaler["scale"], scaler.scale_)
            self.assertEqual(restored.feature_scaler["n_samples_seen"], 4)

    def test_public_inference_uses_one_center_window_and_actual_fs(self):
        import ps26147_toolkit.hybrid_model as hybrid_module

        with tempfile.TemporaryDirectory() as tmp:
            classifier = HybridModulationClassifier(Path(tmp) / "absent.pt", device="cpu",
                                                    load_existing=False)
            classifier.is_fitted = True
            classifier.feature_scaler = {"mean": [0.0] * 16, "scale": [1.0] * 16}
            signal = (np.arange(10_000, dtype=np.float32)
                      + 1j * np.arange(10_000, dtype=np.float32)[::-1]).astype(np.complex64)
            observed = {}

            def fake_features(window, fs):
                observed["window"] = np.asarray(window).copy()
                observed["fs"] = fs
                return np.arange(16, dtype=np.float32)

            with patch.object(hybrid_module, "extract_features", fake_features):
                probs = classifier.predict_proba(signal, fs=500_000.0)
                start = (len(signal) - 4096) // 2
                np.testing.assert_array_equal(observed["window"], signal[start:start + 4096])
                self.assertEqual(observed["fs"], 500_000.0)
                self.assertEqual(len(probs), 9)
                self.assertAlmostEqual(sum(probs.values()), 1.0, places=5)
                _, iq_input, feature_input = classifier._prepare_inference_inputs(signal, 500_000.0)
            self.assertEqual(tuple(iq_input.shape), (1, 2, 4096))
            self.assertEqual(tuple(feature_input.shape), (1, 16))
            self.assertTrue(torch.isfinite(iq_input).all())
            self.assertTrue(torch.isfinite(feature_input).all())

    def test_old_checkpoint_warns_without_fabricating_scaler(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "legacy.pt"
            model = HybridModulationClassifier(root / "missing.pt", device="cpu",
                                               load_existing=False).model
            torch.save({"state_dict": model.state_dict(),
                        "class_names": ("BPSK", "QPSK", "8PSK", "16QAM", "64QAM",
                                        "2FSK", "4FSK", "AM", "FM"),
                        "sample_length": 4096, "feature_count": 16}, checkpoint)
            with self.assertWarnsRegex(RuntimeWarning, "predates feature standardization"):
                classifier = HybridModulationClassifier(checkpoint, device="cpu")
            self.assertIsNone(classifier.feature_scaler)


if __name__ == "__main__":
    unittest.main()
