"""Hybrid raw-IQ CNN and handcrafted-feature fusion classifier."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import warnings
from torch import nn
from torch.utils.data import DataLoader

from .dataset import CLASSES, SAMPLE_LENGTH, IQModulationDataset, fit_feature_scaler, prepare_iq
from .feature_extractor import extract_features


class HybridNetwork(nn.Module):
    """CNN/MLP feature fusion model producing nine class logits."""
    def __init__(self, num_classes: int = 9, feature_count: int = 16,
                 *, disable_dropout: bool = False):
        super().__init__()
        dropout = 0.0 if disable_dropout else 0.30
        dropout_second = 0.0 if disable_dropout else 0.20
        self.iq_branch = nn.Sequential(
            nn.Conv1d(2, 32, 7, padding=3, bias=False), nn.BatchNorm1d(32), nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, 5, padding=2, bias=False), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1, bias=False), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
        )
        self.feature_branch = nn.Sequential(nn.Linear(feature_count, 32), nn.ReLU(inplace=True),
                                            nn.Linear(32, 32), nn.ReLU(inplace=True))
        self.fusion = nn.Sequential(nn.Linear(160, 128), nn.ReLU(inplace=True), nn.Dropout(dropout),
                                    nn.Linear(128, 64), nn.ReLU(inplace=True), nn.Dropout(dropout_second),
                                    nn.Linear(64, num_classes))

    def forward(self, iq: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        return self.fusion(torch.cat((self.iq_branch(iq), self.feature_branch(features)), dim=1))


class HybridModulationClassifier:
    """Train and run hybrid modulation recognition; RF artifacts stay independent."""
    def __init__(self, model_path: str | Path | None = None, device: str | None = None,
                 *, load_existing: bool = True, disable_dropout: bool = False):
        self.model_path = Path(model_path) if model_path else Path(__file__).resolve().parents[1] / "hybrid_model.pt"
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.class_names = tuple(CLASSES)
        self.disable_dropout = disable_dropout
        self.model = HybridNetwork(len(self.class_names), disable_dropout=disable_dropout).to(self.device)
        self.feature_scaler: dict[str, Any] | None = None
        self.is_fitted = False
        if load_existing and self.model_path.exists():
            self.load(self.model_path)

    def train(self, train_data: Any, labels: np.ndarray | None = None, *, epochs: int = 20,
              batch_size: int = 32, learning_rate: float = 1e-3, weight_decay: float = 1e-4,
              validation_data: Any = None, num_workers: int = 0) -> list[dict[str, float]]:
        """Train from a DataLoader/dataset, or from iterable records plus labels.

        Dataset/DataLoader items must be ``(iq, features, class_index)``. Raw
        signals can be supplied as records with ``path`` and ``label`` keys.
        """
        if epochs < 1 or batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")
        if isinstance(train_data, list) and (not train_data or isinstance(train_data[0], dict)):
            dataset = IQModulationDataset(train_data)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                pin_memory=self.device.type == "cuda")
        elif isinstance(train_data, DataLoader):
            loader = train_data
        elif isinstance(train_data, torch.utils.data.Dataset):
            loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                pin_memory=self.device.type == "cuda")
        else:
            raise TypeError("train_data must be a DataLoader, Dataset, or list of labeled records")
        if len(loader.dataset) == 0:
            raise ValueError("Cannot train on an empty dataset")
        training_dataset = loader.dataset
        if isinstance(training_dataset, IQModulationDataset):
            if training_dataset.feature_scaler is None:
                fitted_scaler = fit_feature_scaler(training_dataset.records,
                                                   training_dataset.sample_length)
                self.feature_scaler = {
                    "mean": fitted_scaler.mean_.astype(np.float64).tolist(),
                    "scale": fitted_scaler.scale_.astype(np.float64).tolist(),
                    "var": fitted_scaler.var_.astype(np.float64).tolist(),
                    "n_samples_seen": int(np.max(np.asarray(fitted_scaler.n_samples_seen_))),
                    "n_features_in": int(fitted_scaler.n_features_in_),
                }
                training_dataset.feature_scaler = fitted_scaler
            elif self.feature_scaler is None:
                fitted_scaler = training_dataset.feature_scaler
                self.feature_scaler = {
                    "mean": np.asarray(fitted_scaler.mean_).tolist(),
                    "scale": np.asarray(fitted_scaler.scale_).tolist(),
                    "var": np.asarray(fitted_scaler.var_).tolist(),
                    "n_samples_seen": int(np.max(np.asarray(fitted_scaler.n_samples_seen_))),
                    "n_features_in": int(fitted_scaler.n_features_in_),
                }

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()
        amp_enabled = self.device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        history: list[dict[str, float]] = []
        for epoch in range(epochs):
            self.model.train()
            total_loss, correct, count = 0.0, 0, 0
            for iq, features, target in loader:
                iq = iq.to(self.device, non_blocking=True)
                features = features.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True).long()
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=amp_enabled):
                    logits = self.model(iq, features)
                    loss = criterion(logits, target)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.item() * target.size(0)
                correct += int((logits.argmax(1) == target).sum().item())
                count += target.size(0)
            metrics = {"epoch": float(epoch + 1), "loss": total_loss / count, "accuracy": correct / count}
            history.append(metrics)
            print(f"Epoch {epoch + 1}/{epochs} loss={metrics['loss']:.4f} accuracy={metrics['accuracy']:.4f}")
        self.is_fitted = True
        return history

    @staticmethod
    def _prepare_inference_window(signal: np.ndarray, length: int = SAMPLE_LENGTH) -> np.ndarray:
        """Use one deterministic center crop (and right-pad short signals)."""
        x = np.asarray(signal)
        if x.ndim != 1:
            raise ValueError("signal must be a one-dimensional array")
        if not np.iscomplexobj(x):
            from scipy.signal import hilbert
            x = hilbert(x).astype(np.complex64)
        start = max(0, (len(x) - length) // 2)
        window = np.asarray(x[start:start + length])
        if len(window) < length:
            window = np.pad(window, (0, length - len(window)))
        return np.nan_to_num(window, nan=0.0, posinf=0.0, neginf=0.0)

    def _prepare_features(self, window: np.ndarray, fs: float) -> np.ndarray:
        """Extract and apply the frozen training-only feature transform."""
        features = np.nan_to_num(extract_features(window, fs=fs), nan=0.0,
                                 posinf=100.0, neginf=-100.0).astype(np.float32)
        if features.shape != (16,):
            raise ValueError(f"Expected 16 handcrafted features; got {features.shape}")
        if self.feature_scaler is not None:
            mean = np.asarray(self.feature_scaler["mean"], dtype=np.float32)
            scale = np.asarray(self.feature_scaler["scale"], dtype=np.float32)
            features = (features - mean) / scale
        return features.astype(np.float32, copy=False)

    def _prepare_inference_inputs(self, signal: np.ndarray, fs: float
                                  ) -> tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        """Build both branch inputs from one deterministic window."""
        window = self._prepare_inference_window(signal)
        iq_input = torch.from_numpy(prepare_iq(window)).unsqueeze(0)
        feature_input = torch.from_numpy(self._prepare_features(window, fs)).unsqueeze(0)
        return window, iq_input, feature_input

    @torch.inference_mode()
    def predict_proba(self, signal: np.ndarray | str | Path,
                      fs: float | None = None) -> dict[str, float]:
        if not self.is_fitted:
            raise RuntimeError("Model is not trained or loaded")
        if isinstance(signal, (str, Path)):
            # File-backed inference reads actual per-file metadata, including WAV rate.
            from .dataset import load_signal
            signal, file_fs = load_signal(signal)
            fs = file_fs
        elif fs is None:
            # Array callers must supply their acquisition rate; retain the historic
            # default only for source compatibility while making it explicit here.
            fs = 1_000_000.0
        if not np.isfinite(fs) or fs <= 0:
            raise ValueError(f"fs must be finite and positive, got {fs!r}")
        # Both branches derive from the same physical segment to match training.
        _, iq, ft = self._prepare_inference_inputs(signal, fs)
        iq = iq.to(self.device)
        ft = ft.to(self.device)
        self.model.eval()
        probs = torch.softmax(self.model(iq, ft), dim=1)[0].cpu().numpy()
        return {name: float(probs[i]) for i, name in enumerate(self.class_names)}

    def predict(self, signal: np.ndarray | str | Path,
                fs: float | None = None) -> str:
        probs = self.predict_proba(signal, fs=fs)
        return max(probs, key=probs.get)

    def predict_with_confidence(self, signal: np.ndarray | str | Path,
                                fs: float | None = None) -> dict[str, Any]:
        """Return predicted class, class probabilities, and top-class confidence."""
        probabilities = self.predict_proba(signal, fs=fs)
        label = max(probabilities, key=probabilities.get)
        return {"prediction": label, "confidence": probabilities[label],
                "probabilities": probabilities}

    def save(self, model_path: str | Path | None = None) -> None:
        if not self.is_fitted:
            raise RuntimeError("Cannot save an untrained model")
        path = Path(model_path) if model_path else self.model_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.feature_scaler is None:
            raise RuntimeError("Cannot save a new hybrid checkpoint without a fitted training feature scaler")
        torch.save({"model_state_dict": self.model.state_dict(),
                    "class_names": self.class_names, "sample_length": SAMPLE_LENGTH,
                    "feature_count": 16, "feature_scaler": self.feature_scaler,
                    "feature_scaler_method": "StandardScaler", "feature_scaler_fit_split": "train",
                    "dropout_disabled": self.disable_dropout}, path)
        self.model_path = path

    def load(self, model_path: str | Path) -> None:
        checkpoint = torch.load(Path(model_path), map_location=self.device, weights_only=False)
        self.class_names = tuple(checkpoint["class_names"])
        if checkpoint.get("sample_length", SAMPLE_LENGTH) != SAMPLE_LENGTH or checkpoint.get("feature_count", 16) != 16:
            raise ValueError("Checkpoint architecture does not match this implementation")
        self.disable_dropout = bool(checkpoint.get("dropout_disabled", self.disable_dropout))
        self.model = HybridNetwork(len(self.class_names), disable_dropout=self.disable_dropout).to(self.device)
        weights = checkpoint.get("model_state_dict", checkpoint.get("state_dict"))
        if weights is None:
            raise ValueError("Checkpoint does not contain model weights")
        self.model.load_state_dict(weights)
        saved_scaler = checkpoint.get("feature_scaler")
        if saved_scaler is None:
            self.feature_scaler = None
            warnings.warn("This checkpoint predates feature standardization. It will be loaded with raw features; "
                          "it is not equivalent to a newly trained standardized checkpoint.",
                          RuntimeWarning, stacklevel=2)
        else:
            mean = np.asarray(saved_scaler["mean"], dtype=np.float32)
            scale = np.asarray(saved_scaler["scale"], dtype=np.float32)
            if mean.shape != (16,) or scale.shape != (16,) or not np.all(np.isfinite(mean)) \
                    or not np.all(np.isfinite(scale)) or np.any(scale <= 0):
                raise ValueError("Checkpoint contains an invalid 16-feature scaler")
            self.feature_scaler = {**saved_scaler, "mean": mean.tolist(), "scale": scale.tolist()}
        self.model.eval()
        self.model_path = Path(model_path)
        self.is_fitted = True
