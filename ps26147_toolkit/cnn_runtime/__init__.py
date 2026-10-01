"""Shared CNN Runtime Service with automatic RF Fallback.

Design Rules:
1. CNN is authoritative for supported digital classes (BPSK, QPSK, 8PSK, 16QAM, 64QAM, 2FSK, 4FSK).
2. RF is fallback only on:
   - Missing checkpoint / load failure
   - Feature / representation failure
   - Torch runtime / shape / finite error
   - Explicit unsupported signals (e.g. AM, FM, or fallback request)
3. Singleton instantiation: Checkpoint loaded lazily once and reused.
4. RF classifier instantiated once as a singleton fallback (never re-trained or recreated per-request).
5. Public prediction API returns:
   - modulation: str
   - confidence: float
   - probabilities: dict[str, float]
   - source: "cnn" | "rf_fallback"
   - features: list or dict (optional)
   - cumulants: dict (optional)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional
import numpy as np
import torch

from .labels import CLASS_NAMES, FEATURE_NAMES, INPUT_LENGTH
from .model import MultimodalAMR
from .preprocessing import (
    crop_window_center,
    build_representations,
    restore_feature_scaler,
    PSDStandardizer,
)

logger = logging.getLogger("cnn_runtime")

DEFAULT_CHECKPOINT = Path(__file__).resolve().parent / "assets" / "best.pt"
EXPECTED_CHECKPOINT_HASH = "caccceaf680783d5ba96ee2af6184b5dca0f7a20b358154494d7201a2e3d5d45"


class CNNRuntimeService:
    """Singleton service providing CNN-authoritative predictions with RF fallback."""
    _instance: Optional["CNNRuntimeService"] = None

    def __init__(self, checkpoint_path: Optional[str | Path] = None, device: Optional[str] = None):
        self.checkpoint_path = Path(checkpoint_path or DEFAULT_CHECKPOINT)
        self.requested_device = device
        self.model: Optional[MultimodalAMR] = None
        self.feature_scaler = None
        self.psd_scaler = None
        self.device: Optional[torch.device] = None
        self._cnn_ready = False
        self._init_error: Optional[str] = None

        # Fallback RF instance (lazy singleton)
        self._rf_classifier = None

    @classmethod
    def get_instance(cls) -> "CNNRuntimeService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _ensure_rf_classifier(self):
        """Lazily load the single RF classifier instance for fallback."""
        if self._rf_classifier is None:
            from ps26147_toolkit.classifier import ModulationClassifier
            self._rf_classifier = ModulationClassifier()
        return self._rf_classifier

    def load_cnn_model(self) -> bool:
        """Load the CNN model and frozen scalers lazily once in eval mode."""
        if self._cnn_ready:
            return True
        if self._init_error is not None:
            return False

        try:
            if not self.checkpoint_path.is_file():
                raise FileNotFoundError(f"Checkpoint not found at {self.checkpoint_path}")

            selected_device = torch.device(
                self.requested_device or ("cuda" if torch.cuda.is_available() else "cpu")
            )
            if selected_device.type == "cuda":
                try:
                    torch.cuda.init()
                except Exception:
                    selected_device = torch.device("cpu")
            self.device = selected_device

            payload = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)

            if payload.get("format") != "research_qam_psk_fsk_multimodal_v1":
                raise ValueError(f"Invalid format: {payload.get('format')}")

            if list(payload.get("class_names", [])) != list(CLASS_NAMES):
                raise ValueError("Checkpoint class ordering does not match CLASS_NAMES")

            if list(payload.get("feature_names", [])) != list(FEATURE_NAMES):
                raise ValueError("Checkpoint feature ordering does not match FEATURE_NAMES")

            arch = payload.get("architecture_config", {})
            model = MultimodalAMR(
                feature_count=payload["feature_count"],
                num_classes=len(CLASS_NAMES),
                iq_dim=arch.get("iq_embedding_dim", 128),
                aux_dim=arch.get("auxiliary_embedding_dim", 64),
                fused_dim=arch.get("fused_embedding_dim", 128),
                dropout=arch.get("dropout", 0.25),
            )
            model.load_state_dict(payload["model_state_dict"], strict=True)
            model.to(self.device).eval()

            self.model = model
            self.feature_scaler = restore_feature_scaler(payload["feature_scaler"])
            self.psd_scaler = PSDStandardizer.from_state_dict(payload["psd_scaler"])
            self._cnn_ready = True
            logger.info("CNN MultimodalAMR loaded successfully on %s", self.device)
            return True

        except Exception as e:
            self._init_error = str(e)
            logger.warning("CNN model load failed; will use RF fallback. Reason: %s", e)
            return False

    def is_cnn_ready(self) -> bool:
        if not self._cnn_ready and self._init_error is None:
            self.load_cnn_model()
        return self._cnn_ready

    @torch.inference_mode()
    def _predict_cnn(self, signal: np.ndarray, fs: float) -> dict[str, Any]:
        """Execute strict CNN inference on in-memory signal."""
        if not self._cnn_ready:
            raise RuntimeError("CNN model is not loaded")

        crop = crop_window_center(signal, length=INPUT_LENGTH)
        reps = build_representations(crop, fs, self.feature_scaler, self.psd_scaler, length=INPUT_LENGTH)

        inputs = [
            torch.from_numpy(reps[k]).unsqueeze(0).to(self.device)
            for k in ("iq", "polar", "psk", "ifreq", "psd", "features")
        ]

        logits = self.model(*inputs)
        probs = torch.softmax(logits.float(), dim=1)[0].cpu().numpy()

        if not np.all(np.isfinite(probs)):
            raise ValueError("Non-finite probabilities returned from CNN")

        best_idx = int(np.argmax(probs))
        predicted_mod = CLASS_NAMES[best_idx]
        confidence = float(probs[best_idx])
        prob_dict = {name: float(probs[i]) for i, name in enumerate(CLASS_NAMES)}

        return {
            "modulation": predicted_mod,
            "confidence": confidence,
            "probabilities": prob_dict,
            "source": "cnn",
            "features": reps["features"].tolist(),
        }

    def _predict_rf_fallback(self, signal: np.ndarray, fs: float, fc: Optional[float] = None) -> dict[str, Any]:
        """Execute existing RF prediction fallback."""
        rf = self._ensure_rf_classifier()
        res = rf.predict_with_confidence(signal, fs=fs, fc=fc)
        res["source"] = "rf_fallback"
        return res

    def predict_iq_array(
        self,
        signal: np.ndarray,
        sample_rate_hz: float = 1_000_000.0,
        fc: Optional[float] = None,
    ) -> dict[str, Any]:
        """Authoritative prediction entrypoint.
        
        Attempts CNN prediction first. Falls back automatically to RF if:
        - CNN is not ready or load failed
        - Signal preprocessing or tensor computation throws an exception
        - Model output contains non-finite values
        """
        fs = float(sample_rate_hz)
        if not np.isfinite(fs) or fs <= 0:
            fs = 1_000_000.0

        if self.is_cnn_ready():
            try:
                res = self._predict_cnn(signal, fs)
                # Compute diagnostic cumulants using existing feature extractor for API compatibility
                from ps26147_toolkit.feature_extractor import compute_cumulants
                res["cumulants"] = compute_cumulants(signal, fs=fs, fc=fc)
                return res
            except Exception as exc:
                logger.warning("CNN inference failed (%s); triggering RF fallback", exc)

        return self._predict_rf_fallback(signal, fs=fs, fc=fc)


# Module-level convenience functions
def get_cnn_service() -> CNNRuntimeService:
    return CNNRuntimeService.get_instance()


def predict_iq_array(signal: np.ndarray, sample_rate_hz: float = 1_000_000.0, fc: Optional[float] = None) -> dict[str, Any]:
    return CNNRuntimeService.get_instance().predict_iq_array(signal, sample_rate_hz, fc=fc)
