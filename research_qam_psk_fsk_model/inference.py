"""Research-only single-file and batch inference for this checkpoint format."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from .data.multimodal_dataset import load_capture, read_capture_metadata, crop_window
from .data.psd_utils import PSDStandardizer
from .data.representations import build_representations
from .labels import CLASS_NAMES, FEATURE_NAMES, INPUT_LENGTH
from .model.multimodal_model import MultimodalAMR
from .training.train_multimodal import _restore_feature_scaler


def load_model(checkpoint_path: str | Path, device: str | None = None):
    selected = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if selected.type == "cuda":
        try:
            torch.cuda.init()
        except Exception:
            if device is not None:
                raise
            selected = torch.device("cpu")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("format") != "research_qam_psk_fsk_multimodal_v1":
        raise ValueError("Unsupported or incomplete multimodal checkpoint")
    if payload.get("class_names") != list(CLASS_NAMES) or payload.get("feature_names") != list(FEATURE_NAMES):
        raise ValueError("Checkpoint class/feature ordering does not match this package")
    arch = payload.get("architecture_config", {})
    model = MultimodalAMR(feature_count=payload["feature_count"], num_classes=len(CLASS_NAMES),
        iq_dim=arch.get("iq_embedding_dim",128), aux_dim=arch.get("auxiliary_embedding_dim",64),
        fused_dim=arch.get("fused_embedding_dim",128), dropout=arch.get("dropout",0.25))
    model.load_state_dict(payload["model_state_dict"]); model.to(selected).eval()
    feature_scaler = _restore_feature_scaler(payload["feature_scaler"])
    psd_scaler = PSDStandardizer.from_state_dict(payload["psd_scaler"])
    return model, feature_scaler, psd_scaler, payload, selected


@torch.inference_mode()
def predict_file(checkpoint_path: str | Path, iq_path: str | Path,
                 *, device: str | None = None, fallback_sample_rate_hz: float | None = None) -> dict[str, Any]:
    model, feature_scaler, psd_scaler, checkpoint, selected = load_model(checkpoint_path, device)
    return _predict_loaded(model, feature_scaler, psd_scaler, checkpoint, selected,
                           iq_path, fallback_sample_rate_hz)


@torch.inference_mode()
def _predict_loaded(model, feature_scaler, psd_scaler, checkpoint, selected,
                    iq_path, fallback_sample_rate_hz):
    path = Path(iq_path).resolve()
    meta = read_capture_metadata(path, {})
    signal, fs = load_capture(path, meta, fallback_sample_rate_hz)
    crop = crop_window(signal, "val", INPUT_LENGTH)
    reps = build_representations(crop, fs, feature_scaler, psd_scaler)
    inputs = [torch.from_numpy(reps[k]).unsqueeze(0).to(selected)
              for k in ("iq","polar","psk","ifreq","psd","features")]
    logits, aux = model(*inputs, return_aux=True)
    probabilities = torch.softmax(logits.float(), dim=1)[0].cpu().numpy()
    best = int(np.argmax(probabilities))
    weights = aux["weights"].detach().cpu().numpy()
    contrib = aux["contributions"]
    branch_norm = {k: float(v[0].norm().cpu()) for k,v in contrib.items()}
    return {"prediction": CLASS_NAMES[best], "confidence": float(probabilities[best]),
            "probabilities": {name: float(probabilities[i]) for i,name in enumerate(CLASS_NAMES)},
            "branch_weights": {name: float(weights[i]) for i,name in enumerate(model.fusion.NAMES)},
            "branch_contribution_l2": branch_norm, "sample_rate_hz": fs,
            "sample_id": path.name, "checkpoint_seed": checkpoint.get("training_seed"),
            "device": str(selected)}


def predict_batch(checkpoint_path: str | Path, iq_paths: list[str | Path], **kwargs) -> list[dict[str, Any]]:
    """Future convenience batch API; each file is still loaded lazily."""
    model, feature_scaler, psd_scaler, checkpoint, selected = load_model(
        checkpoint_path, kwargs.get("device"))
    fallback = kwargs.get("fallback_sample_rate_hz")
    return [_predict_loaded(model, feature_scaler, psd_scaler, checkpoint, selected, path, fallback)
            for path in iq_paths]
