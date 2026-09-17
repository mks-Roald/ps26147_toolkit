from fastapi import APIRouter, UploadFile, File, Query, HTTPException
from ps26147_toolkit import classifier
from api.schemas import ClassifyResponse
from api.utils import load_signal_from_bytes
import numpy as np

router = APIRouter()

FEATURE_NAMES = [
    "c20", "c21", "c40", "c41", "c42",
    "gamma_max", "sigma_ap", "sigma_dp", "sigma_aa", "sigma_af",
    "p_spectrum", "kurtosis_amp"
]

@router.post("/", response_model=ClassifyResponse)
async def classify_signal(
    file: UploadFile = File(...),
    fs: float = Query(1_000_000.0, description="Sampling rate in Hz if not in metadata")
):
    try:
        contents = await file.read()
        sig, sample_rate = load_signal_from_bytes(contents, file.filename or "", default_fs=fs)
        
        clf = classifier.ModulationClassifier()
        result = clf.predict_with_confidence(sig, fs=sample_rate)
        
        # Format features safely as key-value dict
        raw_feats = result.get("features")
        features_dict = {}
        if isinstance(raw_feats, dict):
            features_dict = {k: float(v) for k, v in raw_feats.items() if isinstance(v, (int, float, np.floating))}
        elif isinstance(raw_feats, (list, np.ndarray)):
            features_dict = {
                (FEATURE_NAMES[i] if i < len(FEATURE_NAMES) else f"feat_{i}"): float(v)
                for i, v in enumerate(raw_feats)
                if np.isfinite(v)
            }
        
        # Include cumulants if available
        if "cumulants" in result and isinstance(result["cumulants"], dict):
            for k, v in result["cumulants"].items():
                if isinstance(v, (int, float, complex, np.number)):
                    features_dict[f"cumulant_{k}"] = float(np.abs(v))

        return ClassifyResponse(
            modulation=result["modulation"],
            confidence=float(result["confidence"]),
            features=features_dict
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification error: {str(e)}")