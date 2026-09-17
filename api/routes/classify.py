from fastapi import APIRouter, UploadFile, File
from ps26147_toolkit import classifier
import numpy as np

router = APIRouter()

@router.post("/")
async def classify_signal(file: UploadFile = File(...)):
    contents = await file.read()
    signal = np.frombuffer(contents, dtype=np.int16).astype(np.float32)
    result = classifier.ModulationClassifier().predict_with_confidence(signal)
    return {
        "modulation": result["modulation"],
        "confidence": result["confidence"]
    }