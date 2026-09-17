from pydantic import BaseModel
from typing import Optional, List, Dict, Any

class HealthResponse(BaseModel):
    status: str = "ok"

class ClassifyResponse(BaseModel):
    modulation: str
    confidence: float
    features: Optional[Dict[str, float]] = None

class ConstellationPoint(BaseModel):
    i: float
    q: float

class PsdPoint(BaseModel):
    freq: float
    psd: float

class ProcessResponse(BaseModel):
    modulation: str
    confidence: float
    baud_rate: Optional[float] = None
    snr: Optional[float] = None
    snr_db: Optional[float] = None
    center_frequency_hz: Optional[float] = None
    bandwidth_hz: Optional[float] = None
    bandwidth_3db_hz: Optional[float] = None
    num_samples: int
    duration_sec: float
    sample_rate: float
    waveform_data: List[float]
    constellation_data: Optional[List[ConstellationPoint]] = None
    psd_data: Optional[List[PsdPoint]] = None

class DecodeResponse(BaseModel):
    modulation: str
    confidence: float
    num_bits: int
    bit_string_preview: str
    hex_preview: str
    evm_db: Optional[float] = None
    evm_percent: Optional[float] = None
    fec_scheme: Optional[str] = None
    decoded_bits_count: int
    decoded_bits: List[int]

class AsyncJobResponse(BaseModel):
    job_id: str
    status: str
    message: str

class JobStatusResponse(BaseModel):
    job_id: str
    status: str  # "queued", "processing", "completed", "failed"
    progress: float  # 0.0 to 1.0
    stage: str
    created_at: float
    result: Optional[ProcessResponse] = None
    error: Optional[str] = None
