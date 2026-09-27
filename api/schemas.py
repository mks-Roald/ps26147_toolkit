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

class WaterfallData(BaseModel):
    time: List[float]
    frequency: List[float]
    power_db: List[List[float]]

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
    waterfall_data: Optional[WaterfallData] = None

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
    decoded_hex: Optional[str] = None
    decoded_ascii: Optional[str] = None
    demodulated_bits: Optional[List[int]] = None
    demodulated_bits_count: Optional[int] = None
    deinterleaved_bits: Optional[List[int]] = None
    deinterleaved_bits_count: Optional[int] = None
    deinterleaver_method: Optional[str] = None
    deinterleaver_params: Optional[Dict[str, Any]] = None
    deinterleaver_entropy: Optional[float] = None
    deinterleaver_baseline_entropy: Optional[float] = None
    # Frame-sync fields (populated after demod, before de-interleave)
    sync_offset: Optional[int] = None
    sync_confidence: Optional[float] = None
    sync_method: Optional[str] = None

class CorrelatedFrame(BaseModel):
    start_bit: int
    end_bit: int
    frame_bits: List[int]
    payload_bits: List[int]
    correlation: float

class PreambleDiscoveryResult(BaseModel):
    discovered: bool
    estimated_frame_period: Optional[int] = None
    periodicity_strength: float = 0.0
    matched_standard_sync: Optional[str] = None
    standard_sync_confidence: Optional[float] = None
    candidate_preamble_bits: Optional[List[int]] = None
    candidate_preamble_hex: Optional[str] = None

class CorrelateResponse(BaseModel):
    status: str
    sync_found: bool
    peak_indices: List[int] = []
    num_frames: int = 0
    frames: List[CorrelatedFrame] = []
    max_correlation: float = 0.0
    is_inverted: bool = False
    detected_frame_length: Optional[int] = None
    sync_word_len: Optional[int] = None
    correlation_curve: Optional[List[float]] = None
    preamble_discovery: Optional[PreambleDiscoveryResult] = None
    num_bits: Optional[int] = None
    bits: Optional[List[int]] = None

class CorrelateRequest(BaseModel):
    bits: Optional[List[int]] = None
    hex_string: Optional[str] = None
    bit_string: Optional[str] = None
    sync_word: Optional[str] = None
    sync_word_hex: Optional[str] = None
    sync_word_bits: Optional[List[int]] = None
    frame_length: Optional[int] = None
    threshold: float = 0.80
    tolerate_inverted: bool = True
    auto_discover: bool = True

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
