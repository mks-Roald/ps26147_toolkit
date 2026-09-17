from fastapi import APIRouter, UploadFile, File, Query, HTTPException, BackgroundTasks
import numpy as np
from ps26147_toolkit import classifier, parameter_extractor, feature_extractor
from api.schemas import (
    ProcessResponse,
    ConstellationPoint,
    PsdPoint,
    AsyncJobResponse,
    JobStatusResponse,
)
from api.utils import load_signal_from_bytes
from api.task_manager import task_manager

router = APIRouter()

def _process_signal_core(contents: bytes, filename: str, sample_rate_hint: float) -> ProcessResponse:
    """Core synchronous processing pipeline reused by both sync and async handlers."""
    sig, sample_rate = load_signal_from_bytes(contents, filename or "", default_fs=sample_rate_hint)
    
    num_samples = len(sig)
    if num_samples == 0:
        raise ValueError("Empty signal file.")

    duration_sec = float(num_samples / sample_rate)

    # 1. Modulation Classification
    clf = classifier.ModulationClassifier()
    clf_res = clf.predict_with_confidence(sig, fs=sample_rate)
    mod = clf_res["modulation"]
    conf = float(clf_res["confidence"])

    # 2. Extract Signal Parameters
    params = parameter_extractor.extract_signal_parameters(
        sig, fs=sample_rate, modulation=mod
    )

    # 3. Waveform data (real part, capped at 1000 samples)
    real_wave = np.real(sig)
    step = max(1, len(real_wave) // 1000)
    waveform_subset = real_wave[::step][:1000].tolist()

    # 4. Constellation data (up to 400 points)
    constellation_pts = []
    if np.iscomplexobj(sig):
        c_step = max(1, len(sig) // 400)
        c_subset = sig[::c_step][:400]
        constellation_pts = [
            ConstellationPoint(i=float(pt.real), q=float(pt.imag))
            for pt in c_subset
        ]
    else:
        from scipy.signal import hilbert
        analytic_sig = hilbert(sig)
        c_step = max(1, len(analytic_sig) // 400)
        c_subset = analytic_sig[::c_step][:400]
        constellation_pts = [
            ConstellationPoint(i=float(pt.real), q=float(pt.imag))
            for pt in c_subset
        ]

    # 5. PSD Data for spectrum chart
    freqs, psd = feature_extractor.compute_psd(sig, sample_rate, nperseg=min(512, max(32, len(sig))))
    psd_db = 10.0 * np.log10(np.maximum(psd, 1e-15))
    psd_step = max(1, len(freqs) // 150)
    psd_pts = [
        PsdPoint(freq=float(f), psd=float(p))
        for f, p in zip(freqs[::psd_step], psd_db[::psd_step])
    ]

    return ProcessResponse(
        modulation=mod,
        confidence=conf,
        baud_rate=float(params["baud_rate"]),
        snr=float(params["snr_db"]),
        snr_db=float(params["snr_db"]),
        center_frequency_hz=float(params["center_frequency_hz"]),
        bandwidth_hz=float(params["bandwidth_hz"]),
        bandwidth_3db_hz=float(params["bandwidth_3db_hz"]),
        num_samples=num_samples,
        duration_sec=duration_sec,
        sample_rate=float(sample_rate),
        waveform_data=waveform_subset,
        constellation_data=constellation_pts,
        psd_data=psd_pts,
    )


def _async_process_worker(job_id: str, contents: bytes, filename: str, fs: float):
    """Background worker executing signal processing stages with live progress reporting."""
    try:
        task_manager.update_job(job_id, status="processing", progress=0.15, stage="Parsing file format & samples…")
        
        task_manager.update_job(job_id, progress=0.40, stage="Classifying modulation & spectral cumulants…")
        
        task_manager.update_job(job_id, progress=0.70, stage="Estimating Baud rate, SNR, & occupied bandwidth…")
        
        res = _process_signal_core(contents, filename, fs)
        
        task_manager.update_job(job_id, progress=0.90, stage="Synthesizing waveform, PSD, & constellation points…")
        
        task_manager.update_job(
            job_id,
            status="completed",
            progress=1.0,
            stage="Analysis complete",
            result=res,
        )
    except Exception as e:
        task_manager.update_job(
            job_id,
            status="failed",
            progress=1.0,
            stage="Analysis failed",
            error=str(e),
        )


@router.post("/file", response_model=ProcessResponse)
async def process_file_sync(
    file: UploadFile = File(...),
    fs: float = Query(1_000_000.0, description="Sampling rate in Hz if not in metadata")
):
    """Synchronous file processing endpoint."""
    try:
        contents = await file.read()
        return _process_signal_core(contents, file.filename or "", fs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Signal processing error: {str(e)}")


@router.post("/async", response_model=AsyncJobResponse)
async def process_file_async(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    fs: float = Query(1_000_000.0, description="Sampling rate in Hz if not in metadata")
):
    """Asynchronous file processing endpoint: returns job ID immediately and processes in background."""
    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Empty signal file.")

    job_id = task_manager.create_job()
    background_tasks.add_task(_async_process_worker, job_id, contents, file.filename or "", fs)

    return AsyncJobResponse(
        job_id=job_id,
        status="queued",
        message="Job queued for background processing",
    )


@router.get("/status/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Poll execution status and progress of an asynchronous processing job."""
    job = task_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    
    return JobStatusResponse(
        job_id=job["job_id"],
        status=job["status"],
        progress=job["progress"],
        stage=job["stage"],
        created_at=job["created_at"],
        result=job.get("result"),
        error=job.get("error"),
    )