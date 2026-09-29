from fastapi import APIRouter, UploadFile, File, Query, HTTPException, BackgroundTasks
import numpy as np
from ps26147_toolkit import classifier, parameter_extractor, feature_extractor
from ps26147_toolkit.analysis import analyze_signal
from api.schemas import (
    ProcessResponse,
    ConstellationPoint,
    PsdPoint,
    WaterfallData,
    AsyncJobResponse,
    JobStatusResponse,
)
from api.utils import load_signal_from_bytes
from api.task_manager import task_manager

router = APIRouter()

def _process_signal_core(contents: bytes, filename: str, sample_rate_hint: float) -> ProcessResponse:
    """Core synchronous processing pipeline reused by both sync and async handlers."""
    sig, sample_rate = load_signal_from_bytes(
        contents, filename or "", default_fs=sample_rate_hint, target_fs=sample_rate_hint
    )
    
    num_samples = len(sig)
    if num_samples == 0:
        raise ValueError("Empty signal file.")

    duration_sec = float(num_samples / sample_rate)

    # 1. Modulation Classification
    analysis = analyze_signal(sig, sample_rate)
    clf_res = analysis.classification
    mod = clf_res["modulation"]
    conf = float(clf_res["confidence"])

    # 2. Extract Signal Parameters
    params = analysis.parameters

    # 3. Waveform data (real part, capped at 1000 samples)
    real_wave = np.real(sig)
    step = max(1, len(real_wave) // 1000)
    waveform_subset = real_wave[::step][:1000].tolist()

    # 4. Constellation data (up to 400 points)
    constellation_pts = []
    if np.iscomplexobj(analysis.baseband_signal):
        symbols = analysis.baseband_signal
        c_step = max(1, len(symbols) // 400)
        c_subset = symbols[::c_step][:400]
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

    # 6. Spectrogram / Waterfall Data (downsampled to ~80x80 bins)
    waterfall_obj = None
    try:
        nperseg = min(256, max(16, len(sig)))
        noverlap = nperseg // 2
        t_spec, f_spec, Sxx_db = feature_extractor.compute_spectrogram(
            sig,
            fs=sample_rate,
            nperseg=nperseg,
            noverlap=noverlap,
        )
        if len(f_spec) > 80:
            f_idx = np.round(np.linspace(0, len(f_spec) - 1, 80)).astype(int)
            f_spec = f_spec[f_idx]
            Sxx_db = Sxx_db[f_idx, :]

        if len(t_spec) > 80:
            t_idx = np.round(np.linspace(0, len(t_spec) - 1, 80)).astype(int)
            t_spec = t_spec[t_idx]
            Sxx_db = Sxx_db[:, t_idx]

        waterfall_obj = WaterfallData(
            time=[round(float(x), 6) for x in t_spec],
            frequency=[round(float(y), 2) for y in f_spec],
            power_db=[[round(float(v), 2) for v in row] for row in Sxx_db],
        )
    except Exception:
        waterfall_obj = None

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
        waterfall_data=waterfall_obj,
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
    fs: float = Query(1_000_000.0, description="Target WAV sample rate; default sample rate for IQ data")
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
    fs: float = Query(1_000_000.0, description="Target WAV sample rate; default sample rate for IQ data")
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
