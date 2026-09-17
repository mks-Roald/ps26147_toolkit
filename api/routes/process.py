from fastapi import APIRouter, UploadFile, File, Query, HTTPException
import numpy as np
from ps26147_toolkit import classifier, parameter_extractor, feature_extractor
from api.schemas import ProcessResponse, ConstellationPoint, PsdPoint
from api.utils import load_signal_from_bytes

router = APIRouter()

@router.post("/file", response_model=ProcessResponse)
async def process_file(
    file: UploadFile = File(...),
    fs: float = Query(1_000_000.0, description="Sampling rate in Hz if not in metadata")
):
    try:
        contents = await file.read()
        sig, sample_rate = load_signal_from_bytes(contents, file.filename or "", default_fs=fs)
        
        num_samples = len(sig)
        if num_samples == 0:
            raise HTTPException(status_code=400, detail="Empty signal file.")

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
            # Analytic signal for real input
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

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Signal processing error: {str(e)}")