import numpy as np
from fastapi import APIRouter, UploadFile, File, Query, HTTPException
from ps26147_toolkit import classifier, parameter_extractor, demodulator, fec_decoders, deinterleaver
from api.schemas import DecodeResponse
from api.utils import load_signal_from_bytes

router = APIRouter()

@router.post("/", response_model=DecodeResponse)
async def decode_signal(
    file: UploadFile = File(...),
    fs: float = Query(1_000_000.0, description="Sampling rate in Hz if not in metadata"),
    fec_scheme: str = Query("none", description="FEC Scheme: 'none', 'viterbi', 'reed-solomon', 'concatenated', 'ldpc'"),
    auto_deinterleave: bool = Query(True, description="Automatically detect and apply deinterleaving")
):
    try:
        contents = await file.read()
        sig, sample_rate = load_signal_from_bytes(contents, file.filename or "", default_fs=fs)
        
        # 1. Classify modulation (CNN-authoritative with RF fallback)
        from ps26147_toolkit.cnn_runtime import predict_iq_array
        clf_res = predict_iq_array(sig, sample_rate_hz=sample_rate)
        mod = clf_res["modulation"]
        conf = float(clf_res["confidence"])

        # 2. Extract parameters (baud rate & center frequency)
        params = parameter_extractor.extract_signal_parameters(sig, fs=sample_rate, modulation=mod)
        baud_rate = float(params["baud_rate"])
        fc = float(params["center_frequency_hz"])

        # 3. Demodulate signal
        demod_res = demodulator.demodulate_signal(
            sig,
            fs=sample_rate,
            modulation=mod,
            center_freq=fc,
            baud_rate=baud_rate if baud_rate > 0 else None,
        )

        raw_bits = demod_res["bits"]
        llr = demod_res.get("llr")
        demodulated_bits = raw_bits.tolist()

        # 4. Optional FEC Decode
        if fec_scheme.lower() != "none" and len(raw_bits) > 0:
            fec_res = fec_decoders.decode_fec(raw_bits, scheme=fec_scheme, llr=llr)
            fec_bits_arr = fec_res["bits"]
            decoded_bits = fec_bits_arr.tolist()
        else:
            fec_bits_arr = raw_bits
            decoded_bits = raw_bits.tolist()

        # 5. Deinterleaving (auto-detect and deinterleave)
        deinterleaved_bits = None
        deinterleaved_bits_count = None
        deint_method = None
        deint_params = None
        deint_entropy = None
        deint_base_entropy = None

        if auto_deinterleave and len(decoded_bits) > 0:
            deint_res = deinterleaver.auto_detect_and_deinterleave(np.asarray(decoded_bits, dtype=np.uint8))
            deint_out = deint_res.get("bits")
            if deint_out is not None:
                deinterleaved_bits = deint_out.tolist()
                deinterleaved_bits_count = len(deinterleaved_bits)
            deint_method = deint_res.get("method")
            deint_params = deint_res.get("params")
            if "entropy" in deint_res and deint_res["entropy"] is not None:
                deint_entropy = float(deint_res["entropy"])
            if "baseline_entropy" in deint_res and deint_res["baseline_entropy"] is not None:
                deint_base_entropy = float(deint_res["baseline_entropy"])

        return DecodeResponse(
            modulation=mod,
            confidence=conf,
            num_bits=int(demod_res["num_bits"]),
            bit_string_preview=demod_res["bit_string_preview"],
            hex_preview=demod_res["hex_preview"],
            evm_db=float(demod_res["evm_db"]),
            evm_percent=float(demod_res["evm_percent"]),
            fec_scheme=fec_scheme,
            decoded_bits_count=len(decoded_bits),
            decoded_bits=decoded_bits,
            demodulated_bits=demodulated_bits,
            demodulated_bits_count=len(demodulated_bits),
            deinterleaved_bits=deinterleaved_bits,
            deinterleaved_bits_count=deinterleaved_bits_count,
            deinterleaver_method=deint_method,
            deinterleaver_params=deint_params,
            deinterleaver_entropy=deint_entropy,
            deinterleaver_baseline_entropy=deint_base_entropy,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Decoding error: {str(e)}")