from fastapi import APIRouter, UploadFile, File, Query, HTTPException
from ps26147_toolkit import classifier, parameter_extractor, demodulator, fec_decoders
from api.schemas import DecodeResponse
from api.utils import load_signal_from_bytes

router = APIRouter()

@router.post("/", response_model=DecodeResponse)
async def decode_signal(
    file: UploadFile = File(...),
    fs: float = Query(1_000_000.0, description="Sampling rate in Hz if not in metadata"),
    fec_scheme: str = Query("none", description="FEC Scheme: 'none', 'viterbi', 'reed-solomon', 'concatenated', 'ldpc'")
):
    try:
        contents = await file.read()
        sig, sample_rate = load_signal_from_bytes(contents, file.filename or "", default_fs=fs)
        
        # 1. Classify modulation
        clf = classifier.ModulationClassifier()
        clf_res = clf.predict_with_confidence(sig, fs=sample_rate)
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

        # 4. Optional FEC Decode
        if fec_scheme.lower() != "none" and len(raw_bits) > 0:
            fec_res = fec_decoders.decode_fec(raw_bits, scheme=fec_scheme, llr=llr)
            decoded_bits = fec_res["bits"].tolist()
        else:
            decoded_bits = raw_bits.tolist()

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
            decoded_bits=decoded_bits[:512],
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Decoding error: {str(e)}")