import logging
from typing import Optional, List, Dict, Any
import numpy as np
from fastapi import APIRouter, UploadFile, File, Query, HTTPException
from ps26147_toolkit import classifier, parameter_extractor, demodulator, fec_decoders, deinterleaver, correlator
from api.schemas import DecodeResponse
from api.utils import load_signal_from_bytes

logger = logging.getLogger(__name__)

router = APIRouter()


def _resolve_sync_word(sync_word: Optional[str] = None) -> Optional[np.ndarray]:
    """Resolve a sync word pattern from hex, binary, or standard name."""
    if not sync_word:
        return None

    sw = sync_word.strip()
    if sw in correlator.STANDARD_SYNC_WORDS:
        return correlator.STANDARD_SYNC_WORDS[sw]

    for k, v in correlator.STANDARD_SYNC_WORDS.items():
        if k.lower() == sw.lower():
            return v

    clean = sw.replace("0x", "").replace("0X", "").replace(" ", "")
    if all(c in "0123456789abcdefABCDEF" for c in clean) and len(clean) % 2 == 0 and not all(c in "01" for c in clean):
        try:
            return correlator.hex_to_bits(clean)
        except Exception:
            pass

    if all(c in "01" for c in sw) and len(sw) > 0:
        return np.array([int(c) for c in sw], dtype=np.uint8)

    try:
        return correlator.hex_to_bits(sw)
    except Exception:
        pass

    return None


@router.post("/", response_model=DecodeResponse)
async def decode_signal(
    file: UploadFile = File(...),
    fs: float = Query(1_000_000.0, description="Sampling rate in Hz if not in metadata"),
    fec_scheme: str = Query("none", description="FEC Scheme: 'none', 'viterbi', 'reed-solomon', 'concatenated', 'ldpc'"),
    auto_deinterleave: bool = Query(True, description="Automatically detect and apply deinterleaving"),
    sync_word: Optional[str] = Query(None, description="Sync word in hex (e.g. '1ACFFC1D' or '0x47') or standard name"),
    auto_detect_sync: bool = Query(True, description="Automatically detect preamble/sync word when sync_word is not given"),
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
        demodulated_bits = raw_bits.tolist()

        # Guard: verify LLR length matches raw_bits length before slicing (regression tripwire for compute_soft_llr)
        if llr is not None and len(llr) != len(raw_bits):
            raise HTTPException(
                status_code=500,
                detail=f"Demodulation LLR length mismatch: len(llr)={len(llr)} != len(raw_bits)={len(raw_bits)}. Check compute_soft_llr().",
            )

        # 4. Frame Synchronization
        sync_offset: Optional[int] = None
        sync_confidence: float = 0.0
        sync_method: str = "none"

        target_sync: Optional[np.ndarray] = None
        if sync_word:
            target_sync = _resolve_sync_word(sync_word=sync_word)
            if target_sync is not None:
                sync_method = "sync_word"
            else:
                logger.warning(f"Could not parse provided sync_word: {sync_word}")

        if target_sync is None and auto_detect_sync and len(raw_bits) > 0:
            disc = correlator.auto_discover_preamble(raw_bits)
            if disc.get("discovered", False):
                if disc.get("matched_standard_sync") and disc["matched_standard_sync"] in correlator.STANDARD_SYNC_WORDS:
                    target_sync = correlator.STANDARD_SYNC_WORDS[disc["matched_standard_sync"]]
                    sync_method = f"auto_{disc['matched_standard_sync']}"
                elif disc.get("candidate_preamble_bits") is not None and len(disc["candidate_preamble_bits"]) >= 7:
                    target_sync = disc["candidate_preamble_bits"]
                    sync_method = "auto_detect"

        if target_sync is not None and len(raw_bits) >= len(target_sync):
            sync_res = correlator.frame_synchronize(raw_bits, target_sync)
            if sync_res.get("sync_found", False) and len(sync_res.get("peak_indices", [])) > 0:
                first_peak = int(sync_res["peak_indices"][0])
                sync_offset = first_peak
                sync_confidence = float(sync_res.get("max_correlation", 0.0))

                # Single slice operation applied to both raw_bits and llr
                sync_slice = slice(sync_offset, None)
                raw_bits = raw_bits[sync_slice]
                if llr is not None:
                    llr = llr[sync_slice]
                    demod_res["llr"] = llr
            else:
                logger.warning("Frame synchronization failed: no confident match found. Proceeding unaligned.")
                sync_offset = None
                sync_confidence = 0.0
                sync_method = "none"
        else:
            if sync_word or (auto_detect_sync and len(raw_bits) > 0):
                logger.warning("Frame synchronization: no sync pattern found or bitstream too short. Proceeding unaligned.")
            sync_offset = None
            sync_confidence = 0.0
            sync_method = "none"

        # 5. Optional FEC Decode
        if fec_scheme.lower() != "none" and len(raw_bits) > 0:
            fec_res = fec_decoders.decode_fec(raw_bits, scheme=fec_scheme, llr=llr)
            fec_bits_arr = fec_res["bits"]
            decoded_bits = fec_bits_arr.tolist()
        else:
            fec_bits_arr = raw_bits
            decoded_bits = raw_bits.tolist()

        # 6. Hex and ASCII representation of decoded output
        decoded_hex = ""
        decoded_ascii = ""
        if len(decoded_bits) > 0:
            f_byte_arr = np.packbits(np.asarray(decoded_bits, dtype=np.uint8))
            f_bytes = f_byte_arr.tobytes()
            decoded_hex = " ".join(f"{b:02X}" for b in f_bytes)
            decoded_ascii = "".join(chr(b) if 32 <= b <= 126 else "." for b in f_bytes)

        # 7. Deinterleaving (auto-detect and deinterleave)
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
            decoded_hex=decoded_hex,
            decoded_ascii=decoded_ascii,
            demodulated_bits=demodulated_bits,
            demodulated_bits_count=len(demodulated_bits),
            deinterleaved_bits=deinterleaved_bits,
            deinterleaved_bits_count=deinterleaved_bits_count,
            deinterleaver_method=deint_method,
            deinterleaver_params=deint_params,
            deinterleaver_entropy=deint_entropy,
            deinterleaver_baseline_entropy=deint_base_entropy,
            sync_offset=sync_offset,
            sync_confidence=sync_confidence,
            sync_method=sync_method,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Decoding error: {str(e)}")