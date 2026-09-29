import numpy as np
from fastapi import APIRouter, UploadFile, File, Query, HTTPException, Body
from typing import Optional, List, Dict, Any

from ps26147_toolkit import classifier, parameter_extractor, demodulator, fec_decoders, deinterleaver, correlator
from api.schemas import (
    CorrelateResponse,
    CorrelatedFrame,
    PreambleDiscoveryResult,
    CorrelateRequest,
)
from api.utils import load_signal_from_bytes

router = APIRouter()


def _resolve_sync_word(
    sync_word: Optional[str] = None,
    sync_word_hex: Optional[str] = None,
    sync_word_bits: Optional[List[int]] = None,
) -> Optional[np.ndarray]:
    """Resolve a sync word pattern from bits, hex, or standard name."""
    if sync_word_bits is not None and len(sync_word_bits) > 0:
        return np.asarray(sync_word_bits, dtype=np.uint8)

    if sync_word_hex:
        return correlator.hex_to_bits(sync_word_hex)

    if sync_word:
        sw = sync_word.strip()
        # Direct dictionary match
        if sw in correlator.STANDARD_SYNC_WORDS:
            return correlator.STANDARD_SYNC_WORDS[sw]

        # Case-insensitive match against standard dictionary
        for k, v in correlator.STANDARD_SYNC_WORDS.items():
            if k.lower() == sw.lower():
                return v

        # Hexadecimal string check (e.g., 0x1ACFFC1D or 1ACFFC1D)
        clean = sw.replace("0x", "").replace("0X", "").replace(" ", "")
        if all(c in "0123456789abcdefABCDEF" for c in clean) and len(clean) % 2 == 0 and not all(c in "01" for c in clean):
            try:
                return correlator.hex_to_bits(clean)
            except Exception:
                pass

        # Binary string check (e.g. "1110010")
        if all(c in "01" for c in sw) and len(sw) > 0:
            return np.array([int(c) for c in sw], dtype=np.uint8)

    return None


def _correlate_bitstream_core(
    bits: np.ndarray,
    sync_word: Optional[str] = None,
    sync_word_hex: Optional[str] = None,
    sync_word_bits: Optional[List[int]] = None,
    frame_length: Optional[int] = None,
    threshold: float = 0.80,
    tolerate_inverted: bool = True,
    auto_discover: bool = True,
) -> CorrelateResponse:
    """Core correlation, preamble discovery, and frame synchronization logic."""
    if len(bits) == 0:
        return CorrelateResponse(
            status="Empty bitstream",
            sync_found=False,
            peak_indices=[],
            num_frames=0,
            frames=[],
            max_correlation=0.0,
            is_inverted=False,
            num_bits=0,
            bits=[],
        )

    # 1. Automatic Preamble Discovery (if requested)
    preamble_disc_res = None
    target_sync = _resolve_sync_word(sync_word, sync_word_hex, sync_word_bits)

    if auto_discover or target_sync is None:
        disc = correlator.auto_discover_preamble(bits)
        cand_bits = (
            disc["candidate_preamble_bits"].tolist()
            if disc.get("candidate_preamble_bits") is not None
            else None
        )
        preamble_disc_res = PreambleDiscoveryResult(
            discovered=bool(disc.get("discovered", False)),
            estimated_frame_period=disc.get("estimated_frame_period"),
            periodicity_strength=float(disc.get("periodicity_strength", 0.0)),
            matched_standard_sync=disc.get("matched_standard_sync"),
            standard_sync_confidence=(
                float(disc["standard_sync_confidence"])
                if disc.get("standard_sync_confidence") is not None
                else None
            ),
            candidate_preamble_bits=cand_bits,
            candidate_preamble_hex=disc.get("candidate_preamble_hex"),
        )

        # If user did not provide a sync word, adopt the auto-discovered one
        if target_sync is None:
            if disc.get("matched_standard_sync") and disc["matched_standard_sync"] in correlator.STANDARD_SYNC_WORDS:
                target_sync = correlator.STANDARD_SYNC_WORDS[disc["matched_standard_sync"]]
            elif disc.get("candidate_preamble_bits") is not None and len(disc["candidate_preamble_bits"]) >= 7:
                target_sync = disc["candidate_preamble_bits"]

    # If still no sync word, default to Barker-13 for baseline correlation
    if target_sync is None:
        target_sync = correlator.STANDARD_SYNC_WORDS["Barker-13"]

    # 2. Frame Synchronization
    sync_res = correlator.frame_synchronize(
        bits,
        target_sync,
        frame_length=frame_length,
        threshold=threshold,
        tolerate_inverted=tolerate_inverted,
    )

    # 3. Format frames
    formatted_frames: List[CorrelatedFrame] = []
    for f in sync_res.get("frames", []):
        f_bits = f["frame_bits"]
        p_bits = f["payload_bits"]
        formatted_frames.append(
            CorrelatedFrame(
                start_bit=int(f["start_bit"]),
                end_bit=int(f["end_bit"]),
                frame_bits=f_bits.tolist() if isinstance(f_bits, np.ndarray) else list(f_bits),
                payload_bits=p_bits.tolist() if isinstance(p_bits, np.ndarray) else list(p_bits),
                correlation=float(f["correlation"]),
            )
        )

    # 4. Correlation curve formatting
    curve = sync_res.get("correlation_curve")
    corr_curve_list: Optional[List[float]] = None
    if curve is not None and len(curve) > 0:
        if len(curve) > 1000:
            step = len(curve) // 1000
            curve_sub = curve[::step][:1000]
            corr_curve_list = [round(float(v), 4) for v in curve_sub]
        else:
            corr_curve_list = [round(float(v), 4) for v in curve]

    return CorrelateResponse(
        status=str(sync_res.get("status", "Completed")),
        sync_found=bool(sync_res.get("sync_found", False)),
        peak_indices=[int(p) for p in sync_res.get("peak_indices", [])],
        num_frames=int(sync_res.get("num_frames", len(formatted_frames))),
        frames=formatted_frames,
        max_correlation=float(sync_res.get("max_correlation", 0.0)),
        is_inverted=bool(sync_res.get("is_inverted", False)),
        detected_frame_length=(
            int(sync_res["detected_frame_length"])
            if sync_res.get("detected_frame_length") is not None
            else None
        ),
        sync_word_len=int(len(target_sync)),
        correlation_curve=corr_curve_list,
        preamble_discovery=preamble_disc_res,
        num_bits=len(bits),
        bits=bits.tolist(),
    )


@router.get("/sync-words")
def get_standard_sync_words() -> Dict[str, Any]:
    """List all standard preambles and sync words supported out of the box."""
    return {
        name: {
            "length": len(bits),
            "bit_string": "".join(str(b) for b in bits),
            "hex": np.packbits(bits).tobytes().hex().upper(),
        }
        for name, bits in correlator.STANDARD_SYNC_WORDS.items()
    }


@router.post("/", response_model=CorrelateResponse)
@router.post("/file", response_model=CorrelateResponse)
async def correlate_signal_file(
    file: UploadFile = File(...),
    sync_word: Optional[str] = Query(None, description="Sync word name ('Barker-13', 'CCSDS-32', etc.), hex string, or binary string"),
    frame_length: Optional[int] = Query(None, description="Expected frame length in bits (optional)"),
    threshold: float = Query(0.80, description="Correlation threshold (0.0 to 1.0)"),
    tolerate_inverted: bool = Query(True, description="Tolerate 180° carrier phase inversion"),
    fs: float = Query(1_000_000.0, description="Target WAV sample rate; default sample rate for IQ data"),
    fec_scheme: str = Query("none", description="FEC Scheme: 'none', 'viterbi', 'reed-solomon', 'concatenated', 'ldpc'"),
    auto_deinterleave: bool = Query(True, description="Auto deinterleave demodulated bitstream"),
    auto_discover: bool = Query(True, description="Run automatic preamble discovery"),
):
    """Demodulate, optionally FEC decode and deinterleave, then correlate and frame synchronize an RF signal file."""
    try:
        contents = await file.read()
        sig, sample_rate = load_signal_from_bytes(contents, file.filename or "", default_fs=fs, target_fs=fs)

        # 1. Classify modulation
        clf = classifier.ModulationClassifier()
        clf_res = clf.predict_with_confidence(sig, fs=sample_rate)
        mod = clf_res["modulation"]

        # 2. Extract parameters
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
            bits_for_sync = fec_res["bits"]
        else:
            bits_for_sync = raw_bits

        # 5. Optional Deinterleaving
        if auto_deinterleave and len(bits_for_sync) > 0:
            deint_res = deinterleaver.auto_detect_and_deinterleave(np.asarray(bits_for_sync, dtype=np.uint8))
            if deint_res.get("bits") is not None:
                bits_for_sync = deint_res["bits"]

        # 6. Correlation & Frame Synchronization
        return _correlate_bitstream_core(
            bits=np.asarray(bits_for_sync, dtype=np.uint8),
            sync_word=sync_word,
            frame_length=frame_length,
            threshold=threshold,
            tolerate_inverted=tolerate_inverted,
            auto_discover=auto_discover,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Correlation error: {str(e)}")


@router.post("/bits", response_model=CorrelateResponse)
async def correlate_bits_direct(request: CorrelateRequest = Body(...)):
    """Directly correlate and frame synchronize an input bitstream (bits list, hex string, or bit string)."""
    try:
        bits_arr: Optional[np.ndarray] = None

        if request.bits is not None and len(request.bits) > 0:
            bits_arr = np.asarray(request.bits, dtype=np.uint8)
        elif request.hex_string:
            bits_arr = correlator.hex_to_bits(request.hex_string)
        elif request.bit_string:
            clean_str = "".join(c for c in request.bit_string if c in "01")
            bits_arr = np.array([int(c) for c in clean_str], dtype=np.uint8)
        else:
            raise HTTPException(status_code=400, detail="Must provide 'bits', 'hex_string', or 'bit_string'.")

        return _correlate_bitstream_core(
            bits=bits_arr,
            sync_word=request.sync_word,
            sync_word_hex=request.sync_word_hex,
            sync_word_bits=request.sync_word_bits,
            frame_length=request.frame_length,
            threshold=request.threshold,
            tolerate_inverted=request.tolerate_inverted,
            auto_discover=request.auto_discover,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Correlation error: {str(e)}")
