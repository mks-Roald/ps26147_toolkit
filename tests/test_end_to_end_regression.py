"""Broad API and simulator regression checks (run with ``pytest -m regression``)."""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.routes.stream import generate_sdr_frame
from ps26147_toolkit import fec_decoders

client = TestClient(app)
FS, BAUD = 1_000_000.0, 25_000.0
MODULATIONS = ["BPSK", "QPSK", "8PSK", "16QAM", "64QAM", "2FSK", "4FSK"]


def _record(request, expected, actual, tolerance=None, stage="API"):
    request.node._regression_record = {"expected": expected, "actual": actual,
                                      "tolerance": tolerance, "pipeline_stage": stage}


def _signal(modulation="BPSK", n_symbols=512, snr_db=30, seed=26147):
    rng = np.random.default_rng(seed)
    sps = int(FS / BAUD)
    if modulation == "BPSK":
        symbols = rng.choice([-1, 1], n_symbols)
    elif modulation == "QPSK":
        symbols = (rng.choice([-1, 1], n_symbols) + 1j*rng.choice([-1, 1], n_symbols))/np.sqrt(2)
    elif modulation == "8PSK":
        symbols = np.exp(1j*rng.integers(0, 8, n_symbols)*np.pi/4)
    elif modulation in ("16QAM", "64QAM"):
        levels = [-3, -1, 1, 3] if modulation == "16QAM" else [-7, -5, -3, -1, 1, 3, 5, 7]
        symbols = (rng.choice(levels, n_symbols) + 1j*rng.choice(levels, n_symbols))/np.sqrt(10 if modulation == "16QAM" else 42)
    else:
        count = 2 if modulation == "2FSK" else 4
        f = np.repeat((rng.integers(count, size=n_symbols)-(count-1)/2)*BAUD, sps)
        phase = 2*np.pi*np.cumsum(f)/FS
        sig = np.exp(1j*phase)
        noise = np.sqrt(10**(-snr_db/10)/2)*(rng.normal(size=len(sig))+1j*rng.normal(size=len(sig)))
        return (sig+noise).astype(np.complex64)
    sig = np.repeat(symbols, sps).astype(np.complex64)
    noise = np.sqrt(10**(-snr_db/10)/2)*(rng.normal(size=len(sig))+1j*rng.normal(size=len(sig)))
    return sig+noise


def _upload(sig, name="generated.iq"):
    return {"file": (name, np.asarray(sig, dtype=np.complex64).tobytes(), "application/octet-stream")}


@pytest.mark.regression
@pytest.mark.parametrize("size_mb", [1, 5, 10])
def test_upload_large_binary_iq(size_mb, request):
    # A deterministic arbitrary binary byte pattern, intentionally not base64.
    contents = np.random.default_rng(size_mb).integers(0, 256, size_mb*1024*1024, dtype=np.uint8).tobytes()
    response = client.post("/process/file", params={"fs": FS}, files={"file": ("arbitrary.iq", contents, "application/octet-stream")})
    actual = {"status": response.status_code, "bytes": len(contents)}
    _record(request, {"status": 200, "bytes": len(contents)}, actual, "exact byte count", "upload/storage")
    assert response.status_code == 200, response.text
    assert response.json()["num_samples"] > 0


@pytest.mark.regression
@pytest.mark.parametrize("modulation", MODULATIONS)
def test_simulator_config_detection_classification_and_representation(modulation, request):
    frame = generate_sdr_frame(modulation, snr_db=25, baud_rate=BAUD, fs=FS, n_samples=8192, seed=26147)
    representation = frame["visualization"]
    actual = {"configured": frame["configured_modulation"], "detected": frame["detected_modulation"],
              "correct": frame["classification_correct"], "representation": representation,
              "points": len(frame["constellation"] or frame["frequency_states"])}
    expected_representation = "frequency" if modulation.endswith("FSK") else "constellation"
    _record(request, {"configured": modulation, "detected": modulation, "correct": True,
                      "representation": expected_representation, "points": ">0"}, actual,
            "exact classification", "live simulator")
    assert actual["configured"] == modulation
    assert actual["detected"] == modulation
    assert actual["correct"] is True
    assert representation == expected_representation and actual["points"] > 0


@pytest.mark.regression
@pytest.mark.parametrize("modulation", MODULATIONS)
def test_process_visualization_payload_for_each_modulation(modulation, request):
    response = client.post("/process/file", params={"fs": FS}, files=_upload(_signal(modulation)))
    assert response.status_code == 200, response.text
    body = response.json()
    actual = {"modulation": body["modulation"], "waveform": len(body["waveform_data"] or []),
              "psd": len(body["psd_data"] or []), "waterfall": bool(body["waterfall_data"]),
              "constellation": body["constellation_metadata"], "fsk": body["fsk_visualization_data"]}
    expected_rep = "frequency_states" if modulation.endswith("FSK") else "recovered_symbols"
    _record(request, {"modulation": modulation, "waveform": ">0", "psd": ">0", "waterfall": True,
                      "representation": expected_rep}, actual, "non-empty charts", "visualization")
    assert actual["waveform"] and actual["psd"] and actual["waterfall"]
    assert body["modulation"] == modulation
    if modulation.endswith("FSK"):
        assert body["fsk_visualization_data"]["frequency_state_count"] == (4 if modulation == "4FSK" else 2)
    else:
        assert body["constellation_metadata"]["representation"] == "recovered_symbols"
        assert body["recovered_symbols"]


@pytest.mark.regression
def test_same_signal_routes_parameters_and_modulation_agree(request):
    raw = _upload(_signal("BPSK", 512))
    responses = [client.post(path, params={"fs": FS, "fec_scheme": "none", "auto_detect_sync": "false"}, files=raw)
                 for path in ("/process/file", "/classify/", "/decode/")]
    assert all(r.status_code == 200 for r in responses), [r.text for r in responses]
    bodies = [r.json() for r in responses]
    keys = ("modulation", "center_frequency_hz", "bandwidth_hz", "baud_rate", "snr_db")
    actual = {key: [body[key] for body in bodies] for key in keys}
    _record(request, "same modulation and parameter estimates across all endpoints", actual,
            "numeric rtol=1e-6, atol=1e-3", "cross-route consistency")
    assert actual["modulation"].count(actual["modulation"][0]) == len(bodies)
    for key in keys[1:]:
        assert np.allclose(actual[key], actual[key][0], rtol=1e-6, atol=1e-3)


@pytest.mark.regression
def test_bpsk_barker_16x16_viterbi_recovers_hello_world(request):
    from ps26147_toolkit.correlator import STANDARD_SYNC_WORDS
    from ps26147_toolkit.fec_decoders import ConvolutionalCodec
    text = "hello world\n" * 4
    payload = np.unpackbits(np.frombuffer(text.encode(), dtype=np.uint8))
    encoded = ConvolutionalCodec().encode(payload, flush=True)
    pad = np.random.default_rng(7).integers(0, 2, (-len(encoded)) % 256).astype(np.uint8)
    encoded = np.concatenate([encoded, pad])
    interleaved = np.concatenate([encoded[i:i+256].reshape(16, 16).ravel(order="F")
                                  for i in range(0, len(encoded), 256)])
    bits = np.concatenate([np.random.default_rng(11).integers(0, 2, 64).astype(np.uint8),
                           STANDARD_SYNC_WORDS["Barker-13"].astype(np.uint8), interleaved])
    sps = int(FS/BAUD)
    base = np.repeat(np.where(bits == 1, 1., -1.), sps)
    t = np.arange(len(base))/FS
    rng = np.random.default_rng(42)
    signal = base*np.exp(2j*np.pi*50_000*t)
    signal += rng.normal(0,.01,len(signal))+1j*rng.normal(0,.01,len(signal))
    response = client.post("/decode/", params={"fs": FS, "fec_scheme": "viterbi", "sync_word": "Barker-13",
        "auto_deinterleave": "true", "auto_detect_sync": "true"}, files=_upload(signal))
    assert response.status_code == 200, response.text
    body = response.json()
    actual = {"ascii": body["decoded_ascii"][:len(text)], "sync": body["sync_method"],
              "deinterleaver": body["deinterleaver_method"], "fec_ran": body["fec_ran"]}
    _record(request, {"ascii": text, "sync": "sync_word", "deinterleaver": "block", "fec_ran": True},
            actual, "exact payload and selected stages", "end-to-end decode")
    assert text.replace("\n", ".") in body["decoded_ascii"]
    assert body["sync_method"] == "sync_word"
    assert body["deinterleaver_method"] == "block"
    assert body["fec_ran"] is True


@pytest.mark.regression
@pytest.mark.parametrize("noise_kind", ["awgn", "low_snr"])
def test_unknown_signals_do_not_claim_high_confidence_modulation(noise_kind, request):
    rng = np.random.default_rng(4242)
    sig = (rng.normal(size=8192)+1j*rng.normal(size=8192)).astype(np.complex64)
    if noise_kind == "low_snr":
        sig = _signal("BPSK", 512, snr_db=-15, seed=4242)
    response = client.post("/classify/", params={"fs": FS}, files=_upload(sig))
    assert response.status_code == 200, response.text
    body = response.json()
    actual = {"modulation": body["modulation"], "confidence": body["confidence"]}
    _record(request, {"uncertainty": "confidence < 0.7 or unknown label"}, actual,
            "confidence <0.7", "classification/rejection")
    assert body["confidence"] < 0.7 or body["modulation"].lower() in {"unknown", "uncertain"}


@pytest.mark.regression
def test_unsupported_simulator_modulation_is_rejected(request):
    from api.routes.stream import _waveform
    _record(request, "ValueError for unsupported modulation", "raises ValueError",
            "exact rejection", "simulator input validation")
    with pytest.raises(ValueError, match="Unsupported modulation"):
        _waveform("256QAM", BAUD, FS, 1024, np.random.default_rng(1))


@pytest.mark.regression
@pytest.mark.parametrize("payload,expected_status", [(b"", 500), (b"\x01\x02\x03", 500), (b"\x00\x00\x00\x00", 200)])
def test_malformed_and_too_short_uploads_fail_cleanly(payload, expected_status, request):
    response = client.post("/process/file", params={"fs": FS}, files={"file": ("bad.iq", payload, "application/octet-stream")})
    actual = {"status": response.status_code, "body": response.text[:300]}
    _record(request, {"status": expected_status}, actual, "exact status", "input validation")
    assert response.status_code == expected_status


@pytest.mark.regression
@pytest.mark.parametrize("scheme", ["viterbi", "reed-solomon", "concatenated", "ldpc"])
def test_selected_fec_dispatches_decoder(scheme, monkeypatch, request):
    called = []
    original = fec_decoders.decode_fec
    def observe(bits, scheme, **kwargs):
        called.append(scheme)
        return original(bits, scheme=scheme, **kwargs)
    monkeypatch.setattr("api.routes.decode.fec_decoders.decode_fec", observe)
    response = client.post("/decode/", params={"fs": FS, "fec_scheme": scheme,
        "sync_word": "", "auto_detect_sync": "false"}, files=_upload(_signal("BPSK", 512)))
    assert response.status_code == 200, response.text
    actual = {"dispatches": called, "fec_ran": response.json()["fec_ran"]}
    _record(request, {"dispatches": [scheme], "fec_ran": True}, actual, "decoder invoked", "FEC")
    assert called == [scheme]
    assert actual["fec_ran"]
