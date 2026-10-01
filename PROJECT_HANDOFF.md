# PS26147 AMR — Project Handoff

**Active repository:** C:\Users\amity\Desktop\ps26147_toolkit

This is the only project covered by this handoff and the sole active development/deployment target. No code was modified, no training or inference was run, no tests were run, and no TEST data was accessed in preparing this document.

## Project Overview

PS26147 is a digital/RF signal analysis toolkit. It ingests WAV, complex-IQ and raw captures; predicts modulation and confidence; estimates signal parameters; and supports demodulation, FEC/deinterleaving, synchronization/correlation, and a live synthetic SDR visualization.

Current architecture:

- RF classifier: ps26147_toolkit/classifier.py and associated feature/rule code.
- Research CNN: research_qam_psk_fsk_model/, currently disconnected from the API and UI.
- Backend: FastAPI under api/.
- Frontend: Next.js/React under frontend/.

### Existing RF pipeline

The inspected API routes import ps26147_toolkit.classifier and call ModulationClassifier().predict_with_confidence directly. The RF classifier computes baseband/cumulant/feature values, uses RF predict_proba when fitted, and combines max class probability with entropy certainty for confidence. If RF prediction is unavailable/fails, it uses rule_based_classify and returns confidence 0.85 with one-hot probabilities. The classifier constructor may build/train an RF model on synthetic data if it cannot load a fitted model. Do not create a new classifier per request or per fallback.

The RF path has a different feature contract from the CNN. RF remains the current API-facing classifier and is intended to become CNN fallback only.

### Existing web UI

Upload workflow is in frontend/components/UploadZone.tsx. API fetch/types live in frontend/services/api.ts. Results are rendered in frontend/app/results/page.tsx. Simulator UI is frontend/app/live/page.tsx and its WebSocket helper is frontend/services/stream.ts.

### Existing API routes

api/main.py registers GET /health and routers:
- /process: file, async, status endpoints
- /classify/
- /decode/
- /correlate/, /correlate/file, /correlate/bits and sync-word lookup
- WebSocket /stream/live

Exact current route flow and likely integration touchpoints are below.

## CNN Model Overview

### Architecture

Code:
- research_qam_psk_fsk_model/model/multimodal_model.py
- research_qam_psk_fsk_model/model/encoders.py
- research_qam_psk_fsk_model/model/fusion.py
- research_qam_psk_fsk_model/labels.py

Six dedicated encoders with global average pooling:

| Branch | Input | Output |
|---|---|---:|
| Raw I/Q | [B,2,4096] | 128-D |
| Polar amplitude/phase | [B,4,4096] | 64-D |
| Nonlinear PSK phase | [B,4,4096] | 64-D |
| Instantaneous frequency | [B,1,4096] | 64-D |
| Welch PSD | [B,1,2048] | 64-D |
| Handcrafted features | [B,22] | 128-D |

IQ convolution widths are 2→32→64→128 with kernels 7/5/3, one max-pool and adaptive global average pooling. Auxiliary temporal branches use shallow 2–3 convolutional blocks with 32/64/96 widths, pooling, global average pooling and projection to 64-D. PSD uses 1→16→32→64 convolutions and pooling. Feature MLP is 22→32→32→128. Auxiliary embeddings project into 128-D.

Each branch is LayerNormed. Six learnable scalar logits are softmax-normalized to sum-to-one modality weights. Their weighted sum is a 128-D fused vector. Classifier: 128→128→64→7 with GELU and dropout (configured 0.25, then half-rate). No attention, recurrence, or Transformer.

Architecture doc reports 228,701 total parameters. Real fine-tuning hardware metadata reports 179,949 trainable parameters because some encoder parameters were frozen then; this does not represent a smaller inference architecture.

### Representations and feature pipeline

All representations derive from the same 4096-sample complex IQ crop. I and Q are jointly RMS-normalized with RMS sqrt(mean(I²+Q²)/2) over valid samples. Never independently normalize I and Q.

- IQ: normalized I,Q channels.
- Polar: clip(abs(x)/sqrt(2),0,6), sin(phi), cos(phi), differential phase/pi. Wrapped phase only.
- Nonlinear PSK: cos(2phi), sin(2phi), cos(4phi), sin(4phi).
- IF: angle(x[n]*conj(x[n-1]))/(2*pi), equivalent to observed instantaneous frequency/Fs. Length 4096 via repeating first valid differential sample.
- PSD: complex two-sided Welch; Hann; nperseg 1024; overlap 512; nfft 2048; density scaling; FFT-shift; log(max(P,1e-30)); frozen TRAIN-fitted per-bin standardization.
- Features: existing base 16-feature extractor plus six radial QAM descriptors; frozen TRAIN-only StandardScaler.

Feature code is in research_qam_psk_fsk_model/data/representations.py and data/psd_utils.py. Base features come from ps26147_toolkit/feature_extractor.py and six radial values from ps26147_toolkit/qam_candidate_features.py. The API/RF feature contract is not interchangeable.

Exact 22-feature order from research_qam_psk_fsk_model/labels.py:
1. abs_C20
2. abs_C40
3. abs_C41
4. Re_C42
5. abs_C60
6. Re_C63
7. gamma_max
8. sigma_aa
9. sigma_dp
10. sigma_af
11. spec_entropy
12. kurtosis_env
13. fsk_persistence
14. qpsk_metric
15. psk8_metric
16. PAPR
17. radial_q90_over_q50
18. radial_q50_over_q10
19. radial_moment4_rmsnorm
20. radial_moment6_rmsnorm
21. radial_hist_entropy_8
22. radial_midband_fraction_075_125

Do not provide labels, generator seeds, nominal deviation, recorded CFO or SNR as classifier inputs.

### Classes and confidence

Central ordering:
0 BPSK, 1 QPSK, 2 8PSK, 3 16QAM, 4 64QAM, 5 2FSK, 6 4FSK.

Seven logits are softmaxed. Argmax is the predicted class; maximum softmax is confidence. It is not calibrated by the saved audits and differs semantically from RF confidence.

## Training History

### Synthetic pretraining

Run: research_qam_psk_fsk_model/results/synthetic_pretrain/seed_42/

- 20,002 TRAIN, 4,998 VALIDATION (714/class); deterministic 80/20 file-level split.
- Seed 42; RTX 2050/CUDA AMP; AdamW; batch 16; focal gamma 2.0; low-SNR QAM training weight 0.25; maximum 40 epochs; patience 8.
- Best epoch 38.
- Validation accuracy 0.783313; macro-F1 0.781564; macro-precision 0.792057; macro-recall 0.783313.

### Real fine-tuning

Run: research_qam_psk_fsk_model/results/real_finetune/seed_42/

- TRAIN 546 (78/class), VALIDATION 119 (17/class). TEST was not opened for this run.
- Best epoch 18, selection priority macro-F1 → 64QAM recall → accuracy.
- Ran 26 of 40 epochs; stopped after eight epochs without a better selection tuple (patience 8).
- Validation accuracy 103/119 = 86.5546%; macro-F1 0.868306; macro-precision 0.885199; macro-recall 0.865546.
- Per-class P/R/F1:
  - BPSK 1.000/1.000/1.000
  - QPSK 0.600/0.882/0.714
  - 8PSK 0.923/0.706/0.800
  - 16QAM 0.857/0.706/0.774
  - 64QAM 0.875/0.824/0.848
  - 2FSK 1.000/1.000/1.000
  - 4FSK 0.941/0.941/0.941

Validation confusion matrix (true rows, predicted columns, canonical order):

| | BPSK | QPSK | 8PSK | 16QAM | 64QAM | 2FSK | 4FSK |
|---|---:|---:|---:|---:|---:|---:|---:|
| BPSK | 17 | 0 | 0 | 0 | 0 | 0 | 0 |
| QPSK | 0 | 15 | 1 | 0 | 0 | 0 | 1 |
| 8PSK | 0 | 5 | 12 | 0 | 0 | 0 | 0 |
| 16QAM | 0 | 3 | 0 | 12 | 2 | 0 | 0 |
| 64QAM | 0 | 1 | 0 | 2 | 14 | 0 | 0 |
| 2FSK | 0 | 0 | 0 | 0 | 0 | 17 | 0 |
| 4FSK | 0 | 1 | 0 | 0 | 0 | 0 | 16 |

These validation examples were used for checkpoint selection; not independent TEST results. The saved analysis says synthetic-vs-real differences are descriptive across domains and do not quantify transfer gain; no same-architecture real-from-scratch control exists.

## Best Checkpoint

Selected checkpoint:
C:\Users\amity\Desktop\ps26147_toolkit\research_qam_psk_fsk_model\results\real_finetune\seed_42\best.pt

Do not substitute final.pt: best.pt corresponds to selected epoch 18. Synthetic initialization checkpoint is research_qam_psk_fsk_model/results/synthetic_pretrain/seed_42/best.pt.

Recorded prior size for real best.pt: 1,008,172 bytes. Previously recorded SHA256: caccceaf680783d5ba96ee2af6184b5dca0f7a20b358154494d7201a2e3d5d45. This handoff pass did not recompute hash; recompute before packaging.

Checkpoint format marker expected: research_qam_psk_fsk_multimodal_v1. Payload includes model state_dict, architecture config, class/feature names and ordering, input length, branch/representation config, feature scaler, PSD scaler, normalization, seed/stage/config and loss metadata.

Before serving: verify checksum, strict state_dict load, seven class order, exact 22 feature order/count, finite 22-D feature scaler, finite 2048-bin PSD scaler, 4096 crop policy, actual Fs handling, and finite seven-way probabilities summing to one. Scalers remain frozen; never fit them at inference. No TEST access.

## Runtime Inference Pipeline

Complex IQ → actual sample rate → deterministic crop/pad to 4096 → joint I/Q RMS normalize → generate exact 22 features plus IQ/polar/PSK/IF/PSD views from that same crop → apply checkpoint feature/PSD scalers → six encoders → weighted fusion → logits → softmax → argmax label and max-softmax confidence.

Files:
- research_qam_psk_fsk_model/inference.py
- research_qam_psk_fsk_model/model/multimodal_model.py
- research_qam_psk_fsk_model/model/encoders.py
- research_qam_psk_fsk_model/model/fusion.py
- research_qam_psk_fsk_model/data/representations.py
- research_qam_psk_fsk_model/data/psd_utils.py
- research_qam_psk_fsk_model/data/multimodal_dataset.py
- research_qam_psk_fsk_model/labels.py
- ps26147_toolkit/feature_extractor.py
- ps26147_toolkit/qam_candidate_features.py
- selected checkpoint above

Current inference helpers are file-based (predict_file/predict_batch). API routes already hold decoded arrays; implement an in-memory IQ+Fs adapter using the same preprocessing. Current inference imports scaler restoration from training/train_multimodal.py: isolate scaler loading so API serving does not import the trainer.

## Existing Application Architecture

### Upload and response flow

frontend/components/UploadZone.tsx accepts WAV/IQ/raw/bin/SigMF data. It calls either processFile or startAsyncProcess + pollJobStatus. It then separately calls decodeSignal and correlateSignal, merges all outputs and routes to results.

Fetch functions and TypeScript contracts: frontend/services/api.ts. Backend response models: api/schemas.py.

api/routes/process.py:
- _process_signal_core loads bytes via api.utils.load_signal_from_bytes
- constructs ModulationClassifier and calls predict_with_confidence
- estimates parameters and chart data
- returns ProcessResponse with modulation, confidence, rates, waveform, constellation, PSD and waterfall
- /process/file is sync; /process/async schedules worker; /process/status/{job_id} returns result.

api/routes/classify.py: POST /classify/ loads bytes, constructs RF classifier, predicts, returns ClassifyResponse with modulation, confidence, features.

api/routes/decode.py: POST /decode/ classifies, estimates baud/carrier, demodulates using predicted modulation, optionally FEC decodes and deinterleaves, returns DecodeResponse with modulation/confidence, bits and EVM.

api/routes/correlate.py: POST /correlate/ and /correlate/file classify uploaded IQ, estimate parameters, demodulate, then correlate/synchronize bitstream. /correlate/bits and sync-word listing do not require IQ classification.

api/routes/stream.py: WebSocket /stream/live generates a synthetic signal frame, classifies each frame, and returns detected/configured modulation, confidence, waveform/PSD/constellation details.

api/main.py registers all these routes and GET /health, currently only status=ok.

### Decode and correlate flow

Decode prediction drives demodulator configuration and downstream FEC/deinterleave. Uploaded-signal correlate does classification then demodulation and framing; bits-only correlation bypasses modulation classification.

### SDR simulator

api/routes/stream.py is synthetic generation, not live hardware or replay. It synthesizes PSK/QAM or FSK, applies channel effects and calls RF classifier. Configured label is known to simulator for generation/scoring and must not be classifier input. Audited frame size is 2048 samples; CNN input requires 4096. Decide on valid contiguous accumulation or keep simulator on RF fallback until a full model window is available. Do not silently pad/duplicate samples.

### UI confidence

frontend/app/results/page.tsx renders modulation and confidence ×100 as percent and exports it as Confidence (%). frontend/app/live/page.tsx renders detected modulation and confidence percent from WebSocket frame. Preserve scalar field names/types and current UI behavior; no dual result display.

## Audit Findings

Repository audit inventory is under diagnostics/repository_amr_audit/ (component_inventory.csv, dependency_map.json, integration_candidates.csv). Other saved reports:
- diagnostics/real_fsk_validation_analysis/report.md
- diagnostics/real_fsk_train_analysis/report.md
- diagnostics/frozen_fsk_representation_audit/report.md
- diagnostics/model_comparison_7class/comparison/report.md
- research_qam_psk_fsk_model/results/real_finetune/seed_42/real_finetune_analysis_report.md

Recorded conclusions:
- Research CNN is not currently called by API/UI.
- CNN requires 22 ordered features; route RF feature implementation is not compatible.
- Frozen FSK analysis: old baseline real validation was 2FSK 15/17, 4FSK 5/17, 4FSK→2FSK 9/17; aligned synthetic was 2FSK 430/504, 4FSK 198/504, 4FSK→2FSK 223/504. Real failures generally retained four nominal regions, with small sample support and mixed/inconclusive interpretation.
- Earlier seed-42 representation comparison: baseline accuracy 60.5%, macro-F1 .5875, 4FSK recall 70.6%; IQ+IF+22 accuracy 67.2%, macro-F1 .6489, 4FSK recall 88.2%; IQ+PSD+22 accuracy 65.5%, macro-F1 .6557, 4FSK recall 70.6%. This predates the selected real-finetuned model and is not its metric.
- Selected fine-tuned model validation: 86.55% accuracy, macro-F1 .8683; BPSK/2FSK 17/17; 4FSK 16/17; main remaining confusions 8PSK→QPSK (5), QPSK→8PSK (1), QPSK→4FSK (1), 16QAM→QPSK (3), 16QAM→64QAM (2), 64QAM→16QAM (2), 64QAM→QPSK (1), 4FSK→QPSK (1).
- Low-sample validation and audits are descriptive only; no causal/generalization claim.
- Raw IQ upload Fs may come from query parameter; api.utils byte loader does not necessarily receive a SigMF sidecar alongside uploaded data.
- /health currently does not expose model readiness.

## Current Integration Goal

CNN is authoritative for supported digital signals. RF is fallback only for explicitly unsupported inputs (e.g. AM/UNKNOWN policy) or CNN load/inference failure. No RF-vs-CNN comparison, shadow mode, dual predictions, or A/B testing. Keep existing public response schemas and confidence display. Avoid arbitrary max-softmax fallback threshold absent calibration.

## Required Integration Work — File-by-File

All paths are inside the active repository root above.

| File | Likely work |
|---|---|
| ps26147_toolkit/cnn_runtime/__init__.py (new) | Shared runtime/service and singleton entrypoint. |
| ps26147_toolkit/cnn_runtime/model.py (new or copied exact model code) | Reconstruct/load exact six-branch architecture with strict state loading. |
| ps26147_toolkit/cnn_runtime/preprocessing.py (new) | In-memory IQ+Fs crop, normalization and exact representation pipeline. |
| ps26147_toolkit/cnn_runtime/assets/best.pt (new copy) | Verified immutable copy of selected real-finetuned checkpoint. |
| ps26147_toolkit/cnn_runtime runtime dependencies (new) | Exact feature/radial/PSD/labels/scaler restore code; no trainer import. |
| api/routes/process.py | Route sync and async upload classification through shared CNN-authoritative service. |
| api/routes/classify.py | Use same service for standalone classify endpoint. |
| api/routes/decode.py | Use same result before demod/FEC; preserve caller override if present in current version. |
| api/routes/correlate.py | Use shared service on IQ/file routes; leave bits-only endpoint alone. |
| api/routes/stream.py | Only integrate once 2048/4096 policy is specified; otherwise RF fallback. |
| api/main.py | Optional single model initialization/readiness reporting while preserving health schema. |
| api/utils.py | Possibly support actual Fs/SigMF metadata path if needed; avoid unrelated upload behavior changes. |
| requirements.txt, pyproject.toml, Dockerfile.api | Add compatible PyTorch/scientific runtime dependencies and package checkpoint asset. |
| frontend/services/api.ts, UploadZone.tsx, results/page.tsx, services/stream.ts, live/page.tsx | Prefer unchanged; current scalar modulation/confidence contract already renders output. |

Prefer not to alter ps26147_toolkit/classifier.py, feature_extractor.py, parameter_extractor.py, demodulator.py or the frontend unless a concrete incompatibility requires it. The old ps26147_toolkit/hybrid_model.py and root hybrid_model_best.pt are disconnected, older experimental artifacts; do not mistake them for the selected six-branch model.

## Files That Must Remain Untouched

- All data/ and data_splits*/ content, manifests and split definitions.
- All research_qam_psk_fsk_model/results/** checkpoints, including selected best.pt and final.pt.
- The trained architecture/checkpoint source artifacts; if runtime copies are needed, copy without modifying originals.
- ps26147_toolkit/classifier.py and RF model asset as fallback behavior.
- Existing response contracts in api/schemas.py unless a necessary compatible extension is separately approved.
- Existing scientific behavior in feature_extractor.py, parameter_extractor.py, demodulator.py.
- Reserved TEST manifests and IQ captures.
- Existing experiment and diagnostic outputs.

## Known Risks

- Main/RF features differ from the CNN's exact 22 features.
- Feature and PSD scalers must match checkpoint training and remain frozen.
- Strict checkpoint state/config and PyTorch version compatibility.
- Current inference imports training code for scaler restoration.
- CNN 4096 input vs simulator 2048 samples/frame.
- AM is not a CNN class; use explicit RF fallback.
- UNKNOWN is not a CNN class; define safe unsupported behavior.
- RF constructor may synthesize/train fallback model if model missing; avoid per-request construction.
- CNN max-softmax and RF confidence have different semantics.
- Upload frontend runs process, decode and correlate as separate requests; classification can be repeated.
- Sample rate may be query-provided for raw IQ; missing/wrong Fs distorts PSD and sample-rate-aware features.
- Signal conditioning in API may differ from training preprocessing; do not silently DC-remove/downconvert without checking.
- Concurrent API requests, model load, memory and inference latency need handling.
- Preserve schemas and caller-supplied decode modulation semantics.
- Never leak simulator's configured label as model input.

## Current Status

Complete:
- six-branch seven-class research model built.
- seed-42 synthetic pretraining and real fine-tuning completed.
- selected checkpoint and fitted scalers/config/history/metrics saved.
- repository component/integration candidate audit and FSK/representation audits saved.
- real fine-tuned validation is 86.55% accuracy, macro-F1 .8683.

Incomplete:
- CNN runtime is not wired to routes.
- no in-memory array+Fs adapter.
- no shared CNN-primary/RF-fallback service.
- direct RF route calls remain.
- checkpoint/runtime assets have not been isolated for serving.
- AM/UNKNOWN and 2048-frame simulator policies are unresolved.
- no integration validation yet.

## Exact Next Steps

1. Re-read current source and saved reports; verify best.pt metadata and recompute hash, without TEST access.
2. Package a runtime-only copy of exact architecture, labels, 22-feature/radial code, PSD code and checkpoint under this repository. Keep research originals untouched.
3. Remove training/data-loading dependencies from serving path; implement predict_iq(iq, Fs).
4. Match exact crop, RMS, derived representations, scaler and class order; strict-load checkpoint once into a singleton.
5. Add a shared classifier service: CNN authoritative for supported digital signals; RF fallback only on explicit unsupported scope or CNN failure. Log internal fallback reason; don't produce a second public prediction.
6. Wire process, classify, decode, and IQ/file correlate to that service; leave bits-only correlate alone.
7. Decide simulator 2048-to-4096 semantics before connecting CNN; preserve output schema and exclude ground-truth label from inference.
8. Ensure actual sample rate reaches runtime for WAV/SigMF/raw inputs.
9. Resolve dependency/container packaging, startup behavior, concurrency, health/readiness and CPU/GPU device policy.
10. After implementation authorization, verify representation parity, strict checkpoint load, route contracts, fallback behavior, scalar confidence UI and simulator behavior. Never use TEST.

## Rollback Strategy

Keep current RF classifier and asset intact. Gate CNN routing behind one local configuration switch. If load, preprocessing, inference or API behavior fails, disable that switch and route to the existing RF singleton. Do not instantiate a new RF classifier on every failure. Keep CNN assets isolated; rollback need not delete them. Revert only integration changes and confirm RF-only routes retain prior schemas. No TEST inference.

## Repository references

- Research architecture: research_qam_psk_fsk_model/ARCHITECTURE.md
- Research guide: research_qam_psk_fsk_model/README.md
- Selected run artifacts: research_qam_psk_fsk_model/results/real_finetune/seed_42/
- Synthetic run artifacts: research_qam_psk_fsk_model/results/synthetic_pretrain/seed_42/
- Component audit: diagnostics/repository_amr_audit/
- FSK analyses: diagnostics/real_fsk_validation_analysis/, diagnostics/real_fsk_train_analysis/, diagnostics/frozen_fsk_representation_audit/
- Earlier model comparison: diagnostics/model_comparison_7class/comparison/

