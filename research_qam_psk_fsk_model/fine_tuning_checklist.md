# Stage 2 Fine-Tuning Checklist

## Prepared checks

- [x] Synthetic seed-42 initialization checkpoint exists and its SHA256 is recorded.
- [x] Real TRAIN manifest hash matches the expected digest; 546 samples, 78/class.
- [x] Real VALIDATION manifest hash matches the expected digest; 119 samples, 17/class.
- [x] TRAIN/VALIDATION path overlap is zero; IQ files and SigMF sidecars exist.
- [x] TEST manifest and TEST IQ were not accessed.
- [x] Feature and PSD transforms are reused from the pretrained checkpoint; validation does not fit transforms.
- [x] Stage 2 config uses 4096 samples, batch 16, AdamW, AMP when available, focal gamma 2.0, and low-SNR QAM weight 0.25.
- [x] Four-worker persistent DataLoader options, prefetching, pinned CUDA memory, timing, and GPU telemetry are configured.
- [x] Encoder freeze/unfreeze and discriminative LR controls are configured.
- [x] Output folder is separate from the synthetic run and existing checkpoints.

## Before the user starts training

- [ ] Confirm the new GNU Radio/synthetic-pretraining results are the intended source checkpoint and has not been replaced.
- [ ] Review the real split audit and keep the recorded hashes unchanged.
- [ ] Confirm the RTX 2050 is available and close competing GPU workloads.
- [ ] Review the scaler policy (reuse synthetic TRAIN scalers); do not fit on validation.
- [ ] Record current Git commit/repository version in the run notes.
- [ ] Start only with the exact command in `fine_tuning_run_instructions.md`.
- [ ] Inspect logged DataLoader wait/preprocessing/GPU timing and memory after the run; do not alter the experimental protocol mid-run without recording a new experiment.
- [ ] Keep TEST entirely untouched until a separately authorized final evaluation.

No checklist item launches work automatically.
