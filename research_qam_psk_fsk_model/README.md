# Research QAM/PSK/FSK multimodal model

This is an isolated research implementation for the digital seven-class AMR target. It is not imported by, or connected to, `classifier.py`, the API, or the frontend. The existing 7-class model and its checkpoints remain the reference baseline.

**Training has not been performed as part of this task.** No signals were generated and no dataset was changed. The reserved TEST split is intentionally unsupported by the new dataset API.

## Classes and input views

Class order is centralized in `labels.py`: `BPSK`, `QPSK`, `8PSK`, `16QAM`, `64QAM`, `2FSK`, `4FSK`.

Each lazy sample loader selects one complex IQ crop of 4096 samples. TRAIN uses a seeded random crop per sample and epoch; VALIDATION and inference use a deterministic center crop. All branches are recomputed from that exact crop.

| Branch | Input shape | Construction | Encoder output |
|---|---|---|---:|
| Raw IQ | `[B,2,4096]` | I/Q jointly divided by RMS over valid samples | 128-D |
| Polar | `[B,4,4096]` | `[clip(abs(x)/sqrt(2),0,6), sin(phi), cos(phi), dphi/pi]` | 64-D |
| Nonlinear PSK | `[B,4,4096]` | `[cos(2phi), sin(2phi), cos(4phi), sin(4phi)]` | 64-D |
| IF | `[B,1,4096]` | signed `angle(x[n]conj(x[n-1]))/(2*pi) = f_inst/Fs`; first value repeats first valid difference | 64-D |
| PSD | `[B,1,2048]` | Hann Welch, 1024 segment, 512 overlap, 2048 FFT, complex two-sided density, FFT-shift, `log(max(P,1e-30))`, TRAIN-only per-bin standardization | 64-D |
| Handcrafted | `[B,22]` | exact existing 16 features plus the exact six `qam_candidate_features` radial descriptors | 128-D |

The 22 feature names and their exact order live in `labels.py`; extraction delegates to the existing toolkit functions. Feature StandardScaler and PSD per-bin statistics are fit on deterministic center crops from the active TRAIN manifest only. Validation/inference reuse those stored transforms. IQ sample rate is read from SigMF metadata; if a raw IQ capture lacks a rate, configure an explicit fallback or the loader errors. WAV rate is read from its header.

## Model architecture and fusion

Each signal view has a separate shallow encoder. The IQ, Polar, PSK, IF, and PSD branches use 2–3 convolutional blocks, BatchNorm/GELU, pooling, and global average pooling; none uses a sequence-wide Flatten layer. The feature branch is `22→32→32→128`. Auxiliary 64-D embeddings are projected to 128-D.

Each 128-D branch is LayerNormed. Six learned scalar logits are softmax-normalized to weights that sum to one. Fusion is the weighted sum of the six normalized embeddings. A compact `128→128→64→7` classifier produces class logits. There is no recurrent layer, attention, or production integration.

## Loss and augmentation

The reusable focal loss defaults to `gamma=2.0`, with no class alpha weighting. TRAIN examples with target 16QAM/64QAM and known metadata `SNR < 0 dB` receive configurable sample-loss weight `0.25`; all other samples receive weight `1.0`. Missing SNR is never guessed: those examples use weight 1.0 and produce a warning. Validation loss and predictions are unmasked/unweighted by this training rule.

TRAIN-only phase augmentation samples `theta ~ Uniform(0,2*pi)` and rotates the cropped complex signal. Every representation, including the existing features and PSD, is then regenerated from the rotated signal. Validation and inference do not randomly rotate.

## Future data and two-stage workflow

Provide JSON list manifests whose records contain `path` and canonical `label`; optional fields can include `sample_id`, `snr_db`, and `metadata`. SigMF sidecars are read for actual sample rate and optional SNR/SPS/generator metadata. Relative paths resolve under `dataset_root` (or the manifest folder if no root is configured). No fixed file count is assumed.

Stage 1 is configured for the generated dataset at `C:/Users/amity/Desktop/ps26147_toolkit/data_synthetic_25k`. The checked-in experiment-area manifests are deterministic, seed-42, stratified file-level references to the source IQ files; they do not copy or alter any capture. They contain 20,002 TRAIN and 4,998 VALIDATION rows, with 714 validation rows per class. Recreate them only if the source manifest changes, using `research_qam_psk_fsk_model.data.prepare_synthetic_manifests`.

To change the dataset later, edit `configs/multimodal_7class.yaml`:

1. Set `data.synthetic_pretraining_dataset_root`, `synthetic_pretraining_train_manifest`, and `synthetic_pretraining_validation_manifest` for Stage 1. Real Stage 2 paths remain separately configured under `real_finetuning_*`.
2. Run `--stage synthetic_pretrain` and preserve the resulting checkpoint.
3. Set the real fine-tuning root and manifests, then pass the Stage 1 checkpoint to `--stage real_finetune --init-checkpoint ...`. The same model architecture is reconstructed. Stage-2 feature/PSD scalers are fit only from the real TRAIN manifest and frozen for that stage.

For direct training on one dataset, set `data.dataset_root`, `train_manifest`, and `validation_manifest` and keep stage `real_from_scratch`.

Training is guarded by a required explicit `--execute-training` flag. After the future synthetic GNU Radio TRAIN/VALIDATION manifests are configured, Stage 1 can be started from the repository root with:

```powershell
.\venv\Scripts\python.exe -m research_qam_psk_fsk_model.training.train_multimodal `
  --config research_qam_psk_fsk_model/configs/multimodal_7class.yaml `
  --stage synthetic_pretrain `
  --execute-training
```

For a directly configured dataset without synthetic pretraining, use `--stage real_from_scratch`. After Stage 1, Stage 2 uses the preserved Stage 1 best checkpoint:

```powershell
.\venv\Scripts\python.exe -m research_qam_psk_fsk_model.training.train_multimodal `
  --config research_qam_psk_fsk_model/configs/multimodal_7class.yaml `
  --stage real_finetune `
  --init-checkpoint research_qam_psk_fsk_model/results/synthetic_pretrain/seed_42/best.pt `
  --execute-training
```

The manifest paths must be filled in before a future command is run. These commands were **not run** in this task.

## Inference

After training, use `research_qam_psk_fsk_model.inference.predict_file(checkpoint, iq_path)` or `predict_batch(checkpoint, paths)`. The result includes predicted class, confidence, all class probabilities, learned branch weights, contribution norms, sample rate, and checkpoint seed. This remains research-only.

## Checkpoints and diagnostics

New checkpoints save the model state, canonical class/feature order, architecture and representation configuration, feature/PSD scalers, input length, focal/low-SNR settings, seed, training stage, and config. Future training writes run-specific best/final checkpoints, histories, validation metrics, confusion matrices, predictions with metadata, and hardware details under this experiment directory.

Evaluation helpers include overall and per-class precision/recall/F1, confusion counts for 16QAM↔64QAM and 2FSK↔4FSK, QAM SNR bins, and FSK SNR/SPS/sample-rate breakdowns.
