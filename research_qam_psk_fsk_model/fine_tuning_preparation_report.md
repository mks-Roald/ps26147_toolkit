# Real-Data Fine-Tuning Preparation Report

## Status and scope

Stage 2 is configured and prepared. No training, fine-tuning, evaluation, or model inference was run. The reserved TEST manifest was not opened and no TEST IQ or metadata was loaded. No dataset, existing split, architecture, feature extractor, loss, fusion design, production code, or existing checkpoint was modified.

## Inputs and split verification

Pretrained initialization: `results/synthetic_pretrain/seed_42/best.pt` (SHA256 `2570dae8d5a4974fa62a9a14b34bd09d737408ac412c9f23237c8f56d90e3290`).

Real dataset root: `C:/Users/amity/Desktop/ps26147_toolkit/data`.

| Split | Manifest | SHA256 | Samples |
|---|---|---|---:|
| TRAIN | `data_splits_realistic_7class/data_7class_ec8746df76/train.json` | `b6b605f99e8838cca10b8f2780868c091b54b1fe386ad070600db20b7c15f844` | 546 (78/class) |
| VALIDATION | `data_splits_realistic_7class/data_7class_ec8746df76/val.json` | `b33530fb26962e10b82fb94bb02e5c0f639d1338715775fbc23d3ba7378af722` | 119 (17/class) |

The path overlap count is zero; every TRAIN/VALIDATION IQ path and SigMF sidecar exists. The existing manifests are referenced directly and were not copied or edited. TEST access flags are false in `data/real_finetune_manifest_audit.json`.

## Configuration

The Stage 2 YAML is `configs/real_finetune_7class.yaml`. It selects `real_finetune`, the real TRAIN/VALIDATION manifests above, the synthetic seed-42 checkpoint, the existing seven-class order, input length 4096, and the checkpoint's full multimodal model. It retains AdamW, focal gamma 2.0, low-SNR QAM weight 0.25, batch 16, maximum 40 epochs, patience 8, AMP when CUDA is available, and phase rotation only on TRAIN.

### Scaler policy

The configuration reuses the serialized feature StandardScaler and PSD standardizer from the synthetic-pretraining checkpoint (`scaler_policy: pretrained_checkpoint`). Both were fitted from synthetic TRAIN only. Reusing them keeps the representation coordinates stable at the transfer boundary and avoids fitting any transform on validation. This is the selected reproducible default, not a claim that synthetic and real feature distributions are identical. If domain-shift analysis later motivates real-TRAIN scaler adaptation, treat that as a separate, documented experiment with scalers fitted only on real TRAIN and frozen before validation.

### Freeze schedule and learning rates

For epochs 1–4, all six encoders (IQ, polar, nonlinear PSK, IF, PSD, and handcrafted features) are frozen; the branch projections not classified as encoders, fusion, and classifier remain trainable. At epoch 5, the YAML selectively unfreezes the configured late encoder blocks/projections and feature MLP/projection. Remaining encoder layers stay frozen. The optimizer groups use encoder LR `1e-5`, fusion/head LR `3e-5`, and weight decay `1e-4`. The transfer hooks also keep frozen encoder modules in evaluation mode so BatchNorm running state is not updated during the frozen warm-up. These controls are configuration only and have not been exercised in a training run.

## Prepared output location

Future outputs are isolated at `results/real_finetune/seed_42/`; the directory currently contains only a `.gitkeep`. It does not contain a newly trained checkpoint.

## Guardrails

Training has not started. Validation is configured only for model selection/monitoring; it is not used for scaler fitting. The TEST split remains untouched. Before manually starting Stage 2, review the checklist and record the repository revision in the run record.
