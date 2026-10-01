# Stage 2 Execution Plan (Prepared Only)

1. Load `configs/real_finetune_7class.yaml` with stage `real_finetune`.
2. Verify the TRAIN/VALIDATION hashes and disjointness before reading capture data. Do not inspect the TEST manifest.
3. Load model weights from `results/synthetic_pretrain/seed_42/best.pt`; preserve that source checkpoint.
4. Restore feature and PSD scalers from the checkpoint. Do not fit transforms on validation.
5. Warm up for epochs 1–4 with multimodal encoders frozen and fusion/head trainable.
6. Starting epoch 5, unfreeze only YAML-listed late blocks/projections. Use encoder LR 1e-5 and fusion/head LR 3e-5.
7. Keep all other model and loss settings in the config fixed. Select the checkpoint by validation macro-F1, then 64QAM recall, then accuracy.
8. Save outputs only under `results/real_finetune/seed_42/`; examine timing, GPU utilization, validation summaries, and saved scaler/config metadata after training is explicitly authorized and manually launched.

The plan is descriptive. No step was executed in this preparation task.
