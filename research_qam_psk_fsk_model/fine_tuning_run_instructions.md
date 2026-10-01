# Manually Starting Stage 2 Real Fine-Tuning

This command is prepared for later manual use. It has **not** been executed in this preparation task. Run it from the repository root after reviewing `fine_tuning_checklist.md`.

The run loads the synthetic-pretrained multimodal checkpoint, uses only the existing real TRAIN and VALIDATION manifests, and writes new artifacts under `research_qam_psk_fsk_model/results/real_finetune/seed_42/`. It does not point at the TEST manifest.

## VS Code terminal (PowerShell)

```powershell
.\venv\Scripts\python.exe -m research_qam_psk_fsk_model.training.train_multimodal `
  --config research_qam_psk_fsk_model/configs/real_finetune_7class.yaml `
  --stage real_finetune `
  --execute-training
```

## CMD terminal

```bat
.\venv\Scripts\python.exe -m research_qam_psk_fsk_model.training.train_multimodal --config research_qam_psk_fsk_model/configs/real_finetune_7class.yaml --stage real_finetune --execute-training
```

Expected behavior when explicitly started: CUDA is selected if initialization succeeds; otherwise the script records the CUDA initialization error and falls back to CPU. AMP is enabled on CUDA if supported. The configured worker count is four. The run's checkpoint files are new Stage 2 outputs and do not overwrite the synthetic-pretraining checkpoint.

This document gives a command only. Do not interpret its presence as evidence that training has started.
