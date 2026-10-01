# README for the PS26147 Toolkit

## Regression suite
Run the complete Python API/unit and end-to-end regression suite, then the frontend request and storage regressions, from the repository root:
```powershell
.\.venv\Scripts\python.exe -m pytest -q
Push-Location frontend
npm run test:decode; npm run test:storage
Pop-Location
```
Run only the end-to-end cases and write their machine-readable ledger to `regression-report.json`:
```powershell
.\.venv\Scripts\python.exe -m pytest -m regression -q
```
The JSON ledger records each end-to-end test's expected result, measured result, tolerance, pipeline stage, and pass/fail state. It is generated from test outcomes and is not the clean-corpus accuracy report.

## Overview
`ps26147_toolkit` is a Python package that provides a command‑line interface and a lightweight Streamlit UI for automated analysis of raw signal files (`.iq` and `.wav`).

Key features:
- Load and preprocess IQ and WAV recordings.
- Extract spectral features and estimate signal parameters (center frequency, bandwidth, SNR, baud‑rate).
- Classify modulation type using a **Random Forest** model trained on RadioML2016.10a (BPSK, QPSK, 8‑PSK, 16‑QAM).
- Produce a JSON report per file and an optional CSV roll‑up for batch runs.
- Streamlit demo for drag‑and‑drop processing with spectrogram visualisation.

## Installation
```powershell
# From the toolkit root directory
uv pip install -e .   # or: python -m venv venv && .\venv\Scripts\activate && pip install -e .
```

## Quick start (CLI)
```powershell
ps26147 path\to\signal.iq --output result.json   # single file
ps26147 path\to\folder\*.wav --csv batch.csv    # batch processing
```

## Streamlit demo
```powershell
streamlit run web_demo\app.py
```

## Training the model
If you want to retrain the classifier on the original RadioML2016.10a dataset:
```powershell
python scripts\download_datasets.py   # download data to `data/`
python scripts\train_classifier.py   # train RandomForest and save model.pkl
```

## Project structure
```
ps26147_toolkit/
  __init__.py
  preprocess.py
  feature_extractor.py
  parameter_extractor.py
  classifier.py
  cli.py

data/                # RadioML samples (downloaded by script)
scripts/
  download_datasets.py
  train_classifier.py
web_demo/
  app.py
tests/
  test_preprocess.py
  test_feature_extractor.py
requirements.txt
pyproject.toml
README.md
```
