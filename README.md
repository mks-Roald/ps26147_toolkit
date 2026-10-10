# PS26147 Toolkit — Automated `.IQ` / `.wav` Signal Analysis

**Smart India Hackathon 2026 · Problem Statement PS26147**

A toolkit and web dashboard that takes a raw signal recording (`.iq` or `.wav`) and works out *what it is*: its sampling/centre frequency, bandwidth, SNR, baud rate and modulation. It then demodulates it to a bit stream, de-interleaves, applies forward error correction (FEC), and correlates the bits to find headers (sync words) and payload.

It ships as:

| Interface | Use it for |
|---|---|
| **Web dashboard** (Next.js, port 3000) | Upload a file, see spectrum / constellation / waterfall / bits. *This is the main interface for evaluators.* |
| **REST + WebSocket API** (FastAPI, port 8000) | The backend that does the analysis; scriptable with `curl`/Python |
| **CLI** (`ps26147`) | Batch-analyse files into JSON/CSV |
| **Python library** (`ps26147_toolkit`) | Import the stages directly |
| **Streamlit demo** (legacy) | Older single-file GUI |

> **Read [Known limitations & disclaimers](#10-known-limitations--disclaimers) before relying on any result.** This is a hackathon prototype.

---

## Contents

1. [Features vs. problem statement](#1-features-vs-problem-statement)
2. [How it works](#2-how-it-works)
3. [Setup for evaluators (any device)](#3-setup-for-evaluators-any-device)
4. [Test data and datasets](#4-test-data-and-datasets)
5. [Using the website](#5-using-the-website)
6. [Command line and Python usage](#6-command-line-and-python-usage)
7. [API reference](#7-api-reference)
8. [Configuration](#8-configuration)
9. [Testing](#9-testing)
10. [Known limitations & disclaimers](#10-known-limitations--disclaimers)
11. [Troubleshooting](#11-troubleshooting)
12. [Project structure](#12-project-structure)
13. [Retraining the classifier (optional)](#13-retraining-the-classifier-optional)
14. [Roadmap, team, licence](#14-roadmap-team-licence)

---

## 1. Features vs. problem statement

| Problem-statement task | What the project provides | Code |
|---|---|---|
| **i. Identify parameters** (sampling frequency, modulation, FEC, interleaving) | Centre frequency, bandwidth (several contours, OBW 95/99 %), in-band SNR, baud rate, modulation class with confidence. Sample rate is read from the WAV header, or supplied by the user for raw `.iq` (raw files carry none). FEC / interleaver are detected heuristically. | `parameter_extractor.py`, `classifier.py`, `preprocess.py` |
| **ii. Demodulate** FSK / QAM / PSK | BPSK, QPSK, 8PSK, 16QAM, 64QAM, 2FSK, 4FSK. Symbol timing recovery, carrier recovery, Gray slicing, EVM, soft LLRs for FEC. | `demodulator.py` |
| **iii. De-interleave** (block, convolutional, diagonal, pseudo-random) | All four methods with explicit parameters, plus auto-detection. | `deinterleaver.py` |
| **iv. FEC** (convolutional + Viterbi, RS, concatenated, LDPC) | Viterbi (K=7, rate ½, hard & soft), Reed-Solomon (255,223) over GF(2⁸), concatenated Viterbi + RS, LDPC (min-sum). | `fec_decoders.py`, `ldpc_matrices.py` |
| **v. Bit-stream correlation** | Sliding bipolar cross-correlation with standard sync words, 180° inversion handling, periodic-frame detection, preamble auto-discovery, frame slicing. | `correlator.py` |
| **GUI**: spectrum, constellation, waterfall | Dashboard with waveform, PSD, constellation, spectrogram and waterfall views, bit/hex/ASCII viewer, bit download. | `frontend/` |
| Input as `.iq` or `.wav` | Raw `.iq` (auto-detected dtype), SigMF companion metadata, `.wav`. | `preprocess.py` |

Extras: DC-offset removal, adaptive band-pass and spectral-subtraction denoising (`filters.py`); async job processing; a **simulated** live stream page; JSON / CSV reports; ground-truth test-signal generator and accuracy report.

---

## 2. How it works

```
 .iq / .wav
     │
     ▼
 ┌──────────────┐   dtype auto-detect (cf32/int16/int8/uint8), SigMF metadata,
 │  preprocess  │   WAV header sample-rate, optional resample
 └──────┬───────┘
        ▼
 ┌──────────────────────┐  DC removal → Welch PSD → centre freq, bandwidth,
 │  analysis (analyze_  │  in-band SNR, baud rate → down-convert to baseband →
 │  signal)             │  cumulant features → modulation classifier
 └──────┬───────────────┘
        ▼
 ┌──────────────┐  timing recovery → carrier recovery (PSK/QAM) → normalise →
 │ demodulator  │  slice to bits (Gray) → EVM + soft LLRs     (FSK: frequency domain)
 └──────┬───────┘
        ▼
 ┌──────────────┐  sync-word / preamble search; bits after the sync word are kept
 │  correlator  │
 └──────┬───────┘
        ▼
 ┌──────────────┐  optional, accepted only when it measurably improves the
 │deinterleaver │  Viterbi metric (when auto mode is used)
 └──────┬───────┘
        ▼
 ┌──────────────┐  none | viterbi | reed-solomon | concatenated | ldpc
 │ fec_decoders │
 └──────┬───────┘
        ▼
   JSON + plot data  ──►  FastAPI  ──►  Next.js dashboard
```

**Stage notes**

- **Parameter extraction.** Centre frequency and bandwidth come from a Welch PSD; bandwidth uses a per-modulation calibrated dB contour. SNR is integrated in-band. Baud rate comes from a cyclic envelope / transition estimator.
- **Classification.** Higher-order cumulants (C20, C40, C42, C63) feed a Random Forest (`model.joblib`), with a rule-based fallback. Out-of-domain captures (negative SNR, noise-like full-band spectrum) are reported as `UNKNOWN` rather than forced to a label.
- **Demodulation.** Gardner timing recovery with fractional interpolation, a Costas loop (order 2 / 4 / 8), energy normalisation, Gray-coded slicing, EVM, and LLR generation. A run-length check tries to fix baud-rate sub-multiples. FSK skips carrier recovery and works from instantaneous frequency.
- **Sync words** built in: Barker-7/11/13, CCSDS-32 (`0x1ACFFC1D`), DVB-S (`0x47`) and its inverse (`0xB8`), ZigBee SFD (`0xA7`), WiFi SFD (`0xF3A0`), a 16-bit `1010…` preamble. You can also pass any hex (`1ACFFC1D`, `0x47`) or binary string.
- **De-interleaving.** Block (R×C), convolutional (Forney/Ramsey), diagonal, pseudo-random (PRBS seed). Auto-detect scores candidate outputs by byte entropy / run-length / Viterbi metric.
- **Order inside `/decode`:** analysis → demodulate → sync → (optional) de-interleave → FEC.

---

## 3. Setup for evaluators (any device)

Works on Windows, macOS and Linux. Choose **one** of: [A. Manual (recommended)](#a-manual-setup-recommended) or [B. Docker](#b-docker-compose).

### 3.0 Prerequisites

| Tool | Version | Check | Notes |
|---|---|---|---|
| **Git** | any recent | `git --version` | |
| **Git LFS** | 3.x | `git lfs version` | Needed for the sample `.iq` / `.wav` files (see [§4](#4-test-data-and-datasets)) |
| **Python** | **3.11 recommended** | `python --version` | The project pins NumPy, SciPy and scikit-learn versions. Python 3.11 is the best-supported choice; compatibility with newer Python versions is not guaranteed. |
| **Node.js** | 20.9 or newer (22 recommended) | `node --version` | Dashboard only. The frontend tests need Node ≥ 22.6. |
| **npm** | 10+ | `npm --version` | Comes with Node |
| Docker + Compose | optional | `docker compose version` | Option B only |

Disk/time: a first `pip install` downloads a large scientific stack (and currently `torch`, which is declared as a dependency but is not imported by the code); allow several GB and a few minutes on a good connection.

### A. Manual setup (recommended)

**Step 1 — Get the code and the data files**

```bash
git lfs install                     # once per machine
git clone https://github.com/mks-Roald/ps26147_toolkit.git
cd ps26147_toolkit
git lfs pull                        # downloads the real .iq / .wav files
```

> Do **not** use GitHub's "Download ZIP": it contains only ~130-byte LFS pointer text files instead of the recordings.
>
> To update later: `git pull && git lfs pull`.

**Step 2 — Python environment and backend dependencies**

Linux / macOS:
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

Windows (PowerShell):
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1        # if blocked: Set-ExecutionPolicy -Scope Process Bypass
python -m pip install --upgrade pip
pip install -e .
```

Windows (cmd): activate with `.venv\Scripts\activate.bat`.

This installs the library, the `ps26147` command and the API dependencies.

**Step 3 — Start the backend (terminal 1, venv active, repo root)**

```bash
python -m uvicorn api.main:app --reload --port 8000
```

Check it: open <http://localhost:8000/health> → `{"status":"ok"}`. Interactive API docs: <http://localhost:8000/docs>.

**Step 4 — Start the dashboard (terminal 2)**

```bash
cd frontend
npm install
npm run dev
```

Open **<http://localhost:3000>**.

The frontend talks to `http://localhost:8000` by default. To change that, copy `frontend/.env.example` to `frontend/.env.local` and edit `NEXT_PUBLIC_API_URL`.

**Step 5 — Analyse a signal.** Follow [§5](#5-using-the-website). No LFS files? Generate test signals offline, see [§4](#4-test-data-and-datasets).

### B. Docker Compose

```bash
git lfs install
git clone https://github.com/mks-Roald/ps26147_toolkit.git
cd ps26147_toolkit
git lfs pull
docker compose up --build -d
docker compose ps                   # backend should become "healthy"
```

- Dashboard: <http://localhost:3000> · API: <http://localhost:8000> (docs at `/docs`)
- Logs: `docker compose logs -f` · Stop: `docker compose down`
- The first build is slow (Python scientific stack plus the Next.js build). More options are in [`DEPLOYMENT.md`](DEPLOYMENT.md).
- `NEXT_PUBLIC_API_URL` is **baked in at frontend build time**; if you change it, run `docker compose up --build` again.
- If the frontend container fails with a message about `output: export` and `next start`, use Option A (dev server) or serve the static build with `npx serve out`.

### C. Using the dashboard from a different device on the same network

On the host machine:

```bash
# backend: listen on all interfaces and allow the dashboard origin
export CORS_ORIGINS="http://<HOST_IP>:3000"          # PowerShell: $env:CORS_ORIGINS="http://<HOST_IP>:3000"
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000

# frontend: point it at the host's backend, listen on all interfaces
cd frontend
echo NEXT_PUBLIC_API_URL=http://<HOST_IP>:8000 > .env.local
npm run dev -- -H 0.0.0.0
```

Then browse to `http://<HOST_IP>:3000` from the other device. Allow ports 3000 and 8000 through the host firewall.

### D. Backend/CLI only (no website)

Steps 1–2 of Option A are enough. Then use the [CLI](#6-command-line-and-python-usage) or the [API](#7-api-reference).

---

## 4. Test data and datasets

### What is in the repository

| Path | Contents |
|---|---|
| `_cal/` | Calibration corpus: 7 clean `.wav` files (BPSK, QPSK, 8PSK, 16QAM, 64QAM, 2FSK, 4FSK) each with a ground-truth `.json` |
| `bpsk_hello_world_viterbi_16x16_1Msps_25kbaud.iq`, `qpsk_hello_world_…` | Sample recordings; the file name indicates Viterbi FEC, a 16×16 interleaver, 1 Msps and 25 kbaud |
| `synthetic.iq`, `scripts/synthetic.iq` | A plain complex tone (not a modulated signal) |
| `model.joblib` | The pre-trained modulation classifier (a normal file, not LFS) |

All `*.iq` and `*.wav` files are stored with **Git LFS** (`.gitattributes`). Without `git lfs pull` they are tiny text pointers beginning `version https://git-lfs.github.com/spec/v1` and fail to load.

### No dataset download is needed to run the website

The classifier ships pre-trained. The RadioML download in [§13](#13-retraining-the-classifier-optional) is only for retraining.

### If Git LFS is unavailable (quota exceeded, blocked, or ZIP download)

Generate known test signals locally. This needs no network:

```bash
python scripts/generate_ground_truth_corpus.py --corpus --out ground_truth
```

This writes a `.wav` **and** a `.iq` for each of the 7 modulations, plus a `.json` with the true parameters.

Expected values for the generated files:

| Parameter | Value |
|---|---|
| Sample rate | **48,000 Hz** (in the UI, pick *48,000 Hz* for the `.iq` files; WAV is read automatically) |
| Carrier (centre) frequency | ≈ 10,000 Hz |
| Baud rate | 1,200 |
| Payload (ASCII) | `the quick brown fox jumps over the lazy dog 0123456789` repeated 3× |
| FEC / interleaving / sync | none |

Quick check from the command line (this was run on a generated QPSK file and returned QPSK, ≈1199 baud, ≈9989 Hz):

```bash
ps26147 ground_truth/qpsk_clean_iq.iq --fs 48000 --output out.json
```

### Making your own recordings

A GNU Radio flowgraph file is **not** a recording. To record: **File Sink** (type `complex`, after a **Head** block to limit length) writes raw cf32 I/Q; for WAV use *Complex to Float* → 2-channel *Wav File Sink*. Raw files have no header, so write down the sample rate. See also [WAV caveat](#10-known-limitations--disclaimers) before using 2-channel WAV.

### Accuracy report

```bash
python scripts/run_accuracy_report.py --corpus _cal      # writes accuracy_report.md / accuracy_summary.json
```

---

## 5. Using the website

Navigation tabs: **Upload & Process** (`/`), **Live SDR Sim** (`/live`), **Dashboard Results** (`/results`), **About & Toolkit** (`/about`). There is a light/dark theme toggle in the header.

### Step by step

1. Open <http://localhost:3000> and **drag & drop** (or browse for) a `.iq` / `.wav` file. Accepted extensions: `.wav .iq .bin .raw .sigmf-data`.
2. **Sample rate.**
   - `.wav`: detected automatically from the file header.
   - `.iq`: **you must select the correct rate** (presets 1, 2, 2.4, 5, 10 MHz, 44.1 kHz, 48 kHz, or *Custom…*). Raw files carry no rate, and a wrong value gives wrong frequencies and baud rates without any error.
3. **Optional decoding settings**
   - **FEC scheme:** None / Viterbi / Reed-Solomon / Concatenated / LDPC.
   - **Sync word:** a name such as `Barker-13`, `CCSDS-32`, or a hex value like `1ACFFC1D`.
   - **Auto sync:** search for a known sync word or a repeating preamble automatically.
   - **Auto deinterleave:** try de-interleavers; only accepted when the Viterbi metric improves (needs FEC = Viterbi).
4. Click the analyse button and watch the progress stages (async processing is on by default).
5. The **Results** page shows:
   - Summary cards: modulation (with confidence), baud rate, SNR, centre frequency, bandwidth.
   - Tabbed plots: **Waveform**, **PSD spectrum**, **Constellation** (or recovered-frequency view for FSK), **Spectrogram**, **Waterfall**.
   - **Demodulated Bitstream**: bits, with download.
   - **Deinterleaved Output** (when produced).
   - **FEC Decoded Output** (or "Processed Bits (FEC not run)"), with bit / hex / ASCII views.
   - **Correlation / Sync Results**: sync word found, correlation peak, frames, header vs payload.
   - A signal metadata table (samples, duration, sample rate, etc.).

### Suggested evaluation walkthrough

1. Upload `_cal/bpsk_clean_wav.wav` (or the generated one) → expect **BPSK**, ≈1200 baud, ≈10 kHz. Look at the 2-cluster constellation.
2. Repeat with `qpsk`, `8psk`, `16qam`, `64qam` and compare constellations. Try `2fsk` / `4fsk` and view the frequency states.
3. Open the spectrogram and waterfall tabs.
4. Upload `bpsk_hello_world_viterbi_16x16_1Msps_25kbaud.iq`, set sample rate **1,000,000 Hz**, FEC **Viterbi**, tick **Auto deinterleave** and **Auto sync**, and inspect the decoded output and correlation results. *(The file name describes the intended parameters; confirm the output on your copy.)*
5. Open **Live** to see the streaming demo.

### Live page

`/live` connects to `ws://localhost:8000/stream/live` and shows a stream of **simulated** SDR frames (the server generates a synthetic signal with the modulation, SNR, baud rate and carrier offset you choose, then runs the same analysis on it). It does **not** read real SDR hardware. Use it to see the classifier respond as you change the controls.

---

## 6. Command line and Python usage

### CLI

```bash
ps26147 capture.iq --fs 2048000 --output result.json       # one file → JSON
ps26147 "folder/*.wav" --csv batch.csv                      # many files → CSV
ps26147 folder/                                              # all .iq/.wav under a folder
ps26147 capture.iq --fs 2048000 --filter --denoise --plot   # band-pass + denoise + spectrogram
```

| Flag | Meaning |
|---|---|
| `--fs <Hz>` | Sample rate for `.iq` (**default 1,000,000 — always set it explicitly**). Ignored for `.wav`. |
| `--filter` / `--denoise` | Adaptive band-pass / spectral-subtraction denoise |
| `--plot` | Show the spectrogram |
| `--output <file>` | JSON path for a single file (default `output.json`) |
| `--csv <file>` | Batch summary; processes every matched file |

Write plain numbers (`2048000`), not `2_048_000`.

### Python

```python
from ps26147_toolkit.preprocess import load_iq
from ps26147_toolkit.analysis import analyze_signal
from ps26147_toolkit import demodulator

sig, meta = load_iq("capture.iq", fs=2_048_000)
a = analyze_signal(sig, meta.fs)
print(a.parameters)        # centre freq, bandwidth, snr_db, baud_rate, ...
print(a.classification)    # modulation, confidence

d = demodulator.demodulate_signal(
    sig, meta.fs, a.classification["modulation"],
    center_freq=a.parameters["center_frequency_hz"],
    baud_rate=a.parameters["baud_rate"],
)
print(d["evm_percent"], d["hex_preview"])
```

### Streamlit demo (legacy)

```bash
streamlit run web_demo/app.py
```

---

## 7. API reference

Base URL `http://localhost:8000` · Swagger UI at `/docs`. Every file endpoint takes `multipart/form-data` with a `file` field. Contract notes are in [`api/api.md`](api/api.md).

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Health probe |
| `/process/file` | POST | Full analysis with plot data (PSD, waveform, constellation, waterfall) |
| `/process/session` | POST | One analysis plus first decode stage (used by the dashboard) |
| `/process/async` | POST | Start a background job → `job_id` |
| `/process/status/{job_id}` | GET | Poll a job |
| `/classify/` | POST | Modulation, confidence, features, parameters |
| `/decode/` | POST | Demodulate → sync → de-interleave → FEC |
| `/correlate/`, `/correlate/file` | POST | Sync-word / preamble search on a file |
| `/correlate/bits` | POST | Same on a JSON bit array |
| `/correlate/sync-words` | GET | List built-in sync words |
| `/stream/live` | WebSocket | Simulated live frames; send `{"action":"configure", "modulation":"QPSK", "snr_db":20, "baud_rate":50000, "cfo_hz":500}`, `pause`, `resume` |

Common query parameters:

| Parameter | Default | Meaning |
|---|---|---|
| `fs` | `1000000` | Sample rate for `.iq`. For `.wav` it is the **target** rate the audio is resampled to, so leave it equal to the WAV's own rate. |
| `fec_scheme` | `none` | `none`, `viterbi`, `reed-solomon`, `concatenated`, `ldpc` |
| `sync_word` | — | Name, hex (`1ACFFC1D`, `0x47`) or binary string |
| `auto_detect_sync` | `true` | Search for a preamble |
| `auto_deinterleave` | `false` | Metric-gated, with Viterbi |
| `modulation` | — | Override the classifier (`/decode/`), e.g. `BPSK`, `2FSK` |

Examples:

```bash
curl -F "file=@ground_truth/qpsk_clean_wav.wav" "http://localhost:8000/classify/?fs=48000"

curl -F "file=@capture.iq" \
  "http://localhost:8000/decode/?fs=1000000&fec_scheme=viterbi&auto_deinterleave=true&sync_word=CCSDS-32"

curl http://localhost:8000/correlate/sync-words
```

---

## 8. Configuration

| Variable | Where | Default | Purpose |
|---|---|---|---|
| `CORS_ORIGINS` | backend | `http://localhost:3000,http://127.0.0.1:3000` | Comma-separated allowed browser origins (`*` allows all) |
| `PORT` | backend (Docker) | `8000` | Listen port |
| `NEXT_PUBLIC_API_URL` | frontend | `http://localhost:8000` | Backend URL used by the browser (read at build/start time) |

Samples: `.env.example` (root) and `frontend/.env.example`.

---

## 9. Testing

The test dependencies are not installed by `pip install -e .`, so add them first:

```bash
pip install pytest httpx
python -m pytest -q                      # unit + API tests
python -m pytest -m regression -q        # end-to-end cases → regression-report.json

cd frontend
npm run test:decode && npm run test:storage      # requires Node 22.6+
```

Some tests read the sample recordings, so run `git lfs pull` first. The `regression-report.json` ledger records expected vs measured results per test; it is not the clean-corpus accuracy report.

---

## 10. Known limitations & disclaimers

**General**

- **Prototype, not certified.** Outputs are best-effort estimates. Do not use them as the sole basis for operational, safety-critical or legal decisions.
- **Lawful use only.** Analyse only signals you are authorised to receive and decode. Intercepting or decoding communications may be restricted where you are.
- **No hardware capture.** The toolkit analyses recordings; the *Live* page is a simulation.

**Accuracy**

- The classifier is trained on **simulated** data (RadioML2016.10a and generated signals). Accuracy on real over-the-air captures with other filters, sample rates and hardware impairments can be much lower.
- `accuracy_report.md` (42/42 checks) covers **clean synthetic signals from the project's own generator** at 48 kHz. It does not check payload recovery and is not evidence of real-world accuracy.
- In an independent check using separately generated root-raised-cosine signals, modulation was identified correctly for BPSK / QPSK / 8PSK (including at 10–15 dB SNR) and centre frequency / SNR were close. **The baud-rate estimator sometimes returned exactly half the true rate** (e.g. 12.5 k for 25 k baud), which breaks demodulation, and **16QAM was labelled 64QAM** in one case. If the decoded bits look random, retry with the baud rate ×2 or ×0.5.
- Dense QAM orders and low SNR reduce reliability for every class.

**Input handling**

- **Raw `.iq` has no header.** A wrong sample rate or data type produces wrong parameters silently. The CLI/API default of 1 MHz exists only as a fallback.
- **2-channel `.wav`:** the loader currently **averages the channels into one real signal** rather than treating channel 0 / 1 as I / Q. Recordings made as stereo I/Q WAV (as some SDR programs produce) lose their quadrature information. Prefer `.iq` for such captures.
- WAV audio is resampled to the `fs` query value; keep it equal to the file's own rate.
- Demodulation returns at most **4,096 symbols** per call, so long recordings are truncated for bit-level work.

**Decoding**

- FEC and interleaver detection is **heuristic**: it chooses the best-scoring candidate and can pick wrongly or find nothing. Supplying known parameters (FEC scheme, sync word) gives better results.
- LDPC uses the matrices in `ldpc_matrices.py`; other code families need their own matrices.

**Packaging**

- Dependency pins are strict (numpy 1.26.4, scipy 1.13.1, scikit-learn 1.4.2, joblib 1.4.2) because `model.joblib` was saved with them. Other versions may fail or warn when loading it.
- `pyproject.toml` also lists `torch`, `torchvision`, `librosa`, `soundfile`, `tqdm` and `uv`, which the package code does not import; they only make installation larger and slower.
- Frequency range: tested on synthetic baseband/low-IF signals. HF/VHF/UHF captures must be recorded (or down-converted) at a suitable sample rate first.

---

## 11. Troubleshooting

| Symptom | Likely cause → fix |
|---|---|
| `could not find a version that satisfies numpy==1.26.4` (or build errors) | Python 3.13+. Install **Python 3.11** and recreate the venv. |
| Upload fails or file "can't be decoded"; file is ~130 bytes | LFS pointer. Run `git lfs install && git lfs pull`. Not a ZIP download. |
| `git lfs pull` says bandwidth/quota exceeded | Generate local test signals ([§4](#if-git-lfs-is-unavailable-quota-exceeded-blocked-or-zip-download)). |
| Dashboard shows "Failed to fetch" / network error | Backend not running, wrong `NEXT_PUBLIC_API_URL`, or CORS. Check <http://localhost:8000/health>; set `CORS_ORIGINS` to the dashboard's exact origin. |
| `Address already in use` | Port 8000/3000 busy: stop the other process or use `--port 8001` and update `NEXT_PUBLIC_API_URL`. |
| `ps26147: command not found` | venv not active, or run `pip install -e .` again. |
| PowerShell won't activate the venv | `Set-ExecutionPolicy -Scope Process Bypass`, or use cmd with `activate.bat`. |
| Warning/error when loading `model.joblib` | scikit-learn version differs from the pin. Reinstall with `pip install -e .`. |
| Parameters look wrong for an `.iq` file | Wrong sample rate (or dtype). Re-upload with the correct rate. |
| Decoded bits look random | Baud estimate off by ×2 / ×½, wrong modulation label, wrong FEC scheme, or no sync. |
| `npm install` fails | Use Node 20.9+; delete `node_modules` and retry. |
| Frontend tests fail with `--experimental-strip-types` | Node older than 22.6. |
| `pytest` / `TestClient` import errors | `pip install pytest httpx`. |

---

## 12. Project structure

```
ps26147_toolkit/        Core library
  preprocess.py           .iq / .wav / SigMF loaders, SignalMetadata
  filters.py              DC removal, band-pass, denoise
  feature_extractor.py    PSD, spectrogram
  parameter_extractor.py  centre freq, bandwidth, SNR, baud rate
  classifier.py           cumulant features + Random Forest (+ rule-based fallback)
  analysis.py             canonical analyze_signal() pipeline
  demodulator.py          timing/carrier recovery, slicing, EVM, LLRs
  deinterleaver.py        block / convolutional / diagonal / pseudo-random
  fec_decoders.py         Viterbi, Reed-Solomon, concatenated, LDPC
  ldpc_matrices.py        LDPC parity-check matrices
  correlator.py           sync words, correlation, framing, preamble discovery
  cli.py                  `ps26147` command
api/                    FastAPI app (main.py, routes/, schemas.py, utils.py, api.md)
frontend/               Next.js dashboard (app/, components/, services/, tests/)
web_demo/               Legacy Streamlit app
scripts/                download_datasets, train_classifier, generate_ground_truth_corpus,
                        run_accuracy_report, generate_synthetic_iq
tests/                  pytest suite
_cal/                   Ground-truth calibration corpus (Git LFS)
docs/                   Design notes and phase summaries
model.joblib            Pre-trained classifier
Dockerfile.api, docker-compose.yml, DEPLOYMENT.md, netlify.toml
ROADMAP.md, accuracy_report.md, PHASE6_ACCURACY_HARDENING_PLAN.md
```

---

## 13. Retraining the classifier (optional)

Not needed to run the website. To retrain on RadioML2016.10a (requires internet; the dataset is distributed by DeepSig under its own licence terms, so check them before redistributing models trained on it):

```bash
python scripts/download_datasets.py    # downloads and extracts into data/
python scripts/train_classifier.py     # writes model.joblib
```

`data/` is git-ignored.

---

## 14. Roadmap, licence and acknowledgements

**Status and plans:** see [`ROADMAP.md`](ROADMAP.md). Pending items there include higher-resolution waterfall/time-domain viewers, constellation/EVM accuracy tweaks, classifier tuning and a step-by-step live execution log. Design and phase notes are in [`docs/`](docs/).

**Licence:** `pyproject.toml` declares MIT. The repository currently has no root `LICENSE` file.

**Acknowledgements:** DeepSig (RadioML), GNU Radio, NumPy / SciPy / scikit-learn, FastAPI, Next.js.