# SIH Signal Analysis Dashboard — Frontend + Backend Roadmap & Status

**Project**: Parametric RF Signal Processing, Modulation Classification, Demodulation & Multi-Stage Waveform Decoding Suite.  
**Current State**: 
- Core Toolkit: `ps26147_toolkit/` (Phase 1 to Phase 6 complete)
- Internal Tool: `web_demo/app.py` (Streamlit internal diagnostic suite, untouched)
- REST API Backend: `api/` (FastAPI service wrapping core signal processing, parameter estimation, modulation classification, and FEC decoding)
- Modern Web Client: `frontend/` (Next.js 14+ App Router, Tailwind CSS, Recharts, tri-state theme system, drag-and-drop uploader)

---

## 1. Architecture Overview

The repository separates the signal-processing core, the internal diagnostic tool, the production REST API, and the Next.js web application:

```
ps26147_toolkit/
├── ps26147_toolkit/       # Core Signal Processing Library (Phase 1 to 6)
│   ├── preprocess.py      # IQ & WAV loader, dtype detection, metadata
│   ├── filters.py         # DC removal, IQ imbalance, spectral denoise
│   ├── parameter_extractor.py # SNR (M2M4), Baud rate, center freq, bandwidth
│   ├── classifier.py      # Cumulant features & modulation classification
│   ├── demodulator.py     # Gardner TED sync, Costas PLL, EVM, soft LLR
│   ├── fec_decoders.py    # Viterbi (K=7), Reed-Solomon, Concatenated, LDPC
│   └── correlator.py      # Frame sync, preamble discovery, bit correlation
├── web_demo/app.py        # Streamlit internal debug tool (untouched)
├── api/                   # Production FastAPI REST Backend
│   ├── main.py            # FastAPI entrypoint, CORS configuration
│   ├── schemas.py         # Typed Pydantic request/response schemas
│   ├── utils.py           # Multi-format signal loader (WAV, IQ, SigMF)
│   ├── routes/
│   │   ├── process.py     # /process/file (Full pipeline: params, PSD, constellation)
│   │   ├── classify.py    # /classify/ (Modulation classification + cumulants)
│   │   └── decode.py      # /decode/ (Demodulation + FEC decoding)
│   └── requirements.txt   # FastAPI, Uvicorn, Pydantic, Python-Multipart
├── frontend/              # Next.js 14+ App Router Web Dashboard
│   ├── app/
│   │   ├── layout.tsx     # Root App Router Layout & theme configuration
│   │   ├── globals.css    # Stellar design tokens & Tailwind directives
│   │   ├── page.tsx       # Home upload page with drag-and-drop & fs selection
│   │   ├── results/       # Multi-tab visualization dashboard (Waveform, PSD, IQ)
│   │   └── about/         # Architecture & toolkit documentation
│   ├── components/
│   │   ├── Layout.tsx     # Header, navbar tabs, and status indicator
│   │   └── ThemeToggle.tsx# System / Dark / Light theme switcher
│   └── services/
│       └── api.ts         # Type-safe API client wrapper
├── scripts/               # Training, benchmarking, and accuracy reports
└── tests/                 # Comprehensive test suite for toolkit & API
```

---

## 2. API Contract & Endpoints

| Endpoint | Method | Input | Output / Payload | Status |
|---|---|---|---|---|
| `GET /health` | GET | None | `{"status": "ok"}` | ✅ Completed |
| `POST /process/file` | POST | Multipart Form (`file`, `fs`) | `ProcessResponse` (Modulation, Confidence, Baud, SNR dB, Bandwidth, Waveform points, PSD spectrum, Constellation scatter) | ✅ Completed |
| `POST /process/async` | POST | Multipart Form (`file`, `fs`) | `AsyncJobResponse` (`job_id`, `status: "queued"`, `message`) | ✅ Completed |
| `GET /process/status/{job_id}` | GET | Path (`job_id`) | `JobStatusResponse` (`job_id`, `status`, `progress`, `stage`, `result`, `error`) | ✅ Completed |
| `POST /classify/` | POST | Multipart Form (`file`, `fs`) | `ClassifyResponse` (Modulation, Confidence, Cumulants & Features dictionary) | ✅ Completed |
| `POST /decode/` | POST | Multipart Form (`file`, `fs`, `fec_scheme`) | `DecodeResponse` (Bit count, Bit string preview, Hex dump, EVM dB/%, Decoded bits) | ✅ Completed |

---

## 3. Frontend Pages & Features

1. **Home / Uploader (`/`)**:
   - Drag-and-drop file upload zone supporting `.wav`, `.iq`, `.raw`, `.bin`, and `.sigmf-data`.
   - Sample rate selector (`1 MHz`, `2 MHz`, `2.4 MHz`, `5 MHz`, `10 MHz`, `44.1 kHz`, `48 kHz`).
   - Animated multi-stage processing indicator (`Ingesting -> Extracting -> Classifying`).
2. **Analysis Dashboard (`/results`)**:
   - Top metric summary cards with neon accent indicators: Modulation, Confidence %, Estimated Baud Rate, In-Band SNR (dB), Occupied Bandwidth (kHz/MHz), Center Frequency.
   - Interactive tabbed visualization suite:
     - **Time-Domain Waveform**: Real-time amplitude plot with sample index.
     - **Power Spectral Density (PSD)**: Welch spectral estimate in dB/Hz.
     - **I/Q Constellation Diagram**: 2D scatter plot of normalized baseband symbols.
   - Detailed signal telemetry grid (Sample count, duration, 3dB bandwidth).
3. **About & Documentation (`/about`)**:
   - High-level overview of Phase 1 through Phase 6 signal processing pipeline.
   - Live REST API endpoint reference.

---

## 4. Local Development Guide

### Running the FastAPI Backend
```bash
# Activate virtual environment
.venv\Scripts\activate

# Start API server on port 8000
python -m uvicorn api.main:app --reload --port 8000
```
Backend Swagger Documentation available at `http://localhost:8000/docs`.

### Running the Next.js Frontend
```bash
cd frontend

# Install dependencies (if first time)
npm install

# Start development server on port 3000
npm run dev
```
Open `http://localhost:3000` in the browser.

---

## 5. Implementation Roadmap Checklist

- [x] **1. FastAPI Backend Setup**: Created `api/` directory with `main.py`, `schemas.py`, and `utils.py`.
- [x] **2. Parameter & Classification Integration**: Connected `ps26147_toolkit` modules (`parameter_extractor`, `classifier`, `demodulator`, `fec_decoders`).
- [x] **3. CORS & Middleware**: Configured CORS for frontend `http://localhost:3000`.
- [x] **4. Next.js Scaffold**: Initialized Next.js with TypeScript, Tailwind CSS, Recharts, and SWR.
- [x] **5. "Stellar" SaaS Design System**: Implemented dark theme palette, neon cyan/magenta/blue accents, glassmorphism cards, and tri-state theme toggle (`system` / `dark` / `light`).
- [x] **6. Core Web Pages**: Built Home (`/`), Results (`/results`), and About (`/about`).
- [x] **7. Multi-Format File Ingestion**: WAV RIFF parsing, IQ stream auto-detection, and SigMF handling.
- [x] **8. Multi-Chart Results Dashboard**: Waveform line chart, Welch PSD area chart, and I/Q constellation scatter plot.
- [x] **9. Automated Test Verification**: End-to-end API test suite validated across all endpoints.
- [x] **10. Async Background Processing & Status Polling**: In-memory task queue, `/process/async` submission, live percentage progress bar, and `/process/status/{job_id}` polling.
- [ ] **11. Live Constellation Streaming / WebSocket (Optional Enhancement)**: Real-time SDR streaming via WebSockets.
