# SIH Signal Analysis Dashboard — Frontend + Backend Build Roadmap

**Project**: Signal processing pipeline (baud rate, SNR, modulation classification, waveform decode)
**Current state**: `web_demo/app.py` (Streamlit, internal-only) + `ps26147_toolkit/` (core signal-processing library) + `scripts/` + `tests/`. **Koi FastAPI backend abhi repo mein nahi hai** — ye fresh banana hoga.
**Target**: Streamlit ko touch nahi karna (internal debug tool ke liye rakhna hai) — ek naya, separate polished frontend (React + Next.js, Astro.js option bhi) banana hai jo naye FastAPI backend se baat kare.
**Design direction**: "Stellar" style — dark dashboard, metric cards (Baud Rate / SNR / Modulation), waveform chart, neon accent glow. System-default + light/dark toggle.

---

## Step 0 — Architecture Decision (sabse pehle ye tay karo)

- **Streamlit** (`web_demo/app.py`) → sirf internal/debug tool banega. Isko touch mat karo, isse public-facing mat banao.
- **Naya frontend** → separate folder mein, React + Next.js (ya Astro.js) + Tailwind. Ye naye FastAPI backend ko REST/fetch se call karega.
- **Naya backend** → `api/` folder mein FastAPI se fresh banega, jo `ps26147_toolkit/` ke functions (classifier, demodulator, decoder) ko REST endpoints ke roop mein expose karega.
- **Repo structure**: same repo, alag subfolders — `frontend/` (ya `web/`) aur `api/`. Ek hi git pull se sab milta hai, CI/CD simple rehta hai, local dev loop smooth (dono ek saath `localhost` pe chalte hain).

```
your-repo/
├── ps26147_toolkit/     # existing — signal processing core (untouched)
├── web_demo/app.py      # existing — Streamlit, internal only (untouched)
├── scripts/             # existing
├── tests/               # existing
├── api/                 # NEW — FastAPI backend
└── frontend/            # NEW — React/Next.js frontend
```

---

## Step 1 — Backend Setup (FastAPI) — ye pehle karo, kyunki frontend isी par depend karta hai

Repo mein abhi FastAPI nahi hai, isliye fresh setup karna hai.

### 1.1 Install & scaffold

```bash
mkdir api && cd api
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install fastapi uvicorn python-multipart
pip freeze > requirements.txt
```

### 1.2 Basic app structure

```
api/
├── main.py              # FastAPI app entrypoint
├── routes/
│   ├── process.py        # /process_file
│   ├── classify.py        # /classify
│   └── decode.py          # /decode
├── schemas.py            # Pydantic request/response models
└── requirements.txt
```

### 1.3 Minimal `main.py` with CORS enabled

CORS enable karna zaroori hai warna frontend (localhost:3000) backend (localhost:8000) ko call nahi kar payega.

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Signal Analysis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # prod mein apna frontend domain daalna
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}
```

### 1.4 Wrap existing toolkit as endpoints

`ps26147_toolkit/` ke existing functions (classifier, demodulator, FEC, correlator) ko REST endpoints mein expose karo — koi naya logic likhne ki zaroorat nahi, bas wrap karo:

```python
from fastapi import APIRouter, UploadFile, File
from ps26147_toolkit import classifier  # apna actual import path use karo

router = APIRouter()

@router.post("/process_file")
async def process_file(file: UploadFile = File(...)):
    contents = await file.read()
    # existing toolkit function ko call karo
    result = classifier.process(contents)
    return {
        "baud_rate": result.baud_rate,
        "snr": result.snr,
        "modulation": result.modulation,
        "waveform_data": result.waveform.tolist(),
    }
```

### 1.5 Document the API contract

Har endpoint ke liye ek chhota `api.md` likho (method, request shape, response shape) — ye frontend ki "dictionary" hai:

| Endpoint | Method | Request | Response |
|---|---|---|---|
| `/process_file` | POST | multipart/form-data (file) | `{baud_rate, snr, modulation, waveform_data[]}` |
| `/classify` | POST | JSON | `{modulation, confidence}` |
| `/decode` | POST | JSON | `{decoded_bits[]}` |
| `/status/:id` (agar async job hai) | GET | — | `{completed: bool, result?}` |

### 1.6 Run the backend

```bash
cd api
uvicorn main:app --reload --port 8000
```

---

## Step 2 — Frontend Scaffold

### Option A — React + Next.js (recommended)

```bash
npx create-next-app@latest frontend   # TypeScript, Tailwind, ESLint select karo
cd frontend
npm install framer-motion recharts   # animations + charts ke liye
npm install swr                       # data fetching/polling ke liye
```

### Option B — Astro.js (agar zyada static/fast-loading chahiye)

Astro static-first hai, interactive dashboard components ke liye "islands" architecture use karta hai — agar mostly-static site chahiye with kuch interactive widgets (upload zone, charts), toh ye fit baithta hai:

```bash
npm create astro@latest frontend
cd frontend
npx astro add react tailwind    # React islands + Tailwind add karo
npm install framer-motion recharts swr
```

> **Kaunsa choose karo?** Agar poora dashboard interactive/data-heavy hai (real-time updates, complex state) → **Next.js** better fit hai. Agar landing page + kुछ interactive sections hi hain → **Astro** faster load times deta hai. Is project ke liye (dashboard-heavy) Next.js zyada natural hai.

### API wrapper banao

`frontend/services/api.ts`:

```typescript
const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export async function processFile(file: File) {
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch(`${API_BASE}/process_file`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) throw new Error("Processing failed");
  return res.json();
}
```

---

## Step 3 — Design System ("2026 SaaS" — Stellar style, dark + neon)

Ek `styles/tokens.css` ya Tailwind config mein ye design tokens define karo, poore app mein reuse hoga:

| Element | Value |
|---|---|
| Background (dark) | `#0a0a0a` / `#111` |
| Accent colors | cyan `#00ffff`, magenta `#ff00ff`, electric blue `#00bfff` |
| Typography | Headings: Inter / Space Grotesk. Code/readouts: JetBrains Mono |
| Spacing | 8-point grid |
| Cards | subtle shadow + `backdrop-filter: blur(8px)` |
| Motion | button ripple, hover lift, fade transitions, loading skeletons (framer-motion) |
| Accessibility | contrast ≥ 4.5:1, focus outlines, ARIA labels |

### Light / Dark / System toggle

Tailwind `dark:` variant + `class` strategy use karo, teen states support karo — **System default, Light, Dark**:

```javascript
// tailwind.config.js
module.exports = {
  darkMode: "class",
  // ...
};
```

```typescript
// hooks/useTheme.ts — system default detect + manual override
function applyTheme(theme: "system" | "light" | "dark") {
  const isDark =
    theme === "dark" ||
    (theme === "system" && window.matchMedia("(prefers-color-scheme: dark)").matches);
  document.documentElement.classList.toggle("dark", isDark);
  localStorage.setItem("theme", theme);
}
```

Ek toggle button (navbar mein) teeno options cycle kare: System → Light → Dark.

---

## Step 4 — Page / Route Mapping

| Page | Route | Main UI blocks |
|---|---|---|
| Home / Upload | `/` | Hero (animated gradient), drag-&-drop upload zone |
| Results / Dashboard | `/results` | Metric cards (Baud Rate, SNR, Modulation), waveform chart, classification table |
| About | `/about` | Project info, links to repo |

Next.js `app/` router use karo, common layout `components/Layout.tsx` mein rakho (navbar + theme toggle yahin se aayega).

---

## Step 5 — UI Logic (Upload → Process → Display)

1. **File upload** — `multipart/form-data` POST `/process_file` ko. Upload ke dauran spinner/skeleton dikhao.
2. **Polling** (agar backend async job return karta hai) — `/status/:id` ko poll karo jab tak `completed: true` na aaye.
3. **Display results**:
   - Metric cards → `baudRate`, `snr`, `modulation`
   - Waveform → returned time-series array ko `<canvas>` ya Recharts pe draw karo
   - Optional: decoded bits table / constellation diagram
4. **Error handling** — non-2xx responses catch karo, toast dikhao, retry allow karo.
5. **State management** — chhote app ke liye `useState`/`useEffect` ya SWR kaafi hai. Agar complexity badhe toh Zustand/Redux Toolkit.

---

## Step 6 — Connect Frontend ↔ Backend

- FastAPI mein CORS already Step 1.3 mein enable ho chuka hai — bas origin production mein update karna.
- Har endpoint verify karo ki wahi shape return kar raha hai jo Step 1.5 ke `api.md` mein documented hai.
- Extra fields chahiye (jaise preprocessing time) toh backend mein add karo — Streamlit app ko touch karne ki zaroorat nahi.

---

## Step 7 — Styling & Polish

- `npm run dev` se frontend chalao, component-by-component polish karo (colors, shadows, animations) jab tak Stellar-jaisa na lage.
- Responsive layout test karo — mobile pe upload zone vertically stack ho, metrics horizontal scroll ban jaye agar zaroorat pade.
- Theme toggle final test karo (system/light/dark teeno).

---

## Step 8 — Testing (light)

- **Manual**: kuch sample `.wav`/`.iq` files upload karo, confirm backend process karta hai aur UI sahi update hoti hai.
- **Automated (optional)**: Cypress ya Playwright se ek-do tests likho jo FastAPI responses mock karke UI updates assert karein.

---

## Step 9 — Deployment

**Frontend**
- Vercel (Next.js ke liye native) ya Netlify (static export ke liye).
- `NEXT_PUBLIC_API_URL` env variable set karo production FastAPI host pe pointing.

**Backend**
- Docker / Hugging Face Spaces / AWS Lambda — jo bhi convenient ho.
- CORS origins update karo naye frontend domain ke saath.

---

## TL;DR Checklist (order mein follow karo)

1. [ ] FastAPI backend setup karo (`api/` folder), existing toolkit functions wrap karo.
2. [ ] CORS enable karo.
3. [ ] API contract document karo (`api.md`).
4. [ ] Next.js (ya Astro) + Tailwind scaffold karo (`frontend/` folder).
5. [ ] Design tokens define karo (colors, fonts, spacing).
6. [ ] System/Light/Dark theme toggle implement karo.
7. [ ] Pages banao: Home (`/`), Results (`/results`), About (`/about`).
8. [ ] Upload → poll → display flow build karo.
9. [ ] Framer-motion animations + polish.
10. [ ] Manual testing karo.
11. [ ] Deploy: backend (Docker/HF Spaces) + frontend (Vercel/Netlify).

**Important**: `web_demo/app.py` (Streamlit) ko poori process ke dauran touch mat karna — wo independent internal tool ke roop mein chalta rahega.
