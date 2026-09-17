# Deployment Guide — SIH Signal Analysis Suite

This document covers running and deploying the **SIH PS26147 Signal Processing Backend & Web Dashboard** across local and production environments.

---

## 1. Local Development (Without Docker)

### Backend (FastAPI)
```bash
# 1. Activate virtual environment
.venv\Scripts\activate       # Windows PowerShell / CMD
# or: source .venv/bin/activate  # Linux / macOS

# 2. Start the API server
python -m uvicorn api.main:app --reload --port 8000
```
- API Base: `http://localhost:8000`
- Swagger UI Documentation: `http://localhost:8000/docs`
- Health Probe: `http://localhost:8000/health`

### Frontend (Next.js)
```bash
# 1. Enter frontend folder
cd frontend

# 2. Install dependencies (first time)
npm install

# 3. Start development server
npm run dev
```
- Frontend UI: `http://localhost:3000`

---

## 2. Full-Stack Local Deployment (Docker Compose)

Both services are orchestrated via Docker Compose:

```bash
# Build and run all containers in the background
docker compose up --build -d

# Check running status & health
docker compose ps

# View live logs
docker compose logs -f
```

### Services Started:
- **Backend**: Container `sih-signal-api` on `http://localhost:8000`
- **Frontend**: Container `sih-signal-frontend` on `http://localhost:3000`

To stop:
```bash
docker compose down
```

---

## 3. Production Cloud Deployment

### Option A: Frontend on Vercel + Backend on Render / Railway

#### 1. Backend (FastAPI on Render / Railway / Hugging Face Spaces):
1. Connect the GitHub repository.
2. Configure build & start settings:
   - **Root Directory**: `.` (Repository root)
   - **Build Command**: `pip install -r api/requirements.txt && pip install -e .`
   - **Start Command**: `uvicorn api.main:app --host 0.0.0.0 --port 8000`
   - **Environment Variables**:
     - `CORS_ORIGINS`: `https://your-frontend-domain.vercel.app`
     - `PORT`: `8000`

#### 2. Frontend (Next.js on Vercel):
1. Import repository on [Vercel](https://vercel.com).
2. Set **Root Directory** to `frontend`.
3. Add Environment Variable:
   - `NEXT_PUBLIC_API_URL`: `https://your-backend-service.onrender.com` (your deployed FastAPI URL without trailing slash).
4. Click **Deploy**.

---

### Option B: Self-Hosted Cloud VM / AWS EC2 / DigitalOcean

```bash
# Clone the repository on your server
git clone https://github.com/mks-Roald/ps26147_toolkit.git
cd ps26147_toolkit

# Create .env file with your public domain
echo "CORS_ORIGINS=https://yourdomain.com" > .env
echo "NEXT_PUBLIC_API_URL=https://api.yourdomain.com" >> .env

# Launch with Docker Compose
docker compose up --build -d
```

---

## 4. Environment Variables Reference

| Variable | Service | Description | Default |
|---|---|---|---|
| `PORT` | Backend | Port for Uvicorn server | `8000` |
| `CORS_ORIGINS` | Backend | Comma-separated list of allowed origins or `*` | `http://localhost:3000,http://127.0.0.1:3000` |
| `NEXT_PUBLIC_API_URL` | Frontend | URL of the backend REST API | `http://localhost:8000` |
| `NODE_ENV` | Frontend | Node runtime environment | `production` |
