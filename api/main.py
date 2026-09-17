import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Signal Analysis API",
    description="REST API for RF Signal Parameter Extraction, Modulation Classification & Waveform Decoding",
    version="0.2.0"
)

cors_origins_env = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")
if cors_origins_env.strip() == "*":
    origins = ["*"]
else:
    origins = [orig.strip() for orig in cors_origins_env.split(",") if orig.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

# Import and include routers
from api.routes import process, classify, decode, stream

app.include_router(process.router, prefix="/process")
app.include_router(classify.router, prefix="/classify")
app.include_router(decode.router, prefix="/decode")
app.include_router(stream.router, prefix="/stream")