from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Signal Analysis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Update for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

# Import and include routers
from api.routes import process, classify, decode

app.include_router(process.router, prefix="/process")
app.include_router(classify.router, prefix="/classify")
app.include_router(decode.router, prefix="/decode")