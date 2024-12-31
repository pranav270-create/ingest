import sys
from pathlib import Path

from fastapi import Depends
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

sys.path.append(str(Path(__file__).resolve().parents[1]))

from serving.etl import router as etl_router
from serving.middleware import base_middleware


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.middleware("http")(base_middleware)

app.include_router(etl_router, prefix="/api/ml")

@app.get("/api/ml/health")
async def health_check():
    return {"status": "healthy"}
