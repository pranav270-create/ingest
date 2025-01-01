import sys
from pathlib import Path

from fastapi import Depends
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

sys.path.append(str(Path(__file__).resolve().parents[1]))

from serving.routers.cron import router as cron_router
from serving.routers.qdrant_flow import router as qdrant_router
from serving.routers.sql_flow import router as sql_router
from serving.routers.user import router as user_router
from serving.middleware import base_middleware


app = FastAPI()
origins = [
    "http://localhost:5173",
    "http://localhost:8080",
    "https://dev.astralis.sh"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.middleware("http")(base_middleware)

app.include_router(cron_router, prefix="/api/ml")
app.include_router(qdrant_router, prefix="/api/ml")
app.include_router(sql_router, prefix="/api/ml")
app.include_router(user_router, prefix="/api/ml")

@app.get("/api/ml/health")
async def health_check():
    return {"status": "healthy"}
