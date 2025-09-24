from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uuid
from datetime import datetime
import os
import logging

# Настройка логирования
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Simple API", 
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc", 
    openapi_url="/api/openapi.json"
)

# CORS настройки
origins = [
    "http://localhost",
    "https://localhost",
    "http://localhost:3000",
    "https://localhost:3000",
    "http://10.0.0.2",
    "https://10.0.0.2",    
    "https://frontend",
    "http://frontend",
]

# Добавляем CORS из переменных окружения
cors_origins = os.getenv("CORS_ORIGINS", "").split(",")
if cors_origins and cors_origins[0]:
    origins.extend(cors_origins)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Trusted hosts
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["localhost", "frontend", "backend", "127.0.0.1", "10.0.0.2"]
)

from fastapi import APIRouter, HTTPException
from sqlalchemy import select, update
from database import init_db, BpmProcessedData, BpmArchiveProcessedData, Interfaces, UterusProcessedData, UterusArchiveProcessedData, DocData
import time

engine, SessionLocal = init_db()
db = SessionLocal()

router = APIRouter()

@router.get("/interfaces")
async def get_interfaces():    
    stmt = update(Interfaces).where(Interfaces.is_active == False).values(
                id_patient=time.time())
    result = db.execute(stmt)
    db.commit()
    
    result = db.execute(select(Interfaces))
    interfaces = result.scalars().all()
    return interfaces

@router.get("/{interface}/run")
async def run_interface(interface: str):
    # todo добавить запуск считывания с интерфейса
    return {"status": f"Интерфейс {interface} запущен"}

@router.get("/{interface}/bpm")
async def get_bpm_metrics(interface: str):
    result = db.execute(select(BpmProcessedData))
    metrics = result.scalars().all()
    if not metrics:
        raise HTTPException(status_code=404, detail="BPM metrics not found")
    return metrics

@router.get("/{interface}/uterus")
async def get_uterus_metrics(interface: str):
    result = db.execute(select(UterusProcessedData))
    metrics = result.scalars().all()
    if not metrics:
        raise HTTPException(status_code=404, detail="Uterus metrics not found")
    return metrics

@router.get("/")
async def get_home_page():
    return {
        "message": "Medical Monitor API",
        "endpoints": {
            "interfaces": "/api/interfaces",
            "run_interface": "/api/{interface}/run", 
            "bpm_metrics": "/api/{interface}/bpm",
            "uterus_metrics": "/api/{interface}/uterus"
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "backend"}

app.include_router(router, prefix="/api")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
