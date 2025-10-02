from fastapi import FastAPI, APIRouter, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select, update
import pandas as pd
import asyncio
import json
from datetime import datetime
import logging

from generator import start_generation, find_data
from database import (
    BpmProcessedData, BpmArchiveProcessedData, Interfaces,
    UterusProcessedData, UterusArchiveProcessedData,
    init_db, get_async_db, close_db_connection,
    AsyncSessionLocal
)
from analysis_ctg import start_analysis

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

# Добавляем CORS middleware, разрешая все источники
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

router = APIRouter()

@router.api_route("/start", methods=["GET", "POST"])
async def generate_data_endpoint():
    """Запуск генерации данных в фоне по API вызову"""
    asyncio.create_task(start_generation(AsyncSessionLocal))
    return {"status": "Генерация запущена в фоне"}

app.include_router(router)

@app.on_event("startup")
async def startup_event():
    """Инициализация базы данных и загрузка данных при старте приложения"""
    global AsyncSessionLocal
    engine, AsyncSessionLocal = await init_db()
    print("Инициализация базы данных...")

    # Загружаем данные из CSV при старте
    await find_data(AsyncSessionLocal)

@app.on_event("shutdown")
async def shutdown_event():
    """Закрытие соединений при завершении приложения"""
    await close_db_connection()

@app.websocket("/ws/fetus")
async def websocket_fetus_endpoint(websocket: WebSocket):
    """WebSocket для данных плода (BPM с дополнительным анализом)"""
    await websocket.accept()
    print("✅ Подключен клиент к WebSocket плода")

    try:
        while True:
            async with AsyncSessionLocal() as session:
                # Получаем BPM данные
                result = await session.execute(select(BpmProcessedData.time, BpmProcessedData.bpm))
                metrics = result.all()

                if not metrics:
                    print("⚠️ Нет BPM данных в базе.")
                    df_bpm = pd.DataFrame(columns=["time_sec", "value"])
                else:
                    df_bpm = pd.DataFrame(metrics, columns=["time_sec", "value"])
                    df_bpm["time_sec"] = pd.to_numeric(df_bpm["time_sec"], errors="coerce")
                    df_bpm["value"] = pd.to_numeric(df_bpm["value"], errors="coerce")
                    df_bpm.dropna(subset=["time_sec", "value"], inplace=True)

                print(f"Найдено BPM метрик: {len(df_bpm)}")

                # Получаем данные матки
                result = await session.execute(select(UterusProcessedData.time, UterusProcessedData.power))
                metrics = result.all()

                if not metrics:
                    print("⚠️ Нет uterus данных в базе.")
                    df_uterus = pd.DataFrame(columns=["time_sec", "value"])
                else:
                    df_uterus = pd.DataFrame(metrics, columns=["time_sec", "value"])
                    df_uterus["time_sec"] = pd.to_numeric(df_uterus["time_sec"], errors="coerce")
                    df_uterus["value"] = pd.to_numeric(df_uterus["value"], errors="coerce")
                    df_uterus.dropna(subset=["time_sec", "value"], inplace=True)

                print(f"Найдено uterus метрик: {len(df_uterus)}")

                # Запускаем анализ данных CTG
                new_data = start_analysis(df_bpm, df_uterus)

                # Очищаем данные от NaN перед обновлением БД
                new_data.dropna(subset=[
                    'time', 'bpm', 'basal_rhythm', 'hrv', 'acceleration', 'deceleration',
                    'hypoxia', 'basal_status', 'hrv_status', 'decel_status', 'accel_status',
                    'hypoxia_15', 'hypoxia_30', 'hypoxia_60'
                ], inplace=True)

                if new_data.empty:
                    print("❌ Нет данных для обновления после удаления NaN")
                    return

                for _, row in new_data.iterrows():
                    # Обновляем запись в базе
                    await session.execute(update(BpmProcessedData).where(
                        BpmProcessedData.time == float(row['time']),
                        BpmProcessedData.bpm == int(row['bpm'])
                    ).values(
                        displayed=True,
                        basal_rhythm=int(row['basal_rhythm']),
                        hrv=int(row['hrv']),
                        acceleration=bool(row['acceleration']),
                        deceleration=bool(row['deceleration']),
                        hypoxia=str(row['hypoxia']),
                        basal_status=str(row['basal_status']),
                        hrv_status=str(row['hrv_status']),
                        decel_status=str(row['decel_status']),
                        accel_status=str(row['accel_status']),
                        hypoxia_15=int(row['hypoxia_15']),
                        hypoxia_30=int(row['hypoxia_30']),
                        hypoxia_60=int(row['hypoxia_60']),
                    ))

                    # Формируем и отправляем данные клиенту
                    fetus_data = {
                        "fetus": {
                            "time": float(row['time']),
                            "bpm": int(row['bpm']),
                            "basal_rhythm": int(row['basal_rhythm']),
                            "hrv": int(row['hrv']),
                            "acceleration": bool(row['acceleration']),
                            "deceleration": bool(row['deceleration']),
                            "hypoxia": str(row['hypoxia']),
                            "basal_status": str(row['basal_status']),
                            "hrv_status": str(row['hrv_status']),
                            "decel_status": str(row['decel_status']),
                            "accel_status": str(row['accel_status']),
                            "hypoxia_15": int(row['hypoxia_15']),
                            "hypoxia_30": int(row['hypoxia_30']),
                            "hypoxia_60": int(row['hypoxia_60']),
                            "timestamp": datetime.now().isoformat(),
                            "type": "fetus"
                        }
                    }
                    await websocket.send_text(json.dumps(fetus_data))

                await session.commit()
                await asyncio.sleep(5)

    except Exception as e:
        print(f"❌ Ошибка WebSocket плода: {e}")
    finally:
        await websocket.close()
        print("🔌 Отключен клиент от WebSocket плода")

@app.websocket("/ws/uterus")
async def websocket_uterus_endpoint(websocket: WebSocket):
    """WebSocket для данных матки"""
    await websocket.accept()
    print("✅ Подключен клиент к WebSocket матки")

    try:
        while True:
            async with AsyncSessionLocal() as session:
                result = await session.execute(select(UterusProcessedData.time, UterusProcessedData.power))
                metrics = result.all()

                df_uterus = pd.DataFrame(metrics, columns=["time", "power"])

                print(f"Найдено uterus метрик: {len(metrics)}")

                for _, row in df_uterus.iterrows():
                    uterus_data = {
                        "time": row['time'],
                        "power": row['power'],
                        "timestamp": datetime.now().isoformat(),
                        "type": "uterus"
                    }
                    await websocket.send_text(json.dumps(uterus_data))

            await asyncio.sleep(5)

    except Exception as e:
        print(f"❌ Ошибка WebSocket матки: {e}")
    finally:
        await websocket.close()
        print("🔌 Отключен клиент от WebSocket матки")

@app.get("/")
def read_root():
    """Корневой endpoint с информацией о сервере и эндпоинтах"""
    return {
        "message": "КТГ сервер запущен",
        "endpoints": {
            "fetus_websocket": "/ws/fetus",
            "uterus_websocket": "/ws/uterus",
            "docs": "/docs"
        }
    }

@app.get("/health")
def health_check():
    """Проверка статуса сервера"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

