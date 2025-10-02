from fastapi import FastAPI, APIRouter, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select, update
import pandas as pd
import asyncio
import json
from datetime import datetime
import logging

from generator import start_generation, find_data, clean_processed_data_and_stop
from database import (
    BpmProcessedData, BpmArchiveProcessedData, Interfaces,
    UterusProcessedData, UterusArchiveProcessedData,
    init_db, get_async_db, close_db_connection,
    AsyncSessionLocal
)
from analysis_ctg import main


import logging
# Отключаем логи от SQLAlchemy
logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)

# Настройка логирования
logging.basicConfig(level=logging.WARNING)
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

@router.api_route("/stop", methods=["GET", "POST"])
async def generate_data_endpoint():
    """Stop генерации данных в фоне по API вызову"""
    asyncio.create_task(clean_processed_data_and_stop(AsyncSessionLocal))
    return {"status": "STOP"}

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


Bpm_Last_stop_time = 0
Uterus_Last_stop_time = 0

from fastapi import WebSocket, WebSocketDisconnect
import json
from datetime import datetime

@app.websocket("/ws/fetus")
async def websocket_fetus_endpoint(websocket: WebSocket):
    """WebSocket для данных плода (BPM с дополнительным анализом)"""
    global Bpm_Last_stop_time
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

                # Анализ CTG
                new_data = main(df_bpm, df_uterus)
                print(new_data)
                # print(df_uterus)

                # Очищаем от NaN
                new_data.dropna(subset=[
                    'time', 'bpm', 'basal_rhythm', 'hrv', 'acceleration', 'deceleration',
                    'hypoxia', 'basal_status', 'hrv_status', 'decel_status', 'accel_status',
                    'hypoxia_15', 'hypoxia_30', 'hypoxia_60'
                ], inplace=True)

                if new_data.empty:
                    print("❌ Нет данных для обновления после удаления NaN")
                    await asyncio.sleep(1)
                    continue

                for _, row in new_data.iterrows():
                    if Bpm_Last_stop_time < float(row['time']):
                        # Отправка клиенту
                        fetus_data = {
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

                        try:
                            if Bpm_Last_stop_time < float(row['time']):
                            
                                await websocket.send_text(json.dumps(fetus_data))
                                
                        except RuntimeError as e:
                            print(f"❌ Ошибка отправки WebSocket плода: {e}")
                            break  # выходим из while True

                        # Обновление БД
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

                        Bpm_Last_stop_time = float(row['time'])
                
                await session.commit()
                await asyncio.sleep(1)

    except WebSocketDisconnect:
        print("❌ Клиент WebSocket плода отключился")
    except Exception as e:
        print(f"❌ Неизвестная ошибка WebSocket плода: {e}")
    finally:
        try:
            await websocket.close()
            print("🔌 Отключен клиент от WebSocket плода")
        except RuntimeError:
            print("⚠️ WebSocket уже был закрыт, повторное закрытие не требуется")


@app.websocket("/ws/uterus")
async def websocket_uterus_endpoint(websocket: WebSocket):
    """WebSocket для данных матки"""
    global Uterus_Last_stop_time
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
                    if Uterus_Last_stop_time < float(row['time']):
                        uterus_data = {
                        "time": float(row['time']),
                        "power": row['power'],
                        "timestamp": datetime.now().isoformat(),
                        "type": "uterus"
                    }
                        Uterus_Last_stop_time = float(row['time'])
                        await websocket.send_text(json.dumps(uterus_data))

            await asyncio.sleep(1)

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

