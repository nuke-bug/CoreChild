# ktg_data_generator.py
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
from datetime import datetime
import random
import math

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Начальные значения
time_counter = 0
base_bpm = 145
base_power = 25

@app.websocket("/ws/fetus")
async def websocket_fetus_endpoint(websocket: WebSocket):
    """WebSocket для данных плода"""
    global time_counter, base_bpm
    
    await websocket.accept()
    print("✅ Подключен клиент к WebSocket плода")
    
    try:
        while True:
            # Увеличиваем время
            time_counter += 5
            
            # Создаем плавные изменения BPM
            bpm_sine = math.sin(time_counter * 0.1) * 4
            bpm_noise = random.uniform(-2, 2)
            current_bpm = base_bpm + bpm_sine + bpm_noise
            
            # Акцелерации
            has_acceleration = random.random() < 0.15
            if has_acceleration:
                current_bpm = min(180, current_bpm + random.randint(10, 20))
                acceleration_status = True
            else:
                acceleration_status = False
            
            # Децелерации
            has_deceleration = random.random() < 0.1
            if has_deceleration:
                current_bpm = max(110, current_bpm - random.randint(8, 15))
                deceleration_status = True
            else:
                deceleration_status = False
            
            fetus_data = {
                "time": time_counter,
                "bpm": round(current_bpm, 1),
                "basal_rhythm": 140.0,
                "hrv": random.uniform(8, 16),
                "acceleration": acceleration_status,
                "deceleration": deceleration_status,
                "hypoxia": "normal",
                "basal_status": "normal",
                "hrv_status": "normal", 
                "decel_status": "normal" if not deceleration_status else "suspicious",
                "accel_status": "normal" if not acceleration_status else "suspicious",
                "hypoxia_15": random.randint(2, 10),
                "hypoxia_30": random.randint(5, 15),
                "hypoxia_60": random.randint(8, 20),
                "timestamp": datetime.now().isoformat(),
                "type": "fetus"
            }
            
            await websocket.send_text(json.dumps(fetus_data))
            print(f"📤 Плод: время {time_counter}с, ЧСС {round(current_bpm, 1)}")
            
            await asyncio.sleep(5)
            
    except Exception as e:
        print(f"❌ Ошибка WebSocket плода: {e}")
    finally:
        print("🔌 Отключен клиент от WebSocket плода")

@app.websocket("/ws/uterus")
async def websocket_uterus_endpoint(websocket: WebSocket):
    """WebSocket для данных матки"""
    global time_counter, base_power
    
    await websocket.accept()
    print("✅ Подключен клиент к WebSocket матки")
    
    try:
        while True:
            # Изменения Power - имитация схваток
            power_sine = math.sin(time_counter * 0.05) * 8
            power_noise = random.uniform(-3, 3)
            current_power = max(5, base_power + power_sine + power_noise)
            
            # Схватки
            has_contraction = random.random() < 0.2
            if has_contraction:
                current_power = min(80, current_power + random.randint(20, 40))
                contraction_status = True
            else:
                contraction_status = False
            
            uterus_data = {
                "time": time_counter,
                "power": round(current_power),
                "contraction": contraction_status,
                "timestamp": datetime.now().isoformat(),
                "type": "uterus"
            }
            
            await websocket.send_text(json.dumps(uterus_data))
            
            event_str = " | СХВАТКА" if contraction_status else ""
            print(f"📤 Матка: время {time_counter}с, Активность {round(current_power)}%{event_str}")
            
            await asyncio.sleep(5)
            
    except Exception as e:
        print(f"❌ Ошибка WebSocket матки: {e}")
    finally:
        print("🔌 Отключен клиент от WebSocket матки")

@app.get("/")
def read_root():
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
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=9009, log_level="info")