# simple_ktg_server.py
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
from datetime import datetime
import random
from fastapi.responses import HTMLResponse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Начальные значения
time_counter = 0
base_bpm = 145

@app.websocket("/ws/ktg")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket с изменяющимся временем и небольшими колебаниями данных"""
    global time_counter, base_bpm
    
    await websocket.accept()
    print("✅ Клиент подключен")
    
    try:
        while True:
            # Увеличиваем время
            time_counter += 5
            
            # Небольшие случайные колебания ЧСС
            bpm_variation = random.randint(-3, 3)
            current_bpm = base_bpm + bpm_variation
            
            # Случайные изменения активности матки
            power_variation = random.randint(-5, 5)
            current_power = max(0, 25 + power_variation)
            
            data_to_send = {
                "fetus": {
                    "time": time_counter,
                    "bpm": current_bpm,  # Меняющаяся ЧСС
                    "basal_rhythm": 140.0,
                    "hrv": 12.5,
                    "acceleration": random.choice([True, False]),  # Случайные акцелерации
                    "deceleration": random.choice([True, False]),  # Случайные децелерации
                    "hypoxia": "normal",
                    "basal_status": "normal",
                    "hrv_status": "normal", 
                    "decel_status": "normal",
                    "accel_status": "normal",
                    "hypoxia_15": random.randint(3, 8),
                    "hypoxia_30": random.randint(6, 12),
                    "hypoxia_60": random.randint(10, 18)
                },
                "uterus": {
                    "time": time_counter,
                    "power": current_power,  # Меняющаяся активность
                    "contraction": random.choice([True, False])  # Случайные схватки
                },
                "timestamp": datetime.now().isoformat(),
                "message": f"Время: {time_counter}с, ЧСС: {current_bpm}, Активность: {current_power}%"
            }
            
            await websocket.send_text(json.dumps(data_to_send))
            print(f"📤 Время: {time_counter}с, ЧСС: {current_bpm}, Активность: {current_power}%")
            
            await asyncio.sleep(5)
            
    except Exception as e:
        print(f"❌ Ошибка: {e}")
    finally:
        print("🔌 Клиент отключен")

@app.get("/test", response_class=HTMLResponse)
async def test_page():
    """Простая HTML страница для тестирования WebSocket соединения"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>КТГ WebSocket Тест</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 800px; margin: 0 auto; }
            .status { padding: 10px; margin: 10px 0; border-radius: 5px; }
            .connected { background: #d4edda; color: #155724; }
            .disconnected { background: #f8d7da; color: #721c24; }
            .data-panel { background: #f8f9fa; padding: 15px; border-radius: 5px; margin: 10px 0; }
            pre { background: #2d2d2d; color: #f8f8f2; padding: 15px; border-radius: 5px; overflow-x: auto; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>КТГ WebSocket Тест</h1>
            <div id="status" class="status disconnected">Статус: Отключено</div>
            <button onclick="connect()">Подключиться</button>
            <button onclick="disconnect()">Отключиться</button>
            
            <div class="data-panel">
                <h3>Данные КТГ:</h3>
                <pre id="data">Ожидание данных...</pre>
            </div>
            
            <div class="data-panel">
                <h3>Статистика:</h3>
                <div id="stats">Нет данных</div>
            </div>
        </div>

        <script>
            let ws = null;
            let messageCount = 0;
            
            function connect() {
                ws = new WebSocket('ws://localhost:9009/ws/ktg');
                
                ws.onopen = function(event) {
                    document.getElementById('status').className = 'status connected';
                    document.getElementById('status').textContent = 'Статус: Подключено';
                    messageCount = 0;
                    updateStats();
                };
                
                ws.onmessage = function(event) {
                    messageCount++;
                    const data = JSON.parse(event.data);
                    document.getElementById('data').textContent = JSON.stringify(data, null, 2);
                    updateStats();
                };
                
                ws.onclose = function(event) {
                    document.getElementById('status').className = 'status disconnected';
                    document.getElementById('status').textContent = 'Статус: Отключено';
                };
                
                ws.onerror = function(error) {
                    console.error('WebSocket error:', error);
                };
            }
            
            function disconnect() {
                if (ws) {
                    ws.close();
                    ws = null;
                }
            }
            
            function updateStats() {
                document.getElementById('stats').innerHTML = `
                    <p>Получено сообщений: ${messageCount}</p>
                    <p>Время: ${new Date().toLocaleTimeString()}</p>
                `;
            }
            
            // Автоподключение при загрузке страницы
            window.onload = connect;
        </script>
    </body>
    </html>
    """

@app.get("/")
def read_root():
    return {"message": "КТГ сервер запущен", "endpoint": "/ws/ktg"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=9009, log_level="info")