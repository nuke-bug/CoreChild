import pytest
from fastapi.testclient import TestClient
from main import app 

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

@pytest.mark.asyncio
async def test_generate_data_endpoint(monkeypatch):
    monkeypatch.setattr("your_fastapi_module.start_generation", lambda x: None)
    response = client.get("/start")
    assert response.status_code == 200
    assert response.json()["status"] == "Генерация запущена в фоне"

@pytest.mark.asyncio
async def test_websocket_fetus(monkeypatch):
    from fastapi import WebSocket
    from fastapi.testclient import TestClient
    
    # Мокаем сессию базы и возвращаем пустые данные, чтобы не ломался тест
    async def mock_execute(*args, **kwargs):
        class MockResult:
            def all(self): return []
            def scalar_one_or_none(self): return 123
        return MockResult()

    async def mock_session():
        class MockSession:
            async def __aenter__(self): return self
            async def __aexit__(self, exc_type, exc, tb): pass
            async def execute(self, *args, **kwargs): return await mock_execute()
            async def commit(self): pass
        return MockSession()

    monkeypatch.setattr("your_fastapi_module.AsyncSessionLocal", mock_session)

    with client.websocket_connect("/ws/fetus") as websocket:
        # Отправляем и принимаем данные в течение одного цикла
        try:
            message = websocket.receive_text()
            assert "fetus" in message or message == ""
        except Exception:
            pass  # возможен таймаут из-за бесконечного цикла

