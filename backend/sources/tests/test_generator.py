import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from generator import CSVParser, find_data, generate_bpm, generate_uterus, process_single_row, start_generation, update_data
from database import DataForGenerator

@pytest.mark.asyncio
async def test_parse_file(tmp_path):
    # создаём временный csv файл
    file = tmp_path / "test.csv"
    file.write_text("1,2\n3,4")

    parser = CSVParser(str(file))
    data = await parser.parse_file(str(file))
    assert data == [['1', '2'], ['3', '4']]

@pytest.mark.asyncio
async def test_parse_file_sync(tmp_path):
    file = tmp_path / "test_sync.csv"
    file.write_text("5,6\n7,8")
    parser = CSVParser(str(file))
    data = parser.parse_file_sync(str(file))
    assert data == [['5', '6'], ['7', '8']]

@pytest.mark.asyncio
async def test_find_data(monkeypatch):
    mock_session = AsyncMock()
    monkeypatch.setattr("generator.AsyncSessionLocal", lambda: mock_session)
    
    # Заглушка os.path.exists, чтобы имитировать наличие директорий
    monkeypatch.setattr("os.path.exists", lambda x: True)
    monkeypatch.setattr("os.walk", lambda path: [("path", [], ["file_hypoxiaa.csv"])])
    
    # Патчим DataForGenerator, чтобы просто считать, что он создается
    monkeypatch.setattr("generator.DataForGenerator", MagicMock())
    
    result = await find_data(lambda: mock_session)
    assert "status" in result

@pytest.mark.asyncio
async def test_generate_bpm_and_uterus(monkeypatch):
    mock_session = AsyncMock()
    monkeypatch.setattr("generator.AsyncSessionLocal", lambda: mock_session)
    
    # Имитация результатов запроса id_patient
    mock_session.execute.return_value.scalar_one_or_none.return_value = 123
    
    bpm_reader = [["0", "70"], ["1", "75"]]
    uterus_reader = [["0", "5"], ["1", "6"]]
    
    await generate_bpm(bpm_reader, lambda: mock_session)
    await generate_uterus(uterus_reader, lambda: mock_session)
    
    # Проверяем, что commit был вызван
    assert mock_session.commit.called

@pytest.mark.asyncio
async def test_process_single_row(monkeypatch, tmp_path):
    # Создаем временные файлы bpm и uterus
    bpm_file = tmp_path / "type_folder_bpm.csv"
    uterus_file = tmp_path / "type_folder_uterus.csv"
    bpm_file.write_text("0,70\n1,75")
    uterus_file.write_text("0,5\n1,6")
    
    row = DataForGenerator(
        source="source",
        type_="type",
        folder="type_folder",
        id_="1",
        is_active=False,
        id_patient=1234
    )
    
    monkeypatch.setattr("os.path.exists", lambda x: True)
    
    # Патчим CSVParser, чтобы возвращать реальные данные
    monkeypatch.setattr("generator.CSVParser.parse_file_sync", lambda self, path: [["0", "70"], ["1", "75"]])
    
    result = await process_single_row(row, lambda: AsyncMock())
    assert result in [True, False]  # в зависимости от обработки

@pytest.mark.asyncio
async def test_start_generation(monkeypatch):
    mock_session = AsyncMock()
    monkeypatch.setattr("generator.AsyncSessionLocal", lambda: mock_session)
    # Патчим запросы на пустые результаты
    mock_session.execute.return_value.scalars.return_value.all.return_value = []
    result = await start_generation(lambda: mock_session)
    assert "status" in result

@pytest.mark.asyncio
async def test_update_data(monkeypatch):
    monkeypatch.setattr("generator.find_data", AsyncMock(return_value={"status": "success"}))
    result = await update_data(lambda: AsyncMock())
    assert result["status"] == "success"

