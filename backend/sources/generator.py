import csv
import os
import asyncio
import aiofiles
import time
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Dict, Any

from database import (
    DataForGenerator, Interfaces, BpmProcessedData, 
    BpmArchiveProcessedData, UterusProcessedData, UterusArchiveProcessedData
)

class CSVParser:
    def __init__(self, file_path=None):
        self.file_path = file_path

    async def parse_file(self, filename: str, delimiter: str = ',') -> Any:
        try:
            print(f"Парсим файл: {filename}")
            
            if not os.path.exists(filename):
                print(f"Файл не существует: {filename}")
                return None

            async with aiofiles.open(filename, 'r', encoding='utf-8') as f:
                content = await f.read()

            reader = csv.reader(content.splitlines(), delimiter=delimiter)
            data = list(reader)
            print(f"Успешно распаршено {len(data)} строк из {filename}")
            return data

        except Exception as e:
            print(f"Ошибка при парсинге {filename}: {e}")
            return None

    def parse_file_sync(self, filename: str, delimiter: str = ',') -> Any:
        """Синхронный метод парсинга CSV-файла (для использования с asyncio.to_thread)"""
        try:
            print(f"Парсим файл синхронно: {filename}")
            
            if not os.path.exists(filename):
                print(f"Файл не существует: {filename}")
                return None

            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()

            reader = csv.reader(content.splitlines(), delimiter=delimiter)
            data = list(reader)
            print(f"Успешно распаршено {len(data)} строк из {filename}")
            return data

        except Exception as e:
            print(f"Ошибка при парсинге {filename}: {e}")
            return None


async def find_data(AsyncSessionLocal):
    async with AsyncSessionLocal() as session:
        try:
            print("=== НАЧАЛО ЗАГРУЗКИ ДАННЫХ ИЗ CSV ===")
            data = []

            for data_type in ['hypoxiaa', 'regularr']:
                type_path = f"./{data_type}"

                if not os.path.exists(type_path):
                    print(f"Директория {type_path} не существует!")
                    continue

                for root, dirs, files in os.walk(type_path):
                    for file in files:
                        if file.endswith('.csv'):
                            full_path = os.path.join(root, file)
                            rel_path = os.path.relpath(full_path, '.')
                            path_parts = rel_path.split(os.sep)

                            if len(path_parts) >= 1:
                                source = path_parts[-1].split('_')[-1].split('.')[0]
                                type_ = path_parts[0]
                                folder = path_parts[-1].split('_')[0]
                                file_id = '0'

                                data_record = DataForGenerator(
                                    source=source,
                                    type_=type_,
                                    folder=folder,
                                    id_=file_id,
                                    is_active=False,
                                    id_patient=float(time.time()),
                                )
                                data.append(data_record)

            print(f"Всего найдено записей: {len(data)}")

            if data:
                session.add_all(data)
                await session.commit()
                print("Данные успешно сохранены в БД")
                return {"status": "success", "records": len(data)}
            else:
                print("Не найдено данных для сохранения")
                return {"status": "no_data", "records": 0}

        except Exception as e:
            print(f"Ошибка в find_data: {e}")
            await session.rollback()
            return {"status": "error", "message": str(e)}


async def generate_bpm(reader: List[Dict], AsyncSessionLocal):
    async with AsyncSessionLocal() as session:
        try:
            print("Начинаем генерацию BPM данных...")

            result = await session.execute(
                select(Interfaces.id_patient).where(Interfaces.name == 'archive')
            )
            id_patient_result = result.scalar_one_or_none()
            id_patient = id_patient_result if id_patient_result else float(time.time())

            print(f"ID пациента для BPM: {id_patient}")

            start_time = time.time()
            processed_count = 0

            for row in reader:
                try:
                    time_sec = float(row[0])
                    value = float(row[1])

                    target_absolute_time = start_time + time_sec
                    wait_time = target_absolute_time - time.time()

                    if wait_time > 0:
                        await asyncio.sleep(wait_time)

                    bpm_data = BpmProcessedData(
                        time=time_sec,
                        bpm=int(value),
                        basal_rhythm=int(value),
                        hrv=0,
                        acceleration=False,
                        deceleration=False,
                        hypoxia="false",
                        basal_status="",
                        hrv_status="",
                        decel_status="",
                        accel_status="",
                        hypoxia_15=0,
                        hypoxia_30=0,
                        hypoxia_60=0,
                        displayed=False
                    )
                    bpm_archive_data = BpmArchiveProcessedData(
                        time=time_sec,
                        bpm=int(value),
                        id_patient=id_patient
                    )

                    session.add(bpm_data)
                    session.add(bpm_archive_data)
                    processed_count += 1

                    if processed_count % 5 == 0:
                        await session.commit()
                        print(f"Коммит {processed_count} записей BPM")

                except Exception as row_error:
                    print(f"Ошибка обработки строки BPM: {row_error}")
                    continue

            await session.commit()
            print(f"Завершена генерация BPM данных. Обработано записей: {processed_count}")

        except Exception as e:
            print(f"Ошибка в generate_bpm: {e}")
            await session.rollback()
            raise


async def generate_uterus(reader: List[Dict], AsyncSessionLocal):
    async with AsyncSessionLocal() as session:
        try:
            print("Начинаем генерацию uterus данных...")

            result = await session.execute(
                select(Interfaces.id_patient).where(Interfaces.name == 'archive')
            )
            id_patient_result = result.scalar_one_or_none()
            id_patient = id_patient_result if id_patient_result else float(time.time())

            print(f"ID пациента для uterus: {id_patient}")

            start_time = time.time()
            processed_count = 0

            for row in reader:
                try:
                    time_sec = float(row[0])
                    value = float(row[1])

                    target_absolute_time = start_time + time_sec
                    wait_time = target_absolute_time - time.time()

                    if wait_time > 0:
                        await asyncio.sleep(wait_time)

                    uterus_data = UterusProcessedData(
                        time=time_sec,
                        power=int(value)
                    )
                    uterus_archive_data = UterusArchiveProcessedData(
                        time=time_sec,
                        power=int(value),
                        id_patient=id_patient
                    )

                    session.add(uterus_data)
                    session.add(uterus_archive_data)
                    processed_count += 1

                    if processed_count % 5 == 0:
                        await session.commit()
                        print(f"Коммит {processed_count} записей uterus")

                except Exception as row_error:
                    print(f"Ошибка обработки строки uterus: {row_error}")
                    continue

            await session.commit()
            print(f"Завершена генерация uterus данных. Обработано записей: {processed_count}")

        except Exception as e:
            print(f"Ошибка в generate_uterus: {e}")
            await session.rollback()
            raise


async def process_single_row(row: DataForGenerator, AsyncSessionLocal) -> bool:
    try:
        print(f"Обрабатываем запись: ID={row.id_}, Type={row.type_}, Folder={row.folder}")

        bpm_path = f'./{row.type_}/{row.folder}_bpm.csv'
        uterus_path = f'./{row.type_}/{row.folder}_uterus.csv'

        bpm_exists = await asyncio.to_thread(os.path.exists, bpm_path)
        uterus_exists = await asyncio.to_thread(os.path.exists, uterus_path)

        if not bpm_exists:
            print(f"Файл BPM не существует: {bpm_path}")
            return False

        if not uterus_exists:
            print(f"Файл uterus не существует: {uterus_path}")
            return False

        # Парсим файлы в отдельных потоках (синхронно)
        csv_file_bpm = await asyncio.to_thread(CSVParser, bpm_path)
        csv_file_uterus = await asyncio.to_thread(CSVParser, uterus_path)

        reader_bpm = await asyncio.to_thread(csv_file_bpm.parse_file_sync, csv_file_bpm.file_path)
        reader_uterus = await asyncio.to_thread(csv_file_uterus.parse_file_sync, csv_file_uterus.file_path)

        if not reader_bpm or not reader_uterus:
            print(f"Не удалось прочитать файлы для {row.id_}")
            return False

        print(f"Успешно прочитаны оба файла. BPM: {len(reader_bpm)} строк, uterus: {len(reader_uterus)} строк")

        bpm_task = asyncio.create_task(generate_bpm(reader_bpm, AsyncSessionLocal))
        uterus_task = asyncio.create_task(generate_uterus(reader_uterus, AsyncSessionLocal))

        await asyncio.gather(bpm_task, uterus_task, return_exceptions=True)

        async with AsyncSessionLocal() as session:
            update_stmt = update(DataForGenerator).where(
                DataForGenerator.source == row.source,
                DataForGenerator.type_ == row.type_,
                DataForGenerator.folder == row.folder,
                DataForGenerator.id_patient == row.id_patient
            ).values(is_active=True)

            await session.execute(update_stmt)
            await session.commit()

        print(f"Запись {row.id_} успешно обработана")
        return True

    except Exception as e:
        print(f"Ошибка при обработке записи {row.id_}: {e}")
        return False


async def start_generation(AsyncSessionLocal):
    async with AsyncSessionLocal() as session:
        try:
            print("=== НАЧАЛО ГЕНЕРАЦИИ ДАННЫХ ===")

            result = await session.execute(
                select(Interfaces).where(Interfaces.is_active == False)
            )
            rows = result.scalars().all()
            print(f"Найдено Interfaces: {len(rows)}")

            if not rows:
                return {"status": "no_data", "message": "Нет Interfaces для обработки"}

            # Активируем "archive"
            await session.execute(update(Interfaces).where(
                Interfaces.name == 'archive',
                Interfaces.is_active == False
            ).values(is_active=True))

            # Выбираем неактивные данные
            result = await session.execute(
                select(DataForGenerator).where(DataForGenerator.is_active == False)
            )
            rows = result.scalars().all()
            print(f"Найдено записей для обработки: {len(rows)}")

            if not rows:
                return {"status": "no_data", "message": "Нет данных для обработки"}

            processed_count = 0
            for row in rows:
                success = await process_single_row(row, AsyncSessionLocal)
                if success:
                    processed_count += 1
                break  # FIXME: обрабатывается только одна запись

            print(f"=== ЗАВЕРШЕНИЕ ГЕНЕРАЦИИ ===")
            print(f"Обработано записей: {processed_count}")
            return {"status": "success", "processed": processed_count}

        except Exception as e:
            print(f"Критическая ошибка в start_generation: {e}")
            await session.rollback()
            return {"status": "error", "message": str(e)}


async def update_data(AsyncSessionLocal):
    print("Запуск обновления данных...")
    try:
        result = await find_data(AsyncSessionLocal)
        print("Обновление данных завершено")
        return result
    except Exception as e:
        print(f"Ошибка в update_data: {e}")
        return {"status": "error", "message": str(e)}

