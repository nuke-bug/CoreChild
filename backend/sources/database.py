from sqlalchemy import Column, Integer, String, Float, Boolean
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
import time
import asyncio

# Базовый класс для ORM-моделей
Base = declarative_base()


class DataForGenerator(Base):
    __tablename__ = 'data_for_generator'

    source = Column(String(100), primary_key=True, nullable=False)
    type_ = Column(String(100), primary_key=True, nullable=False)
    folder = Column(String(100), primary_key=True, nullable=False)
    id_ = Column(String(100), primary_key=True, nullable=False)
    is_active = Column(Boolean, default=False)
    id_patient = Column(Float, primary_key=True, nullable=True)


class Interfaces(Base):
    __tablename__ = 'interfaces'

    name = Column(String(100), primary_key=True, nullable=False)
    is_active = Column(Boolean, default=False)
    id_patient = Column(Float, nullable=True)


class BpmProcessedData(Base):
    __tablename__ = 'bpm_processed_data'

    time = Column(Float, primary_key=True, nullable=False)
    bpm = Column(Integer, primary_key=True, nullable=False)

    displayed = Column(Boolean, default=False)
    basal_rhythm = Column(Integer, nullable=True)
    hrv = Column(Integer, nullable=False)
    acceleration = Column(Boolean, default=True)
    deceleration = Column(Boolean, default=True)
    hypoxia = Column(String(100), nullable=True)

    # Статусы по различным метрикам (normal, suspicious, pathological)
    basal_status = Column(String(100), nullable=True)
    hrv_status = Column(String(100), nullable=True)
    decel_status = Column(String(100), nullable=True)
    accel_status = Column(String(100), nullable=True)

    # Риски гипоксии на разных интервалах (может быть None)
    hypoxia_15 = Column(Integer, nullable=True, default=None)
    hypoxia_30 = Column(Integer, nullable=True, default=None)
    hypoxia_60 = Column(Integer, nullable=True, default=None)


class BpmArchiveProcessedData(Base):
    __tablename__ = 'bpm_archive_processed_data'

    id_patient = Column(Float, primary_key=True, nullable=False)
    time = Column(Float, primary_key=True, nullable=False)
    bpm = Column(Float, primary_key=True, nullable=False)


class UterusProcessedData(Base):
    __tablename__ = 'uterus_processed_data'

    time = Column(Float, primary_key=True, nullable=False)
    power = Column(Integer, primary_key=True, nullable=False)

    displayed = Column(Boolean, default=False)


class UterusArchiveProcessedData(Base):
    __tablename__ = 'uterus_archive_processed_data'

    id_patient = Column(Float, primary_key=True, nullable=False)
    time = Column(Float, primary_key=True, nullable=False)
    power = Column(Integer, nullable=False)


class DocData(Base):
    __tablename__ = 'doc_data'

    id_patient = Column(Float, primary_key=True, nullable=False)
    gestation_week = Column(Integer, nullable=False)
    gestation_days = Column(Integer, nullable=False)


# Глобальные переменные
async_engine = None
AsyncSessionLocal = None


async def init_db(db_url='postgresql+asyncpg://pass:pass@10.0.0.4:5432/CORE'):
    """Инициализация базы данных, создание таблиц и начальных данных"""
    global async_engine, AsyncSessionLocal

    async_engine = create_async_engine(
        db_url,
        echo=True,
        pool_pre_ping=True,
        pool_recycle=300
    )

    AsyncSessionLocal = async_sessionmaker(
        async_engine,
        class_=AsyncSession,
        expire_on_commit=False
    )

    # Пересоздание таблиц
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)

    # Добавление начального интерфейса
    async with AsyncSessionLocal() as session:
        data = [
            Interfaces(name='archive', is_active=False, id_patient=time.time())
        ]
        try:
            session.add_all(data)
            await session.commit()
            print("База данных успешно инициализирована")
        except Exception as e:
            await session.rollback()
            print(f"Ошибка при инициализации базы данных: {e}")

    return async_engine, AsyncSessionLocal


async def get_async_db():
    """Асинхронный генератор сессии для FastAPI зависимостей"""
    if AsyncSessionLocal is None:
        raise RuntimeError("База данных не инициализирована")

    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


async def close_db_connection():
    """Закрытие соединения с базой данных"""
    if async_engine:
        await async_engine.dispose()

