from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, Boolean, Text
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid
from datetime import datetime

# Создаем базовый класс для моделей
Base = declarative_base()


class BpmProcessedData(Base):
    __tablename__ = 'bpm_processed_data'  # таблица с рабочими данными, очищается после каждого пациента
    
    time = Column(Float, primary_key=True, nullable=False) # время
    bpm = Column(Integer), nullable=False) # ЧСС плода (здесь округленное до int число, но в real_data без округления, поэтому там как float)
    basal_rhythm = Column(Integer), nullable=False) # базальный ритм плода
    hrv = Column(Integer), nullable=False) # вариабельность сердечного ритма
    acceleration = Column(Boolean, default=False) # есть или нет акселерация
    deceleration = Column(Boolean, default=False) # есть или нет децелерация
    hypoxia = Column(Boolean, default=False) # есть или нет гипоксия сейчас
    hypoxia_risk = Column(Integer), nullable=True) # вероятность развития гипоксии через 15 минут
    
    
class BpmArchiveProcessedData(Base):
    __tablename__ = 'bpm_archive_processed_data'  # таблица с архивными данными всех пациентов
    
    id_patient = Column(Integer), nullable=False, primary_key=True) # id пациента, выводится при обработке данных на фронт
    time = Column(Float) # время
    bpm = Column(Float), nullable=False) # ЧСС плода (здесь округленное до int число, но в real_data без округления, поэтому там как float)
    basal_rhythm = Column(Integer), nullable=False) # базальный ритм плода
    hrv = Column(Integer), nullable=False) # вариабельность сердечного ритма
    acceleration = Column(Boolean, default=False) # есть или нет акселерация
    deceleration = Column(Boolean, default=False) # есть или нет децелерация
    hypoxia = Column(Boolean, default=False) # есть или нет гипоксия сейчас
    hypoxia_risk = Column(Integer), nullable=True) # вероятность развития гипоксии через 15 минут
    
    
class UterusProcessedData(Base):
    __tablename__ = 'uterus_processed_data'  # таблица с рабочими данными, очищается после каждого пациента
    
    time = Column(Float, primary_key=True) # время
    power = Column(Integer), nullable=False) # сила сокращения
    contraction = Column(Boolean, default=False) # есть или нет сокращение
    
       
class UterusArchiveProcessedData(Base):
    __tablename__ = 'uterus_archive_processed_data'  # таблица с архивными данными всех пациентов

    id_patient = Column(Integer), nullable=False, primary_key=True) # id пациента, выводится при обработке данных на фронт
    time = Column(Float) # время
    power = Column(Integer), nullable=False) # сила сокращения
    contraction = Column(Boolean, default=False) # есть или нет сокращение
    

class DocData(Base):
    __tablename__ = 'doc_data'   # доп данные обезличенные для анализа важные, таблица с архивными данными всех пациентов

    id_patient = Column(Integer), nullable=False, primary_key=True) # id пациента, выводится при обработке данных на фронт
    gestation_week = Column(Integer), nullable=False) # срок беременности, недели
    gestation_days = Column(Integer), nullable=False) # срок беременности, дни
    

def init_db(db_url='postgresql://pass:pass@10.0.0.4:5432/CORE'):
    engine = create_engine(db_url)
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return engine, SessionLocal

