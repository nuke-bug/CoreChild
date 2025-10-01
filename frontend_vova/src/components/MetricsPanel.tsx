// src/components/panels/MetricsPanel.tsx
import React from 'react';
import {type FetusData, type StatusType } from '../types/index';

interface MetricsPanelProps {
  latestFetusData: FetusData | null;
}

/**
 * Компонент для отображения текущих метрик и статусов
 */
export const MetricsPanel: React.FC<MetricsPanelProps> = ({ latestFetusData }) => {
  if (!latestFetusData) {
    return (
      <div className="metrics-panel">
        <h3>Метрики плода</h3>
        <div className="no-data">Нет данных</div>
      </div>
    );
  }

  const {
    bpm,
    basal_rhythm,
    hrv,
    acceleration,
    deceleration,
    hypoxia,
    basal_status,
    hrv_status,
    decel_status,
    accel_status,
    hypoxia_15,
    hypoxia_30,
    hypoxia_60
  } = latestFetusData;

  /**
   * Функция для получения CSS класса по статусу
   */
  const getStatusClass = (status: StatusType): string => {
    switch (status) {
      case 'normal': return 'status-normal';
      case 'suspicious': return 'status-warning';
      case 'pathological': return 'status-critical';
      default: return '';
    }
  };

  /**
   * Функция для отображения иконки статуса
   */
  const getStatusIcon = (status: StatusType): string => {
    switch (status) {
      case 'normal': return '✅';
      case 'suspicious': return '⚠️';
      case 'pathological': return '❌';
      default: return '';
    }
  };

  return (
    <div className="metrics-panel">
      <h3>Метрики плода</h3>
      
      <div className="metrics-grid">
        {/* Основные показатели */}
        <div className="metric-card">
          <span className="metric-label">Текущая ЧСС</span>
          <span className="metric-value">{bpm} уд/мин</span>
        </div>

        <div className="metric-card">
          <span className="metric-label">Базальный ритм</span>
          <span className="metric-value">{basal_rhythm} уд/мин</span>
          <span className={`status-badge ${getStatusClass(basal_status)}`}>
            {getStatusIcon(basal_status)} {basal_status}
          </span>
        </div>

        <div className="metric-card">
          <span className="metric-label">Вариабельность (HRV)</span>
          <span className="metric-value">{hrv} мс</span>
          <span className={`status-badge ${getStatusClass(hrv_status)}`}>
            {getStatusIcon(hrv_status)} {hrv_status}
          </span>
        </div>

        {/* События */}
        <div className="metric-card">
          <span className="metric-label">Акцелерация</span>
          <span className={`event-indicator ${acceleration ? 'active' : 'inactive'}`}>
            {acceleration ? '✅ Есть' : '◯ Нет'}
          </span>
          <span className={`status-badge ${getStatusClass(accel_status)}`}>
            {getStatusIcon(accel_status)} {accel_status}
          </span>
        </div>

        <div className="metric-card">
          <span className="metric-label">Децелерация</span>
          <span className={`event-indicator ${deceleration ? 'active' : 'inactive'}`}>
            {deceleration ? '⚠️ Есть' : '◯ Нет'}
          </span>
          <span className={`status-badge ${getStatusClass(decel_status)}`}>
            {getStatusIcon(decel_status)} {decel_status}
          </span>
        </div>

        {/* Статус гипоксии */}
        <div className="metric-card">
          <span className="metric-label">Статус гипоксии</span>
          <span className={`status-badge ${getStatusClass(hypoxia)}`}>
            {getStatusIcon(hypoxia)} {hypoxia}
          </span>
        </div>

        {/* Прогноз гипоксии */}
        <div className="metric-card wide">
          <span className="metric-label">Риск гипоксии</span>
          <div className="hypoxia-prediction">
            <div className="prediction-item">
              <span>15 мин:</span>
              <span>{hypoxia_15 !== null ? `${hypoxia_15}%` : '—'}</span>
            </div>
            <div className="prediction-item">
              <span>30 мин:</span>
              <span>{hypoxia_30 !== null ? `${hypoxia_30}%` : '—'}</span>
            </div>
            <div className="prediction-item">
              <span>60 мин:</span>
              <span>{hypoxia_60 !== null ? `${hypoxia_60}%` : '—'}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};