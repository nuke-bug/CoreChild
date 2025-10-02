// src/components/panels/MetricsPanel.tsx
import React from 'react';
import { type FetusData } from '../types/index';

interface MetricsPanelProps {
  latestFetusData: FetusData | null;
}

export const MetricsPanel: React.FC<MetricsPanelProps> = ({ latestFetusData }) => {
  if (!latestFetusData) {
    return (
      <div className="metrics-panel">
        <h3>Показатели КТГ</h3>
        <div className="no-data">Ожидание данных...</div>
      </div>
    );
  }

  const {
    bpm,
    basal_rhythm,
    hrv,
    // acceleration,
    // deceleration,
    hypoxia_15,
    hypoxia_30,
    hypoxia_60
  } = latestFetusData;

  return (
    <div className="metrics-panel">
      {/* <h3>Показатели КТГ</h3> */}

      <div className='bpm-and-risk'>
        <div className="fetus-heart-rate">
          <h4>ЧСС (уд/мин)</h4>
          <div className="heart-rate-value">{bpm}</div>
        </div>

      <div className="hypoxia-risk">
        <h4>Риск Гипоксии</h4>
        <div className="hypoxia-prediction">
          <div className="prediction-item">
            {/* <div className="prediction-time">15 минут</div> */}
            <div className="prediction-value">
              15 мин:  {hypoxia_15}%
            </div>
          </div>
          <div className="prediction-item">
            {/* <div className="prediction-time">30 минут</div> */}
            <div className="prediction-value">
              30 мин: {hypoxia_30}%
            </div>
          </div>
          <div className="prediction-item">
            {/* <div className="prediction-time">60 минут</div> */}
            <div className="prediction-value">
              60 мин: {hypoxia_60}%
            </div>
          </div>
        </div>
      </div>
      </div>
      

      <div className="other-metrics">
        <div className="metric-item">
          <span className="metric-label">Базальный ритм (уд/мин)</span>
          <span className="metric-value">{basal_rhythm}</span>
        </div>

        <div className="metric-item">
          <span className="metric-label">Вариабельность (мс)</span>
          <span className="metric-value">{hrv}</span>
        </div>

        {/* <div className="metric-item">
          <span className="metric-label">Акцелерация</span>
          <span className="metric-value">
            {acceleration ? 'Есть' : 'Нет'}
          </span>
        </div>

        <div className="metric-item">
          <span className="metric-label">Децелерация</span>
          <span className="metric-value">
            {deceleration ? 'Есть' : 'Нет'}
          </span>
        </div> */}
      </div>
    </div>
  );
};