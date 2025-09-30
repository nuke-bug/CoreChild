// src/components/charts/KTGCharts.tsx
import React, { useRef, useEffect } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  ChartOptions,
  TimeScale
} from 'chart.js';
import { Chart } from 'react-chartjs-2';
import { FetusData, UterusData } from '../types/index';

// Регистрируем необходимые компоненты Chart.js
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  TimeScale
);

interface KTGChartsProps {
  fetusData: FetusData[];
  uterusData: UterusData[];
  height?: number;
}

/**
 * Компонент для отображения графиков КТГ
 * - Верхний график: ЧСС плода (BPM) и базальный ритм
 * - Нижний график: Активность матки
 */
export const KTGCharts: React.FC<KTGChartsProps> = ({
  fetusData,
  uterusData,
  height = 300
}) => {
  const chartRef = useRef<ChartJS>(null);

  // Подготовка данных для графика ЧСС плода
  const fetusChartData = {
    labels: fetusData.map(d => d.time),
    datasets: [
      {
        label: 'ЧСС плода (BPM)',
        data: fetusData.map(d => d.bpm),
        borderColor: 'rgb(75, 192, 192)',
        backgroundColor: 'rgba(75, 192, 192, 0.1)',
        borderWidth: 2,
        pointRadius: 0, // Убираем точки для лучшей производительности
        tension: 0.1,   // Легкое сглаживание кривой
        yAxisID: 'y',
      },
      {
        label: 'Базальный ритм',
        data: fetusData.map(d => d.basal_rhythm),
        borderColor: 'rgb(255, 99, 132)',
        backgroundColor: 'rgba(255, 99, 132, 0.1)',
        borderWidth: 1,
        pointRadius: 0,
        borderDash: [5, 5], // Пунктирная линия
        tension: 0.1,
        yAxisID: 'y',
      }
    ]
  };

  // Подготовка данных для графика активности матки
  const uterusChartData = {
    labels: uterusData.map(d => d.time),
    datasets: [
      {
        label: 'Активность матки',
        data: uterusData.map(d => d.power),
        borderColor: 'rgb(153, 102, 255)',
        backgroundColor: 'rgba(153, 102, 255, 0.1)',
        borderWidth: 2,
        pointRadius: 0,
        tension: 0.1,
        yAxisID: 'y',
      }
    ]
  };

  // Общие настройки для графиков
  const commonOptions: ChartOptions<'line'> = {
    responsive: true,
    maintainAspectRatio: false,
    animation: {
      duration: 0 // Отключаем анимацию для реального времени
    },
    interaction: {
      mode: 'index' as const,
      intersect: false,
    },
    scales: {
      x: {
        type: 'linear' as const,
        title: {
          display: true,
          text: 'Время (секунды)'
        },
        grid: {
          color: 'rgba(0, 0, 0, 0.1)'
        }
      },
      y: {
        beginAtZero: false,
        grid: {
          color: 'rgba(0, 0, 0, 0.1)'
        }
      }
    },
    plugins: {
      legend: {
        position: 'top' as const,
      },
      tooltip: {
        mode: 'index' as const,
        intersect: false
      }
    }
  };

  // Специфические настройки для графика ЧСС плода
  const fetusChartOptions: ChartOptions<'line'> = {
    ...commonOptions,
    scales: {
      ...commonOptions.scales,
      y: {
        ...commonOptions.scales?.y,
        min: 50,   // Минимальная ЧСС
        max: 210,  // Максимальная ЧСС
        title: {
          display: true,
          text: 'ЧСС (уд/мин)'
        }
      }
    }
  };

  // Специфические настройки для графика активности матки
  const uterusChartOptions: ChartOptions<'line'> = {
    ...commonOptions,
    scales: {
      ...commonOptions.scales,
      y: {
        ...commonOptions.scales?.y,
        min: 0,    // Минимальная активность
        max: 100,  // Максимальная активность
        title: {
          display: true,
          text: 'Активность (%)'
        }
      }
    }
  };

  return (
    <div className="ktg-charts">
      {/* График ЧСС плода */}
      <div className="chart-container" style={{ height }}>
        <h3>Кардиотокограмма - ЧСС плода</h3>
        <Chart
          type="line"
          data={fetusChartData}
          options={fetusChartOptions}
        />
      </div>

      {/* График активности матки */}
      <div className="chart-container" style={{ height }}>
        <h3>Токограмма - Активность матки</h3>
        <Chart
          type="line"
          data={uterusChartData}
          options={uterusChartOptions}
        />
      </div>
    </div>
  );
};