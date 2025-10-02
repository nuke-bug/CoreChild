// src/components/charts/KTGCharts.tsx
import React, { useMemo } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  type ChartOptions,
  TimeScale
} from 'chart.js';
import { Line } from 'react-chartjs-2';
import { type FetusData, type UterusData } from '../types';

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
  timeWindow?: number; // в секундах (30 минут = 1800 секунд)
}

export const KTGCharts: React.FC<KTGChartsProps> = ({
  fetusData,
  uterusData,
  height = 300,
  timeWindow = 1800 // 30 минут в секундах
}) => {

  const FIXED_X_AXIS = {
    min: 0,
    max: timeWindow
  };

  const transformDataToFixedAxis = (data: FetusData[] | UterusData[]) => {
    if (data.length === 0) return [];

    const latestTime = data[data.length - 1]?.time || 0;
    
    // Вычисляем смещение времени
    const timeOffset = Math.max(0, latestTime - timeWindow);
    
    return data
      .filter(item => item && typeof item.time === 'number')
      .map(item => ({
        ...item,
        transformedTime: Math.max(0, item.time - timeOffset)
      }))
      .filter(item => item.transformedTime <= timeWindow); // Отсекаем данные за пределами окна
  };

  const transformedFetusData = useMemo(() => 
    transformDataToFixedAxis(fetusData), 
    [fetusData, timeWindow]
  );

  const transformedUterusData = useMemo(() => 
    transformDataToFixedAxis(uterusData), 
    [uterusData, timeWindow]
  );

  //  ФИКСИРОВАННЫЕ ДЕЛЕНИЯ КАЖДЫЕ 5 МИНУТ
  const fixedTicks = useMemo(() => {
    const ticks = [];
    const tickInterval = 300; // 5 минут в секундах
    
    // Создаем деления от 0 до timeWindow с шагом 5 минут
    for (let time = 0; time <= timeWindow; time += tickInterval) {
      ticks.push(time);
    }
    
    return ticks;
  }, [timeWindow]);
  
  // Подготовка данных для графика ЧСС плода
const fetusChartData = {
  datasets: [
    // Основной график
    {
      label: 'ЧСС плода (BPM)',
      data: transformedFetusData.map(d => ({ 
        x: (d as any).transformedTime,
        y: (d as FetusData).bpm 
      })),
      borderColor: 'rgb(75, 192, 192)',
      backgroundColor: 'rgba(75, 192, 192, 0.1)',
      borderWidth: 2,
      pointRadius: 0,
      tension: 0.2,
      yAxisID: 'y',
      fill: true,
    },
    //  Маркеры акселераций
    {
      label: 'Акселерация',
      data: transformedFetusData
        .filter(d => (d as FetusData).acceleration)
        .map(d => ({ 
          x: (d as any).transformedTime,
          y: (d as FetusData).bpm 
        })),
      borderColor: 'rgb(76, 175, 80)',
      backgroundColor: 'rgb(76, 175, 80)',
      borderWidth: 3,
      pointRadius: 4,
      pointStyle: 'line',
      tension: 0,
      yAxisID: 'y',
      showLine: false, // Только точки, без линий
    },
    //  Маркеры децелераций
    {
      label: 'Децелерация',
      data: transformedFetusData
        .filter(d => (d as FetusData).deceleration)
        .map(d => ({ 
          x: (d as any).transformedTime,
          y: (d as FetusData).bpm 
        })),
      borderColor: 'rgb(244, 67, 54)',
      backgroundColor: 'rgb(244, 67, 54)',
      borderWidth: 3,
      pointRadius: 4,
      pointStyle: 'line',
      tension: 0,
      yAxisID: 'y',
      showLine: false, // Только точки, без линий
    },
    // Базальный ритм
    {
      label: 'Базальный ритм',
      data: transformedFetusData.map(d => ({ 
        x: (d as any).transformedTime,
        y: (d as FetusData).basal_rhythm 
      })),
      borderColor: 'rgb(255, 99, 132)',
      backgroundColor: 'rgba(255, 99, 132, 0.1)',
      borderWidth: 1,
      pointRadius: 0,
      borderDash: [5, 5],
      tension: 0.2,
      yAxisID: 'y',
    }
  ]
};

  // Подготовка данных для графика активности матки
  const uterusChartData = {
    datasets: [
      {
        label: 'Активность матки',
        data: transformedUterusData.map(d => ({ 
          x: (d as any).transformedTime, //  Используем преобразованное время
          y: (d as UterusData).power 
        })),
        borderColor: 'rgb(153, 102, 255)',
        backgroundColor: 'rgba(153, 102, 255, 0.1)',
        borderWidth: 2,
        pointRadius: 0,
        tension: 0.2,
        yAxisID: 'y',
        fill: true,
      }
    ]
  };

  //  ОБЩИЕ НАСТРОЙКИ С АБСОЛЮТНО ФИКСИРОВАННОЙ ОСЬЮ X
  const commonOptions: ChartOptions<'line'> = {
    responsive: true,
    maintainAspectRatio: false,
    animation: {
      duration: 0
    },
    scales: {
      x: {
        type: 'linear',
        title: {
          display: false,
          text: 'Время (минуты)'
        },
        max: FIXED_X_AXIS.max,
        ticks: {
          stepSize: 300, // 5 минут в секундах
          callback: function(value) {
            // Форматируем время в минуты
            const totalSeconds = Number(value);
            const minutes = Math.floor(totalSeconds / 60);
            return `${minutes} мин`;
          },
          autoSkip: false,
          maxTicksLimit: 7
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
        enabled: false
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
        min: 50,
        max: 210,
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
        min: 0,
        max: 100,
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
        <Line
          data={fetusChartData}
          options={fetusChartOptions}
        />
      </div>

      {/* График активности матки */}
      <div className="chart-container" style={{ height }}>
        <h3>Токограмма - Активность матки</h3>
        <Line
          data={uterusChartData}
          options={uterusChartOptions}
        />
      </div>
      <div className='hidden'>{ fixedTicks }</div>
    </div>
  );
};