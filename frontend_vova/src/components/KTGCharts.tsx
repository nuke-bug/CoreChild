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
  timeWindow?: number;
}

/**
 * Компонент для отображения графиков КТГ
 * - Верхний график: ЧСС плода (BPM) и базальный ритм
 * - Нижний график: Активность матки
 */
export const KTGCharts: React.FC<KTGChartsProps> = ({
  fetusData,
  uterusData,
  height = 300,
  timeWindow = 1800
}) => {
  // const chartRef = useRef<ChartJS>(null);

// ✅ ДИНАМИЧЕСКАЯ ОСЬ X: вычисляем границы на основе данных
  const dynamicXAxis = useMemo(() => {
    if (fetusData.length === 0 && uterusData.length === 0) {
      return { min: 0, max: timeWindow };
    }

    // Находим максимальное время из всех данных
    const allTimes = [
      ...fetusData.map(d => d.time),
      ...uterusData.map(d => d.time)
    ].filter(time => typeof time === 'number');
    
    if (allTimes.length === 0) {
      return { min: 0, max: timeWindow };
    }

    const maxTime = Math.max(...allTimes);
    const minTime = Math.max(0, maxTime - timeWindow);

    return {
      min: minTime,
      max: maxTime
    };
  }, [fetusData, uterusData, timeWindow]);

  // ✅ ФИЛЬТРАЦИЯ ДАННЫХ: показываем только данные в текущем окне
  const filteredFetusData = useMemo(() => {
    return fetusData.filter(item => 
      item && 
      typeof item.time === 'number' && 
      item.time >= dynamicXAxis.min && 
      item.time <= dynamicXAxis.max
    );
  }, [fetusData, dynamicXAxis]);

  const filteredUterusData = useMemo(() => {
    return uterusData.filter(item => 
      item && 
      typeof item.time === 'number' && 
      item.time >= dynamicXAxis.min && 
      item.time <= dynamicXAxis.max
    );
  }, [uterusData, dynamicXAxis]);

  // // ✅ СОЗДАЕМ ДЕЛЕНИЯ ДЛЯ ОСИ X
  // const xAxisTicks = useMemo(() => {
  //   const ticks = [];
  //   const tickInterval = 30; // Деления каждые 30 секунд
    
  //   // Начинаем с округленного значения минимального времени
  //   let currentTick = Math.floor(dynamicXAxis.min / tickInterval) * tickInterval;
    
  //   while (currentTick <= dynamicXAxis.max) {
  //     ticks.push(currentTick);
  //     currentTick += tickInterval;
  //   }
    
  //   return ticks;
  // }, [dynamicXAxis]);

  //   const transformDataToFixedAxis = (data: FetusData[] | UterusData[]) => {
  //     if (data.length === 0) return [];
  
  //     const latestTime = data[data.length - 1]?.time || 0;
      
  //     // Вычисляем смещение времени
  //     const timeOffset = Math.max(0, latestTime - timeWindow);
      
  //     return data
  //       .filter(item => item && typeof item.time === 'number')
  //       .map(item => ({
  //         ...item,
  //         transformedTime: Math.max(0, item.time - timeOffset) // Сдвигаем время
  //       }))
  //       .filter(item => item.transformedTime <= timeWindow); // Отсекаем данные за пределами окна
  //   };

  // const transformedFetusData = transformDataToFixedAxis(fetusData);
  // const transformedUterusData = transformDataToFixedAxis(uterusData);

  // Подготовка данных для графика ЧСС плода
  const fetusChartData = {
    // labels: fetusData.map(d => d.time),
    datasets: [
      {
        label: 'ЧСС плода (BPM)',
        data: filteredFetusData.map(d => ({ 
          x: d.time,
          y: d.bpm 
        })),
        borderColor: 'rgb(75, 192, 192)',
        backgroundColor: 'rgba(75, 192, 192, 0.1)',
        borderWidth: 2,
        pointRadius: 0, // Убираем точки для лучшей производительности
        tension: 0.2,   // Легкое сглаживание кривой
        yAxisID: 'y',
        fill: true,
      },
      {
        label: 'Базальный ритм',
        data: filteredFetusData.map(d => ({ 
          x: d.time, // ✅ Используем реальное время
          y: d.basal_rhythm 
        })),
        borderColor: 'rgb(255, 99, 132)',
        backgroundColor: 'rgba(255, 99, 132, 0.1)',
        borderWidth: 1,
        pointRadius: 0,
        borderDash: [5, 5], // Пунктирная линия
        tension: 0.2,
        yAxisID: 'y',
      }
    ]
  };

  // Подготовка данных для графика активности матки
  const uterusChartData = {
    // labels: uterusData.map(d => d.time),
    datasets: [
      {
        label: 'Активность матки',
        data: filteredUterusData.map(d => ({ 
          x: d.time, // ✅ Используем реальное время
          y: d.power 
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

  // Общие настройки для графиков
  const commonOptions: ChartOptions<'line'> = {
    responsive: true,
    maintainAspectRatio: false,
    animation: {
      duration: 0 // Отключаем анимацию для реального времени
    },
    // interaction: {
    //   mode: 'dataset',
    //   intersect: false,
    // },
    scales: {
      x: {
        type: 'linear',
        title: {
          display: true,
          text: 'Время (минуты)'
        },
        min: dynamicXAxis.min,
        max: dynamicXAxis.max,
        ticks: {
          stepSize: 60, // Деления каждые 30 секунд
          callback: function(value) {
            // Форматируем время в минуты:секунды
            const totalSeconds = Number(value);
            const minutes = Math.floor(totalSeconds / 60);
            const seconds = Math.floor(totalSeconds % 60);
            return `${minutes}:${seconds.toString().padStart(2, '0')}`;
          }
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
        // mode: 'index',
        // intersect: false,
        // callbacks: {
        //   title: (items) => {
        //     // Показываем реальное время в tooltip
        //     const transformedTime = items[0].parsed.x;
        //     const latestTime = fetusData.length > 0 ? fetusData[fetusData.length - 1].time : 0;
        //     const realTime = latestTime - (timeWindow - transformedTime);
            
        //     const minutes = Math.floor(realTime / 60);
        //     const seconds = Math.floor(realTime % 60);
        //     return `Время: ${minutes}:${seconds.toString().padStart(2, '0')}`;
        //   }
        // }
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
    </div>
  );
};