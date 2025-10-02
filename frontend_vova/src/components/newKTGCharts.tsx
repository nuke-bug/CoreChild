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

//  ДОБАВЛЯЕМ ТИП ДЛЯ ПРЕОБРАЗОВАННЫХ ДАННЫХ
interface TransformedFetusData extends FetusData {
  transformedTime: number;
}

interface TransformedUterusData extends UterusData {
  transformedTime: number;
}

export const KTGCharts: React.FC<KTGChartsProps> = ({
  fetusData,
  uterusData,
  height = 300,
  timeWindow = 1800 // 30 минут в секундах
}) => {

  //  ВЫЧИСЛЯЕМ ДИНАМИЧЕСКИЕ ГРАНИЦЫ ОСИ X С СДВИГОМ ПРИ 25 МИНУТАХ
  const dynamicXAxis = useMemo(() => {
    if (fetusData.length === 0 && uterusData.length === 0) {
      return { min: 0, max: timeWindow };
    }

    // Находим максимальное время из всех данных
    const allData = [...fetusData, ...uterusData];
    const maxTime = Math.max(...allData.map(d => d.time || 0));
    
    //  ЕСЛИ МЕНЬШЕ 25 МИНУТ (1500 СЕКУНД) - ПОКАЗЫВАЕМ 0-30 МИНУТ
    if (maxTime <= 1500) {
      return { min: 0, max: timeWindow };
    }
    
    //  ПРИ ДОСТИЖЕНИИ 25+ МИНУТ СДВИГАЕМ ОКНО НА 5 МИНУТ ВПЕРЕД
    const stepSize = 300; // 5 минут в секундах
    const shiftThreshold = 1500; // 25 минут в секундах
    
    // Вычисляем на сколько шагов сдвинуть окно
    const stepsFromStart = Math.floor((maxTime - shiftThreshold) / stepSize) + 1;
    const minTime = stepsFromStart * stepSize;
    const maxTimeAdjusted = minTime + timeWindow;
    
    return { min: minTime, max: maxTimeAdjusted };
  }, [fetusData, uterusData, timeWindow]);

  //  ПРЕОБРАЗОВАНИЕ ДАННЫХ В ОТНОСИТЕЛЬНОЕ ВРЕМЯ В ОКНЕ
  const transformDataToFixedAxis = (data: FetusData[] | UterusData[]) => {
    if (data.length === 0) {
      // Для пустого графика создаем точки по границам текущего окна
      return [
        { transformedTime: 0, bpm: 0, basal_rhythm: 0 },
        { transformedTime: timeWindow, bpm: 0, basal_rhythm: 0 }
      ] as any;
    }

    return data
      .filter(item => item && typeof item.time === 'number')
      .map(item => {
        //  ПРЕОБРАЗУЕМ В ОТНОСИТЕЛЬНОЕ ВРЕМЯ В ТЕКУЩЕМ ОКНЕ
        const relativeTime = item.time - dynamicXAxis.min;
        return {
          ...item,
          transformedTime: Math.max(0, relativeTime)
        };
      })
      .filter(item => item.transformedTime <= timeWindow); // Отсекаем данные за пределами окна
  };

  const transformedFetusData = useMemo(() => 
    transformDataToFixedAxis(fetusData), 
    [fetusData, dynamicXAxis, timeWindow]
  ) as TransformedFetusData[];

  const transformedUterusData = useMemo(() => 
    transformDataToFixedAxis(uterusData), 
    [uterusData, dynamicXAxis, timeWindow]
  ) as TransformedUterusData[];

  //  ФУНКЦИЯ ФОРМАТИРОВАНИЯ ВРЕМЕНИ (АБСОЛЮТНЫЕ ЗНАЧЕНИЯ)
  const formatTimeTick = (tickValue: string | number) => {
    // Преобразуем tickValue в число (относительное время в окне)
    const relativeTime = typeof tickValue === 'string' ? parseFloat(tickValue) : tickValue;
    
    // Проверяем, что это валидное число
    if (isNaN(relativeTime)) return '';
    
    //  ПРЕОБРАЗУЕМ В АБСОЛЮТНОЕ ВРЕМЯ (МИНУТЫ ОТ НАЧАЛА ЗАПИСИ)
    const absoluteTime = dynamicXAxis.min + relativeTime;
    const minutes = Math.floor(absoluteTime / 60);
    
    return `${minutes} мин`;
  };
  
  // Подготовка данных для графика ЧСС плода
  const fetusChartData = {
    datasets: [
      // Основной график
      {
        label: 'ЧСС плода (BPM)',
        data: transformedFetusData.map((d: TransformedFetusData) => ({ 
          x: d.transformedTime,
          y: fetusData.length > 0 ? d.bpm : null
        })),
        borderColor: 'rgb(75, 192, 192)',
        backgroundColor: 'rgba(75, 192, 192, 0.1)',
        borderWidth: 2,
        pointRadius: 0,
        tension: 0.2,
        yAxisID: 'y',
        fill: true,
      },
      // Маркеры акселераций (только когда есть данные)
      ...(fetusData.length > 0 ? [{
        label: 'Акселерация',
        data: transformedFetusData
          .filter((d: TransformedFetusData) => d.acceleration)
          .map((d: TransformedFetusData) => ({ 
            x: d.transformedTime,
            y: d.bpm 
          })),
        borderColor: 'rgb(76, 175, 80)',
        backgroundColor: 'rgb(76, 175, 80)',
        borderWidth: 3,
        pointRadius: 4,
        pointStyle: 'line' as const,
        tension: 0,
        yAxisID: 'y',
        showLine: false,
      }] : []),
      // Маркеры децелераций (только когда есть данные)
      ...(fetusData.length > 0 ? [{
        label: 'Децелерация',
        data: transformedFetusData
          .filter((d: TransformedFetusData) => d.deceleration)
          .map((d: TransformedFetusData) => ({ 
            x: d.transformedTime,
            y: d.bpm 
          })),
        borderColor: 'rgb(244, 67, 54)',
        backgroundColor: 'rgb(244, 67, 54)',
        borderWidth: 3,
        pointRadius: 4,
        pointStyle: 'line' as const,
        tension: 0,
        yAxisID: 'y',
        showLine: false,
      }] : []),
      // Базальный ритм
      {
        label: 'Базальный ритм',
        data: transformedFetusData.map((d: TransformedFetusData) => ({ 
          x: d.transformedTime,
          y: fetusData.length > 0 ? d.basal_rhythm : null
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
        data: transformedUterusData.map((d: TransformedUterusData) => ({ 
          x: d.transformedTime,
          y: uterusData.length > 0 ? d.power : null
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

  //  ОБНОВЛЕННЫЕ НАСТРОЙКИ С ФИКСИРОВАННОЙ ОСЬЮ X
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
        min: 0, //  ФИКСИРОВАННЫЙ MIN (начало окна)
        max: timeWindow, //  ФИКСИРОВАННЫЙ MAX (конец окна)
        ticks: {
          stepSize: 300, // 5 минут
          callback: formatTimeTick,
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
    </div>
  );
};