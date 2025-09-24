import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const MetricsDisplay = ({ metrics, metricType }) => {
  if (!metrics || metrics.length === 0) {
    return (
      <div className="metrics-display">
        <h3>{metricType === 'bpm' ? 'BPM Метрики' : 'Uterus Метрики'}</h3>
        <p>Нет данных для отображения</p>
      </div>
    );
  }

  // Преобразуем данные для графика
  const chartData = metrics.map(metric => ({
    time: new Date(metric.time * 1000).toLocaleTimeString(),
    value: metricType === 'bpm' ? metric.bpm : metric.power,
    ...(metricType === 'bpm' && {
      basalRhythm: metric.basal_rhythm,
      hrv: metric.hrv
    })
  }));

  return (
    <div className="metrics-display">
      <h3>{metricType === 'bpm' ? 'BPM Метрики' : 'Uterus Метрики'}</h3>
      
      <div className="chart-container">
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="time" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Line 
              type="monotone" 
              dataKey="value" 
              stroke="#8884d8" 
              name={metricType === 'bpm' ? 'BPM' : 'Power'} 
            />
            {metricType === 'bpm' && (
              <>
                <Line type="monotone" dataKey="basalRhythm" stroke="#82ca9d" name="Базальный ритм" />
                <Line type="monotone" dataKey="hrv" stroke="#ffc658" name="HRV" />
              </>
            )}
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="metrics-table">
        <h4>Последние данные:</h4>
        <table>
          <thead>
            <tr>
              <th>Время</th>
              {metricType === 'bpm' ? (
                <>
                  <th>BPM</th>
                  <th>Базальный ритм</th>
                  <th>HRV</th>
                  <th>Акселерация</th>
                  <th>Децелерация</th>
                  <th>Гипоксия</th>
                </>
              ) : (
                <>
                  <th>Сила</th>
                  <th>Сокращение</th>
                </>
              )}
            </tr>
          </thead>
          <tbody>
            {metrics.slice(-10).map((metric, index) => (
              <tr key={index}>
                <td>{new Date(metric.time * 1000).toLocaleTimeString()}</td>
                {metricType === 'bpm' ? (
                  <>
                    <td>{metric.bpm}</td>
                    <td>{metric.basal_rhythm}</td>
                    <td>{metric.hrv}</td>
                    <td>{metric.acceleration ? 'Да' : 'Нет'}</td>
                    <td>{metric.deceleration ? 'Да' : 'Нет'}</td>
                    <td className={metric.hypoxia ? 'warning' : ''}>
                      {metric.hypoxia ? 'Да' : 'Нет'}
                    </td>
                  </>
                ) : (
                  <>
                    <td>{metric.power}</td>
                    <td>{metric.contraction ? 'Да' : 'Нет'}</td>
                  </>
                )}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default MetricsDisplay;
