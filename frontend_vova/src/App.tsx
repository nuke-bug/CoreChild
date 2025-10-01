// src/App.tsx
import { KTGCharts } from "./components/KTGCharts"
import { MetricsPanel } from './components/MetricsPanel';
import { ConnectionStatus } from './components/StatusPanel';
// import { useWebSocket } from './hooks/useWebSocket';
import { useWebSocket } from "./hooks/useWebSocket";
import './App.css';

/**
 * Главный компонент приложения мониторинга КТГ
 * 
 * Функциональность:
 * - Подключение к WebSocket серверу
 * - Отображение графиков ЧСС плода и активности матки
 * - Отображение текущих метрик и статусов
 * - Управление подключением и данными
 */

function App() {
  const fetusWsUrl = 'ws://10.0.0.3:8000/ws/fetus';
  const uterusWsUrl = 'ws://10.0.0.3:8000/ws/uterus';
  
  const {
    fetusData,
    uterusData,
    lastUpdate,
    isFetusConnected,
    isUterusConnected,
    error,
    // clearData,
    fetusDataPoints,
    uterusDataPoints
  } = useWebSocket(fetusWsUrl, uterusWsUrl);

  // Получаем последние данные для панели метрик
  const latestFetusData = fetusData.length > 0 ? fetusData[fetusData.length - 1] : null;

  return (
    <div className="app">
      {/* Заголовок приложения */}
      <header className="app-header">
        <h1> Монитор КТГ</h1>
      </header>

      {/* Основное содержимое */}
      <main className="main-content">
        {/* Левая колонка - графики */}
        <section className="charts-section">
          <KTGCharts
            fetusData={fetusData}
            uterusData={uterusData}
            height={300}
          />
        </section>

        {/* Правая колонка - панели */}
        <aside className="sidebar">
          {/* Статус подключения */}
          <ConnectionStatus
            isConnected={isFetusConnected}
            lastUpdate={lastUpdate}
            dataPointsCount={fetusDataPoints}
            error={error}
          />

          <ConnectionStatus
            isConnected={isUterusConnected}
            lastUpdate={lastUpdate}
            dataPointsCount={uterusDataPoints}
            error={error}
          />

          {/* Метрики плода */}
          <MetricsPanel latestFetusData={latestFetusData} />

          {/* Панель управления */}
          {/* <div className="control-panel">
            <h3>Управление</h3>
            <button
              onClick={clearData}
              disabled={dataPointsCount === 0}
              className="btn-clear"
            >
              🗑️ Очистить данные
            </button>
            <div className="connection-info">
              Подключение: {wsUrl}
            </div>
          </div> */}
        </aside>
      </main>

      {/* Футер */}
      {/* <footer className="app-footer">
        <div className="footer-content">
          <span>КТГ Монитор v1.0</span>
          <span>Режим реального времени</span>
        </div>
      </footer> */}
    </div>
  );
}

export default App;