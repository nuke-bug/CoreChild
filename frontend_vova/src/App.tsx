// src/App.tsx
import { KTGCharts } from "./components/KTGCharts"
import { MetricsPanel } from './components/MetricsPanel';
import { ConnectionStatus } from './components/StatusPanel';
import { StartButton, StopButton } from './components/StartButton';
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
  // const fetusWsUrl = 'ws://10.0.0.3:8000/ws/fetus';
  // const uterusWsUrl = 'ws://10.0.0.3:8000/ws/uterus';
  const fetusWsUrl = 'ws://localhost:9009/ws/fetus';
  const uterusWsUrl = 'ws://localhost:9009/ws/uterus';
  
  const {
    fetusData,
    uterusData,
    lastUpdate,
    isFetusConnected,
    isUterusConnected,
    // error,
    fetusDataPoints,
    uterusDataPoints
  } = useWebSocket(fetusWsUrl, uterusWsUrl);

  // Получаем последние данные для панели метрик
  const latestFetusData = fetusData.length > 0 ? fetusData[fetusData.length - 1] : null;

  return (
    <div className="app">
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
          
          {/* Метрики плода */}
          <MetricsPanel latestFetusData={latestFetusData} />

          <div className="connection-container">
            <div className="buttons-container">
              <StartButton/>
              <StopButton/>
            </div>
            <div className="connection-status-container">
              <ConnectionStatus
                isConnected={isFetusConnected}
                lastUpdate={lastUpdate}
                dataPointsCount={fetusDataPoints}
                type="fetus"
              />

              <ConnectionStatus
                isConnected={isUterusConnected}
                lastUpdate={lastUpdate}
                dataPointsCount={uterusDataPoints}
                type="uterus"
              />
            </div>
          </div>
        </aside>
      </main>
    </div>
  );
}

export default App;