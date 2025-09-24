import React from 'react';
import InterfaceList from './components/InterfaceList';
import MetricsDisplay from './components/MetricsDisplay';
import './App.css';

function App() {
  const [currentInterface, setCurrentInterface] = React.useState('');
  const [bpmData, setBpmData] = React.useState([]);
  const [uterusData, setUterusData] = React.useState([]);
  const [activeTab, setActiveTab] = React.useState('bpm');
  const [status, setStatus] = React.useState('');

  // Простая проверка бэка
  const checkBackend = async () => {
    try {
      const response = await fetch('/api/health');
      if (response) {
        setStatus('✅ Бэкенд работает');
      } else {
        setStatus('❌ Бэкенд не отвечает');
      }
    } catch (error) {
      setStatus('❌ Ошибка подключения к бэкенду');
    }
  };

  React.useEffect(() => {
    checkBackend();
  }, []);

  return (
    <div className="App">
      <header>
        <h1>Монитор плода</h1>
        <div className="status">{status}</div>
      </header>

      <main>
        <InterfaceList 
          onInterfaceSelect={setCurrentInterface}
          onBpmData={setBpmData}
          onUterusData={setUterusData}
        />

        {currentInterface && (
          <div className="metrics-section">
            <div className="tabs">
              <button 
                className={activeTab === 'bpm' ? 'active' : ''}
                onClick={() => setActiveTab('bpm')}
              >
                ЧСС плода (BPM)
              </button>
              <button 
                className={activeTab === 'uterus' ? 'active' : ''}
                onClick={() => setActiveTab('uterus')}
              >
                Сокращения матки
              </button>
            </div>

            {activeTab === 'bpm' ? (
              <MetricsDisplay 
                data={bpmData} 
                type="bpm" 
                interfaceName={currentInterface}
              />
            ) : (
              <MetricsDisplay 
                data={uterusData} 
                type="uterus" 
                interfaceName={currentInterface}
              />
            )}
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
