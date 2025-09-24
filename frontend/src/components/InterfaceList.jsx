import React from 'react';
import { medicalAPI } from '../services/api';

const InterfaceList = ({ onInterfaceSelect, onMetricsLoad }) => {
  const [interfaces, setInterfaces] = React.useState([]);
  const [loading, setLoading] = React.useState(false);
  const [selectedInterface, setSelectedInterface] = React.useState('');

  const loadInterfaces = async () => {
    try {
      setLoading(true);
      const response = await medicalAPI.getInterfaces();
      setInterfaces(response.data);
    } catch (error) {
      console.error('Error loading interfaces:', error);
      alert('Ошибка загрузки интерфейсов');
    } finally {
      setLoading(false);
    }
  };

  const handleRunInterface = async (interfaceName) => {
    try {
      setLoading(true);
      await medicalAPI.runInterface(interfaceName);
      setSelectedInterface(interfaceName);
      onInterfaceSelect(interfaceName);
      alert(`Интерфейс ${interfaceName} запущен`);
    } catch (error) {
      console.error('Error running interface:', error);
      alert('Ошибка запуска интерфейса');
    } finally {
      setLoading(false);
    }
  };

  const handleLoadMetrics = async (interfaceName, metricType) => {
    try {
      setLoading(true);
      let response;
      if (metricType === 'bpm') {
        response = await medicalAPI.getBpmMetrics(interfaceName);
      } else {
        response = await medicalAPI.getUterusMetrics(interfaceName);
      }
      onMetricsLoad(metricType, response.data);
    } catch (error) {
      console.error('Error loading metrics:', error);
      alert('Ошибка загрузки метрик');
    } finally {
      setLoading(false);
    }
  };

  React.useEffect(() => {
    loadInterfaces();
  }, []);

  return (
    <div className="interface-list">
      <h2>Интерфейсы мониторинга</h2>
      
      <button onClick={loadInterfaces} disabled={loading}>
        {loading ? 'Загрузка...' : 'Обновить интерфейсы'}
      </button>

      <div className="interfaces-grid">
        {interfaces.map((iface) => (
          <div key={iface.name} className="interface-card">
            <h3>{iface.name}</h3>
            <p>Статус: {iface.is_active ? 'Активен' : 'Неактивен'}</p>
            <p>ID пациента: {iface.id_patient || 'Не назначен'}</p>
            
            <div className="interface-actions">
              <button 
                onClick={() => handleRunInterface(iface.name)}
                disabled={loading}
                className="run-btn"
              >
                Запустить
              </button>
              
              {selectedInterface === iface.name && (
                <div className="metrics-actions">
                  <button onClick={() => handleLoadMetrics(iface.name, 'bpm')}>
                    Загрузить BPM
                  </button>
                  <button onClick={() => handleLoadMetrics(iface.name, 'uterus')}>
                    Загрузить Uterus
                  </button>
                </div>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default InterfaceList;
