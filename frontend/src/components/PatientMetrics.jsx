import React from 'react';
import MetricsDisplay from './MetricsDisplay';

const PatientMetrics = () => {
  const [currentInterface, setCurrentInterface] = React.useState('');
  const [bpmMetrics, setBpmMetrics] = React.useState([]);
  const [uterusMetrics, setUterusMetrics] = React.useState([]);
  const [activeTab, setActiveTab] = React.useState('bpm');

  const handleInterfaceSelect = (interfaceName) => {
    setCurrentInterface(interfaceName);
    setBpmMetrics([]);
    setUterusMetrics([]);
  };

  const handleMetricsLoad = (metricType, data) => {
    if (metricType === 'bpm') {
      setBpmMetrics(data);
      setActiveTab('bpm');
    } else {
      setUterusMetrics(data);
      setActiveTab('uterus');
    }
  };

  return (
    <div className="patient-metrics">
      <h2>Метрики пациента</h2>
      
      <InterfaceList 
        onInterfaceSelect={handleInterfaceSelect}
        onMetricsLoad={handleMetricsLoad}
      />

      {currentInterface && (
        <div className="metrics-tabs">
          <div className="tab-buttons">
            <button 
              className={activeTab === 'bpm' ? 'active' : ''}
              onClick={() => setActiveTab('bpm')}
            >
              BPM Метрики
            </button>
            <button 
              className={activeTab === 'uterus' ? 'active' : ''}
              onClick={() => setActiveTab('uterus')}
            >
              Uterus Метрики
            </button>
          </div>

          <div className="tab-content">
            {activeTab === 'bpm' ? (
              <MetricsDisplay metrics={bpmMetrics} metricType="bpm" />
            ) : (
              <MetricsDisplay metrics={uterusMetrics} metricType="uterus" />
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default PatientMetrics;
