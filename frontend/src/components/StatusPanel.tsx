// src/components/ui/ConnectionStatus.tsx
import React from 'react';

interface ConnectionStatusProps {
  isConnected: boolean;
  lastUpdate: Date | null;
  dataPointsCount: number;
  type?: 'fetus' | 'uterus';
}

export const ConnectionStatus: React.FC<ConnectionStatusProps> = ({
  isConnected,
  // lastUpdate,
  type = 'fetus'
}) => {
  const getConnectionConfig = () => {
    switch (type) {
      case 'fetus':
        return {
          label: 'Fetus',
          connectedText: 'Подключено',
          disconnectedText: 'Нет подключения'
        };
      case 'uterus':
        return {
          label: 'Uterus',
          connectedText: 'Подключено',
          disconnectedText: 'Нет подключения'
        };
    }
  };

  const config = getConnectionConfig();

  return (
    <div className="connection-status">
      {/* <div className="connection-label">{config.label}</div> */}
      <div className={`connection-indicator ${isConnected ? 'connected' : 'disconnected'}`}>
       {config.label} {isConnected ? config.connectedText : config.disconnectedText}
      </div>
      {/* <div className="status-info">
        {lastUpdate ? lastUpdate.toLocaleTimeString('ru-RU') : '—'}
      </div> */}
    </div>
  );
};