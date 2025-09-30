// src/components/ui/ConnectionStatus.tsx
import React from 'react';

interface ConnectionStatusProps {
  isConnected: boolean;
  lastUpdate: Date | null;
  dataPointsCount: number;
  error: string | null;
}

/**
 * Компонент для отображения статуса подключения и статистики
 */
export const ConnectionStatus: React.FC<ConnectionStatusProps> = ({
  isConnected,
  lastUpdate,
  dataPointsCount,
  error
}) => {
  return (
    <div className="connection-status">
      <div className="status-row">
        <span className="status-label">Статус:</span>
        <span className={`status-indicator ${isConnected ? 'connected' : 'disconnected'}`}>
          {isConnected ? '● Подключено' : '◼ Отключено'}
        </span>
      </div>

      <div className="status-row">
        <span className="status-label">Точек данных:</span>
        <span className="status-value">{dataPointsCount}</span>
      </div>

      <div className="status-row">
        <span className="status-label">Последнее обновление:</span>
        <span className="status-value">
          {lastUpdate ? lastUpdate.toLocaleTimeString('ru-RU') : '—'}
        </span>
      </div>

      {error && (
        <div className="error-message">
          ⚠️ {error}
        </div>
      )}
    </div>
  );
};