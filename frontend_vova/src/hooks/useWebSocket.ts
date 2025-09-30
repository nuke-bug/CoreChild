// src/hooks/useWebSocket.ts
import { useState, useEffect, useCallback } from 'react';
import { KTGData, FetusData, UterusData } from '../types/index';
import { websocketService } from '../servicces/websocketService';

/**
 * Хук для управления WebSocket соединением и данными КТГ
 */
export const useWebSocket = (url: string, maxDataPoints: number = 600) => {
  // Состояние приложения
  const [fetusData, setFetusData] = useState<FetusData[]>([]);
  const [uterusData, setUterusData] = useState<UterusData[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);

  /**
   * Обработка новых данных от WebSocket
   */
  const handleNewData = useCallback((data: KTGData) => {
    try {
      setFetusData(prev => {
        const newData = [...prev, data.fetus];
        // Ограничиваем количество точек данных для производительности
        return newData.slice(-maxDataPoints);
      });

      setUterusData(prev => {
        const newData = [...prev, data.uterus];
        return newData.slice(-maxDataPoints);
      });

      setLastUpdate(new Date());
      setError(null); // Сбрасываем ошибку при успешном получении данных
      
    } catch (err) {
      console.error('Error processing KTG data:', err);
      setError('Ошибка обработки данных');
    }
  }, [maxDataPoints]);

  /**
   * Обработка изменения статуса подключения
   */
  const handleStatusChange = useCallback((connected: boolean) => {
    setIsConnected(connected);
    if (!connected) {
      setError('Потеряно соединение с сервером');
    } else {
      setError(null);
    }
  }, []);

  // Эффект для управления WebSocket соединением
  useEffect(() => {
    // Подписываемся на события WebSocket
    websocketService.onMessage(handleNewData);
    websocketService.onStatusChange(handleStatusChange);

    // Подключаемся к WebSocket
    websocketService.connect(url);

    // Очистка при размонтировании компонента
    return () => {
      websocketService.disconnect();
    };
  }, [url, handleNewData, handleStatusChange]);

  /**
   * Очистка данных
   */
  const clearData = useCallback(() => {
    setFetusData([]);
    setUterusData([]);
    setLastUpdate(null);
  }, []);

  return {
    // Данные
    fetusData,
    uterusData,
    lastUpdate,
    
    // Статус
    isConnected,
    error,
    
    // Методы
    clearData,
    
    // Статистика
    dataPointsCount: fetusData.length
  };
};