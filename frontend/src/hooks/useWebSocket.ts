// src/hooks/useWebSocket.ts
import { useState, useEffect, useCallback } from 'react';
import { type FetusData, type UterusData } from '../types/index';
import { websocketService } from '../services/websocketService';

export const useWebSocket = (fetusUrl: string, uterusUrl: string) => {
  const [fetusData, setFetusData] = useState<FetusData[]>([]);
  const [uterusData, setUterusData] = useState<UterusData[]>([]);
  const [isFetusConnected, setIsFetusConnected] = useState(false);
  const [isUterusConnected, setIsUterusConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);

  // Обработчик данных плода
  const handleFetusData = useCallback((data: FetusData) => {
    try {
      setFetusData(prev => [...prev, data]);
      setLastUpdate(new Date());
      setError(null);
    } catch (err) {
      console.error('Error processing fetus data:', err);
    }
  }, []);

  // Обработчик данных матки
  const handleUterusData = useCallback((data: UterusData) => {
    try {
      setUterusData(prev => [...prev, data]);
      setLastUpdate(new Date());
      setError(null);
    } catch (err) {
      console.error('Error processing uterus data:', err);
    }
  }, []);

  // Мониторинг статуса соединений
  useEffect(() => {
    const checkConnections = setInterval(() => {
      setIsFetusConnected(websocketService.getFetusStatus() === 'CONNECTED');
      setIsUterusConnected(websocketService.getUterusStatus() === 'CONNECTED');
      
      if (!isFetusConnected && !isUterusConnected) {
        setError('Нет подключения к серверу КТГ');
      } else {
        setError(null);
      }
    }, 1000);

    return () => clearInterval(checkConnections);
  }, [isFetusConnected, isUterusConnected]);

  // Подключение WebSocket
  useEffect(() => {
    console.log(`🔄 Connecting to WebSockets...`);
    console.log(`👶 Fetus: ${fetusUrl}`);
    console.log(`🤰 Uterus: ${uterusUrl}`);
    
    websocketService.connectFetus(fetusUrl, handleFetusData);
    websocketService.connectUterus(uterusUrl, handleUterusData);

    return () => {
      websocketService.disconnectAll();
    };
  }, [fetusUrl, uterusUrl, handleFetusData, handleUterusData]);

  const clearData = useCallback(() => {
    setFetusData([]);
    setUterusData([]);
    setLastUpdate(null);
  }, []);

  return {
    fetusData,
    uterusData,
    lastUpdate,
    isFetusConnected,
    isUterusConnected,
    error,
    clearData,
    fetusDataPoints: fetusData.length,
    uterusDataPoints: uterusData.length
  };
};