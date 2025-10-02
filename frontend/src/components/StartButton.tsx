import React, { useState } from 'react';
import axios from 'axios';


export const StartButton: React.FC = () => {
  const [isLoading, setIsLoading] = useState(false);
  // const url = 'http://127.0.0.1:9009'
  const url = 'http://10.0.0.3:8000/start'

  const handleStart = async () => {
    setIsLoading(true);
    
    try {
      await axios.get(url);
    } catch (error) {
      console.error('Ошибка:', error);
    } finally {
      setIsLoading(false);
      location.reload()
    }
  };

  return (
    <div className="button-status">
        <button 
        className="start-button"
        onClick={handleStart}
        disabled={isLoading}
      >
      {isLoading ? 'Подключение...' : 'Запуск'}
      </button>
    </div>
  );
};

export const StopButton: React.FC = () => {
  const [isLoading, setIsLoading] = useState(false);
  // const url = 'http://127.0.0.1:9009'
  const url = 'http://10.0.0.3:8000/stop'

  const handleStop = async () => {
    setIsLoading(true);
    
    try {
      await axios.get(url);
    } catch (error) {
      console.error('Ошибка:', error);
    } finally {
      setIsLoading(false);
      location.reload()
    }
  };

  return (
    <div className="button-status">
        <button 
        className="stop-button"
        onClick={handleStop}
        disabled={isLoading}
      >
      {isLoading ? 'Отключение...' : 'Стоп'}
      </button>
    </div>
  );
};