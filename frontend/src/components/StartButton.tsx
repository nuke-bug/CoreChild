// src/components/SimpleStartButton.tsx
import React, { useState } from 'react';
import axios from 'axios';

export const StartButton: React.FC = () => {
  const [isLoading, setIsLoading] = useState(false);
//   const url = 'http://127.0.0.1:9009'
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
    <button 
      onClick={handleStart}
      disabled={isLoading}
      style={{
        padding: '10px 20px',
        backgroundColor: '#007bff',
        color: 'white',
        border: 'none',
        borderRadius: '5px',
        cursor: isLoading ? 'not-allowed' : 'pointer'
      }}
    >
      {isLoading ? 'Отправка...' : 'Запуск'}
    </button>
  );
};