import axios from 'axios';

const API_BASE_URL = '/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const medicalAPI = {
  // Получить список интерфейсов
  getInterfaces: () => api.get('/interfaces'),
  
  // Запустить интерфейс
  runInterface: (interfaceName) => api.get(`/${interfaceName}/run`),
  
  // Получить BPM метрики
  getBpmMetrics: (interfaceName) => api.get(`/${interfaceName}/bpm`),
  
  // Получить uterus метрики
  getUterusMetrics: (interfaceName) => api.get(`/${interfaceName}/uterus`),
  
  // Проверить здоровье сервиса
  healthCheck: () => api.get('/health'),
};

export default api;
