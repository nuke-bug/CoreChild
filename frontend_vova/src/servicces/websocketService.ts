// src/services/websocketService.ts
import { KTGData } from "../types/index";

class WebSocketService {
  private ws: WebSocket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectInterval = 3000;
  private messageCallbacks: ((data: KTGData) => void)[] = [];
  private statusCallbacks: ((status: boolean) => void)[] = [];

  /**
   * Подключение к WebSocket серверу
   * @param url - URL WebSocket сервера
   */

  
  connect(url: string): void {
    try {
      console.log(`Connecting to WebSocket: ${url}`);
      this.ws = new WebSocket(url);

      this.ws.onopen = () => {
        console.log('WebSocket connected successfully');
        this.reconnectAttempts = 0;
        this.notifyStatusChange(true);
      };

      this.ws.onmessage = (event) => {
        try {
          const data: KTGData = JSON.parse(event.data);
          this.notifyMessageCallbacks(data);
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };

      this.ws.onclose = (event) => {
        console.log(`WebSocket disconnected: ${event.code} - ${event.reason}`);
        this.notifyStatusChange(false);
        this.handleReconnect(url);
      };

      this.ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        this.notifyStatusChange(false);
      };

    } catch (error) {
      console.error('WebSocket connection failed:', error);
    }
  }

  /**
   * Обработка переподключения при разрыве соединения
   */
  private handleReconnect(url: string): void {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      console.log(`Reconnecting... Attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts}`);
      
      setTimeout(() => {
        this.connect(url);
      }, this.reconnectInterval);
    } else {
      console.error('Maximum reconnection attempts reached');
    }
  }

  /**
   * Отключение от WebSocket сервера
   */
  disconnect(): void {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    this.reconnectAttempts = 0;
  }

  /**
   * Подписка на получение данных
   */
  onMessage(callback: (data: KTGData) => void): void {
    this.messageCallbacks.push(callback);
  }

  /**
   * Подписка на изменение статуса подключения
   */
  onStatusChange(callback: (connected: boolean) => void): void {
    this.statusCallbacks.push(callback);
  }

  /**
   * Уведомление подписчиков о новых данных
   */
  private notifyMessageCallbacks(data: KTGData): void {
    this.messageCallbacks.forEach(callback => {
      try {
        callback(data);
      } catch (error) {
        console.error('Error in message callback:', error);
      }
    });
  }

  /**
   * Уведомление подписчиков об изменении статуса
   */
  private notifyStatusChange(connected: boolean): void {
    this.statusCallbacks.forEach(callback => {
      try {
        callback(connected);
      } catch (error) {
        console.error('Error in status callback:', error);
      }
    });
  }

  /**
   * Получение текущего статуса подключения
   */
  getStatus(): string {
    if (!this.ws) return 'DISCONNECTED';
    
    switch (this.ws.readyState) {
      case WebSocket.CONNECTING: return 'CONNECTING';
      case WebSocket.OPEN: return 'CONNECTED';
      case WebSocket.CLOSING: return 'CLOSING';
      case WebSocket.CLOSED: return 'DISCONNECTED';
      default: return 'UNKNOWN';
    }
  }
}

// Создаем единственный экземпляр сервиса
export const websocketService = new WebSocketService();