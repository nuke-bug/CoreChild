// src/services/websocketService.ts
// import { type FetusData, type UterusData } from '../types/index'
import { type FetusData, type UterusData } from '../types/index'

class WebSocketService {
  private fetusWs: WebSocket | null = null;
  private uterusWs: WebSocket | null = null;
  private reconnectAttempts = { fetus: 0, uterus: 0 };
  private maxReconnectAttempts = 5;
  private reconnectInterval = 3000;
  private isManualDisconnect = false;

  // Подключение к WebSocket плода
  connectFetus(url: string, onMessage: (data: FetusData) => void): void {
    this.connectWebSocket(
      url, 
      onMessage, 
      'fetus',
      () => this.reconnectFetus(url, onMessage)
    );
  }

  // Подключение к WebSocket матки
  connectUterus(url: string, onMessage: (data: UterusData) => void): void {
    this.connectWebSocket(
      url, 
      onMessage, 
      'uterus',
      () => this.reconnectUterus(url, onMessage)
    );
  }

  private connectWebSocket(
    url: string, 
    onMessage: (data: any) => void,
    type: 'fetus' | 'uterus',
    onReconnect: () => void
  ): void {
    try {
      const ws = new WebSocket(url);
      
      ws.onopen = () => {
        console.log(`✅ ${type} WebSocket connected`);
        this.reconnectAttempts[type] = 0;
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          onMessage(data);
        } catch (error) {
          console.error(`Error parsing ${type} WebSocket message:`, error);
        }
      };

      ws.onclose = (event) => {
        console.log(`🔴 ${type} WebSocket disconnected: ${event.code}`);
        if (!this.isManualDisconnect) {
          onReconnect();
        }
      };

      ws.onerror = (error) => {
        console.error(`💥 ${type} WebSocket error:`, error);
      };

      if (type === 'fetus') {
        this.fetusWs = ws;
      } else {
        this.uterusWs = ws;
      }

    } catch (error) {
      console.error(`💥 ${type} WebSocket connection failed:`, error);
    }
  }

  private reconnectFetus(url: string, onMessage: (data: FetusData) => void): void {
    this.handleReconnect('fetus', url, onMessage);
  }

  private reconnectUterus(url: string, onMessage: (data: UterusData) => void): void {
    this.handleReconnect('uterus', url, onMessage);
  }

  private handleReconnect(
    type: 'fetus' | 'uterus', 
    url: string, 
    onMessage: (data: any) => void
  ): void {
    if (this.reconnectAttempts[type] < this.maxReconnectAttempts) {
      this.reconnectAttempts[type]++;
      console.log(`🔄 Reconnecting ${type}... Attempt ${this.reconnectAttempts[type]}/${this.maxReconnectAttempts}`);
      
      setTimeout(() => {
        if (type === 'fetus') {
          this.connectFetus(url, onMessage);
        } else {
          this.connectUterus(url, onMessage);
        }
      }, this.reconnectInterval);
    } else {
      console.error(`❌ Max reconnection attempts reached for ${type}`);
    }
  }

  disconnectAll(): void {
    this.isManualDisconnect = true;
    if (this.fetusWs) {
      this.fetusWs.close();
      this.fetusWs = null;
    }
    if (this.uterusWs) {
      this.uterusWs.close();
      this.uterusWs = null;
    }
    this.reconnectAttempts = { fetus: 0, uterus: 0 };
  }

  getFetusStatus(): string {
    return this.getWebSocketStatus(this.fetusWs);
  }

  getUterusStatus(): string {
    return this.getWebSocketStatus(this.uterusWs);
  }

  private getWebSocketStatus(ws: WebSocket | null): string {
    if (!ws) return 'DISCONNECTED';
    
    switch (ws.readyState) {
      case WebSocket.CONNECTING: return 'CONNECTING';
      case WebSocket.OPEN: return 'CONNECTED';
      case WebSocket.CLOSING: return 'CLOSING';
      case WebSocket.CLOSED: return 'DISCONNECTED';
      default: return 'UNKNOWN';
    }
  }
}

export const websocketService = new WebSocketService();