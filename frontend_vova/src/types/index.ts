// src/types/index.ts
export type StatusType = 'normal' | 'suspicious' | 'pathological';

// Данные плода
export interface FetusData {
  time: number;
  bpm: number;
  basal_rhythm: number;
  hrv: number;
  acceleration: boolean;
  deceleration: boolean;
  hypoxia: StatusType;
  basal_status: StatusType;
  hrv_status: StatusType;
  decel_status: StatusType;
  accel_status: StatusType;
  hypoxia_15: number;
  hypoxia_30: number;
  hypoxia_60: number;
  timestamp: string;
  type: 'fetus';
}

// Данные матки
export interface UterusData {
  time: number;
  power: number;
  timestamp: string;
  type: 'uterus';
}

// Состояние приложения
export interface AppState {
  fetusData: FetusData[];
  uterusData: UterusData[];
  lastUpdate: Date | null;
  isFetusConnected: boolean;
  isUterusConnected: boolean;
  error: string | null;
}