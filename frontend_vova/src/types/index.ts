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
  hypoxia_15: number | null;
  hypoxia_30: number | null;
  hypoxia_60: number | null;
  timestamp: string;
  type: 'fetus';
}

// Данные матки
export interface UterusData {
  time: number;
  power: number;
  contraction: boolean;
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