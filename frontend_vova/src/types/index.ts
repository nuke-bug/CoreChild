// Базовые статусы
export type StatusType = 'normal' | 'suspicious' | 'pathological';

// Данные плода
export interface FetusData {
  time: number;           // время в секундах
  bpm: number;           // ЧСС плода
  basal_rhythm: number;  // базальный ритм
  hrv: number;          // вариабельность сердечного ритма
  acceleration: boolean; // есть акселерация
  deceleration: boolean; // есть децелерация
  hypoxia: StatusType;   // статус гипоксии
  basal_status: StatusType;
  hrv_status: StatusType;
  decel_status: StatusType;
  accel_status: StatusType;
  hypoxia_15: number | null; // риск гипоксии через 15 мин
  hypoxia_30: number | null; // риск гипоксии через 30 мин
  hypoxia_60: number | null; // риск гипоксии через 60 мин
}

// Данные матки
export interface UterusData {
  time: number;      // время в секундах
  power: number;     // сила сокращения
  contraction: boolean; // есть сокращение
}

// Полные данные КТГ
export interface KTGData {
  fetus: FetusData;
  uterus: UterusData;
}

// Состояние приложения
export interface AppState {
  fetusData: FetusData[];
  uterusData: UterusData[];
  lastUpdate: Date | null;
  isConnected: boolean;
  error: string | null;
}