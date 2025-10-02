import pandas as pd
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Optional, Deque
from enum import IntEnum
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.interpolate import interp1d
from tensorflow import keras


class HypoxiaStatus(IntEnum):
    """Статус гипоксии (по возрастанию тяжести)."""
    NORMAL = 0
    SUSPICIOUS = 1
    PATHOLOGICAL = 2


# Глобальные параметры стандартизации
X_MEAN = np.array([[[141.45479082, 20.05265615]]])
X_STD = np.array([[[20.58988719, 20.33744231]]])


def load_hypoxia_models(
    model_15_path: str = "model_15.h5",
    model_30_path: str = "model_30.h5",
    model_60_path: str = "model_60.h5"
) -> Tuple[Optional[keras.Model], Optional[keras.Model], Optional[keras.Model]]:
    """
    Загружает три модели прогнозирования гипоксии.

    Args:
        model_15_path: Путь к модели для прогноза на 15 минут
        model_30_path: Путь к модели для прогноза на 30 минут
        model_60_path: Путь к модели для прогноза на 60 минут

    Returns:
        Кортеж (model_15, model_30, model_60) или (None, None, None) при ошибке
    """
    try:
        model_15 = keras.models.load_model(model_15_path)
        model_30 = keras.models.load_model(model_30_path)
        model_60 = keras.models.load_model(model_60_path)
        return model_15, model_30, model_60
    except Exception as e:
        print(f"Ошибка загрузки моделей: {e}")
        return None, None, None


def prepare_window_for_prediction(
    bpm_df: pd.DataFrame,
    uterus_df: pd.DataFrame,
    current_time: float,
    window_length_sec: int = 600,
    min_window_sec: int = 60
) -> Optional[np.ndarray]:
    """
    Подготавливает окно данных для прогнозирования.

    Args:
        bpm_df: DataFrame с данными ЧСС (колонки 'time_sec', 'value')
        uterus_df: DataFrame с данными токограммы
        current_time: Текущее время (секунды)
        window_length_sec: Целевая длина окна (600 сек = 10 минут)
        min_window_sec: Минимальная длина окна для прогноза (60 сек = 1 минута)

    Returns:
        Массив формы (1, window_length, 2) или None если данных недостаточно
    """
    if bpm_df.empty or uterus_df is None or uterus_df.empty:
        return None

    # Гарантируем правильные названия колонок
    bpm_df = bpm_df.copy()
    uterus_df = uterus_df.copy()

    if 'time_sec' not in bpm_df.columns:
        bpm_df = bpm_df.rename(columns={bpm_df.columns[0]: 'time_sec', bpm_df.columns[1]: 'value'})
    if 'time_sec' not in uterus_df.columns:
        uterus_df = uterus_df.rename(columns={uterus_df.columns[0]: 'time_sec', uterus_df.columns[1]: 'value'})

    bpm_df = bpm_df.sort_values("time_sec")
    uterus_df = uterus_df.sort_values("time_sec")

    # Определяем начало окна
    t_min = max(bpm_df['time_sec'].min(), uterus_df['time_sec'].min())
    available_duration = current_time - t_min

    # Проверяем минимальную длину
    if available_duration < min_window_sec:
        return None

    # Используем доступные данные, но не больше window_length_sec
    actual_window = min(available_duration, window_length_sec)
    window_start = current_time - actual_window

    # Фильтруем данные в окне
    bpm_window = bpm_df[
        (bpm_df['time_sec'] >= window_start) & 
        (bpm_df['time_sec'] <= current_time)
    ]
    uterus_window = uterus_df[
        (uterus_df['time_sec'] >= window_start) & 
        (uterus_df['time_sec'] <= current_time)
    ]

    if len(bpm_window) < 2 or len(uterus_window) < 2:
        return None

    # Равномерная сетка по времени (1 сек шаг)
    # Если окно меньше 600 сек, используем фактическую длину
    actual_length = int(actual_window)
    t_common = np.linspace(window_start, current_time, actual_length)

    # Интерполяция
    bpm_interp = interp1d(
        bpm_window['time_sec'], bpm_window['value'],
        bounds_error=False, fill_value="extrapolate"
    )(t_common)

    uterus_interp = interp1d(
        uterus_window['time_sec'], uterus_window['value'],
        bounds_error=False, fill_value="extrapolate"
    )(t_common)

    signals = np.stack([bpm_interp, uterus_interp], axis=1)

    # Если окно меньше целевой длины, паддинг нулями
    if actual_length < window_length_sec:
        padding = np.zeros((window_length_sec - actual_length, 2))
        signals = np.vstack([padding, signals])

    # Стандартизация
    signals = (signals - X_MEAN[0]) / X_STD[0]

    return np.expand_dims(signals, axis=0)  # (1, window_length, 2)


def predict_hypoxia_all(
    bpm_df: pd.DataFrame,
    uterus_df: pd.DataFrame,
    current_time: float,
    model_15: keras.Model,
    model_30: keras.Model,
    model_60: keras.Model
) -> Optional[Tuple[float, float, float]]:
    """
    Прогнозирует вероятность гипоксии на 15, 30 и 60 минут.

    Args:
        bpm_df: DataFrame с данными ЧСС
        uterus_df: DataFrame с данными токограммы
        current_time: Текущее время (секунды)
        model_15: Модель для прогноза на 15 минут
        model_30: Модель для прогноза на 30 минут
        model_60: Модель для прогноза на 60 минут

    Returns:
        Кортеж (prob_15, prob_30, prob_60) или None если данных недостаточно
    """
    X = prepare_window_for_prediction(bpm_df, uterus_df, current_time)
    if X is None:
        return None

    # Получаем прогнозы от каждой модели
    pred_15 = model_15.predict(X, verbose=0)[0, 0]
    pred_30 = model_30.predict(X, verbose=0)[0, 0]
    pred_60 = model_60.predict(X, verbose=0)[0, 0]

    return float(pred_15), float(pred_30), float(pred_60)


@dataclass
class CTGConfig:
    """Конфигурация параметров анализа CTG."""

    # Параметры baseline и событий
    window_max: int = 10 * 60  # секунд (максимальное окно для медианы)
    event_threshold: int = 15   # bpm (отклонение от baseline)
    event_duration: int = 15    # секунд (минимальная длительность события)
    coverage_ratio: float = 0.7 # минимальное покрытие интервала

    # Параметры ВСР
    vsr_window: int = 60        # секунд (окно для ВСР)
    segment_sec: int = 10       # секунд (подсегменты для амплитуды)

    # Параметры артефактов
    artifact_diff: int = 25     # bpm (порог скачка для артефакта)
    artifact_gap: int = 3       # секунд (макс. длительность артефакта)

    # Диапазон допустимых значений ЧСС
    bpm_min: int = 90           # минимальное допустимое значение ЧСС
    bpm_max: int = 190          # максимальное допустимое значение ЧСС

    # Параметры статуса гипоксии
    # Базальный ритм
    baseline_normal_min: int = 110
    baseline_normal_max: int = 160
    baseline_path_min: int = 100
    baseline_path_max: int = 180
    baseline_path_duration: int = 10 * 60  # 10 минут

    # ВСР
    vsr_normal_min: int = 5
    vsr_normal_max: int = 25
    vsr_path_low_duration: int = 50 * 60   # 50 минут
    vsr_path_high_duration: int = 30 * 60  # 30 минут

    # Децелерации
    decel_window: int = 30 * 60            # 30 минут
    decel_path_duration: int = 3 * 60      # 3 минуты

    # Акселерации
    accel_normal_window: int = 20 * 60     # 20 минут
    accel_normal_count: int = 2
    accel_path_window: int = 40 * 60       # 40 минут

    # Параметры ML прогнозирования
    ml_prediction_interval: int = 60       # секунд (интервал между прогнозами)


def is_late_deceleration(
    dec_start: float,
    df_toco: pd.DataFrame,
    window_sec: int = 20,
    min_delta: float = 15,
    early_min: int = 5,
    late_max: int = 5
) -> bool:
    """
    Определяет, является ли децелерация поздней по данным токограммы.

    Параметры
    ---------
    dec_start : float
        Время начала децелерации (секунды).
    df_toco : pd.DataFrame
        DataFrame с токограммой. Должны быть колонки ['time_sec', 'value'].
    window_sec : int, optional
        Длина окна перед децелерацией (по умолчанию 20 секунд).
    min_delta : float, optional
        Минимальная амплитуда «горки» (разница между максимумом и минимумом), по умолчанию 15.
    early_min : int, optional
        Максимальное время (в секундах) от начала окна, где должен находиться минимум, по умолчанию 5.
    late_max : int, optional
        Максимальное время до конца окна, где должен находиться максимум, по умолчанию 5.

    Возвращает
    ----------
    bool
        True – поздняя децелерация (сокращение перед ней есть),
        False – сокращение не обнаружено.
    """
    if df_toco is None or df_toco.empty:
        return False

    # --- Выбираем окно перед децелерацией ---
    seg_start = dec_start - window_sec
    segment = df_toco[
        (df_toco['time_sec'] >= seg_start) &
        (df_toco['time_sec'] <= dec_start)
    ]

    if len(segment) < 2:
        return False  # данных недостаточно

    times = segment['time_sec'].to_numpy()
    values = segment['value'].to_numpy()

    # --- Находим локальный максимум и минимум ---
    max_idx = np.argmax(values)
    min_idx = np.argmin(values)

    t_max = times[max_idx] - seg_start  # позиция пика в окне
    t_min = times[min_idx] - seg_start  # позиция минимума в окне

    # --- Критерии «горки» ---
    cond_max = t_max >= (window_sec - late_max)  # пик в конце окна
    cond_min = t_min <= early_min  # минимум в начале окна
    cond_delta = (values[max_idx] - values[min_idx]) >= min_delta  # достаточная амплитуда

    return cond_max and cond_min and cond_delta


class CTGAnalyzer:
    """Анализатор данных CTG (кардиотокографии)."""

    def __init__(
        self,
        config: CTGConfig = None,
        df_uterus: pd.DataFrame = None,
        model_15: keras.Model = None,
        model_30: keras.Model = None,
        model_60: keras.Model = None
    ):
        self.config = config or CTGConfig()
        self.df_uterus = df_uterus
        self.model_15 = model_15
        self.model_30 = model_30
        self.model_60 = model_60

        # Буферы для скользящих окон
        self.baseline_buffer: Deque[Tuple[float, float]] = deque()
        self.vsr_buffer: Deque[Tuple[float, float, bool]] = deque()

        # Результаты анализа
        self.baselines: List[int] = []
        self.vsr_values: List[int] = []
        self.acc_flags: List[bool] = []
        self.dec_flags: List[bool] = []
        self.events: List[Tuple[float, float, float, float, str]] = []
        self.hypoxia_statuses: List[str] = []
        self.baseline_statuses: List[str] = []
        self.vsr_statuses: List[str] = []
        self.decel_statuses: List[str] = []
        self.accel_statuses: List[str] = []

        # ML прогнозы
        self.ml_predictions_15: List[int] = []
        self.ml_predictions_30: List[int] = []
        self.ml_predictions_60: List[int] = []
        self.last_ml_prediction_time: Optional[float] = None

        # Состояние событий
        self.current_event: Optional[str] = None
        self.event_start: Optional[float] = None
        self.event_baseline: Optional[float] = None

        # Фильтр артефактов
        self.last_valid_bpm: Optional[float] = None
        self.last_valid_time: Optional[float] = None
        self.candidate_artifact: Optional[Tuple[float, float]] = None

        # История для статуса гипоксии
        self.time_history: List[float] = []
        self.baseline_history: List[float] = []
        self.vsr_history: List[float] = []

        # Начальное значение baseline (медиана первых валидных значений)
        self.initial_baseline: Optional[float] = None
        self.initial_baseline_calculated: bool = False

    def filter_artifact(self, time: float, bpm_raw: float) -> float:
        """
        Фильтрует артефакты на основе резких скачков ЧСС.

        Args:
            time: Время измерения
            bpm_raw: Сырое значение ЧСС

        Returns:
            Отфильтрованное значение ЧСС
        """
        if self.last_valid_bpm is None:
            self.last_valid_bpm = bpm_raw
            self.last_valid_time = time
            return bpm_raw

        diff = abs(bpm_raw - self.last_valid_bpm)

        if diff > self.config.artifact_diff:
            # Подозрение на артефакт
            if self.candidate_artifact is None:
                self.candidate_artifact = (time, bpm_raw)

            # Если прошло больше artifact_gap секунд, это не артефакт
            if time - self.candidate_artifact[0] > self.config.artifact_gap:
                self.last_valid_bpm = bpm_raw
                self.last_valid_time = time
                self.candidate_artifact = None
                return bpm_raw
            else:
                # Всплеск < artifact_gap сек → считаем артефактом
                return self.last_valid_bpm
        else:
            # Скачок небольшой → обновляем как норму
            self.last_valid_bpm = bpm_raw
            self.last_valid_time = time
            self.candidate_artifact = None
            return bpm_raw

    def update_buffers(self, time: float, bpm: float):
        """Обновляет буферы скользящих окон."""
        # Проверяем, находится ли ЧСС в допустимом диапазоне
        is_valid_bpm = self.config.bpm_min <= bpm <= self.config.bpm_max

        # Baseline буфер - добавляем только валидные значения
        if is_valid_bpm:
            self.baseline_buffer.append((time, bpm))

        while self.baseline_buffer and \
              (time - self.baseline_buffer[0][0]) > self.config.window_max:
            self.baseline_buffer.popleft()

        # VSR буфер - добавляем только валидные значения
        if is_valid_bpm:
            is_in_event = self.current_event is not None
            self.vsr_buffer.append((time, bpm, is_in_event))

        while self.vsr_buffer and \
              (time - self.vsr_buffer[0][0]) > self.config.vsr_window:
            self.vsr_buffer.popleft()

    def calculate_initial_baseline(self, df: pd.DataFrame) -> float:
        """
        Вычисляет начальный baseline из первых валидных значений.

        Args:
            df: DataFrame с данными

        Returns:
            Начальное значение baseline
        """
        # Берем первые 60 секунд или 20 точек данных
        initial_data = df.head(min(60, len(df)))

        # Фильтруем валидные значения
        valid_values = initial_data[
            (initial_data['value'] >= self.config.bpm_min) &
            (initial_data['value'] <= self.config.bpm_max)
        ]['value']

        if len(valid_values) >= 5:
            # Используем медиану для устойчивости к выбросам
            return float(np.median(valid_values))
        elif len(valid_values) > 0:
            return float(np.mean(valid_values))
        else:
            # Если все значения невалидны, используем среднее из всех
            return float(np.median(initial_data['value']))

    def calculate_baseline(self) -> float:
        """Вычисляет текущий baseline."""
        if self.current_event is None:
            if len(self.baseline_buffer) > 0:
                window_vals = [bpm for _, bpm in self.baseline_buffer]
                baseline = float(np.mean(window_vals))
                self.event_baseline = baseline
            elif len(self.baselines) > 0:
                # Если буфер пуст (все значения выходили за пределы), берем предыдущее
                baseline = self.baselines[-1]
            elif self.initial_baseline is not None:
                # Используем начальный baseline
                baseline = self.initial_baseline
            else:
                # Значение по умолчанию
                baseline = 140.0

            self.event_baseline = baseline
        else:
            baseline = self.event_baseline

        return baseline

    def check_event_start(self, time: float, bpm: float,
                          baseline: float, df: pd.DataFrame) -> None:
        """Проверяет начало события (акселерация/децелерация)."""
        if self.current_event is not None:
            return

        deviation = bpm - baseline

        # Проверка на акселерацию
        if deviation >= self.config.event_threshold:
            if self._validate_event(time, df, baseline, 'accel'):
                self.current_event = 'accel'
                self.event_start = time

        # Проверка на децелерацию
        elif deviation <= -self.config.event_threshold:
            if self._validate_event(time, df, baseline, 'decel'):
                self.current_event = 'decel'
                self.event_start = time

    def _validate_event(self, time: float, df: pd.DataFrame,
                       baseline: float, event_type: str) -> bool:
        """Валидирует начало события по будущим значениям."""
        future = df[
            (df['time_sec'] >= time) &
            (df['time_sec'] <= time + self.config.event_duration)
        ]

        if len(future) <= 1:
            return False

        coverage = future['time_sec'].iloc[-1] - future['time_sec'].iloc[0]
        if coverage < self.config.event_duration * self.config.coverage_ratio:
            return False

        diff = future['value'] - baseline

        if event_type == 'accel':
            return (diff >= self.config.event_threshold).all()
        else:  # decel
            return (diff <= -self.config.event_threshold).all()

    def check_event_end(self, time: float, baseline: float,
                       df: pd.DataFrame) -> None:
        """Проверяет окончание текущего события."""
        if self.current_event is None:
            return

        future = df[
            (df['time_sec'] >= time) &
            (df['time_sec'] <= time + self.config.event_duration)
        ]

        if len(future) <= 1:
            return

        coverage = future['time_sec'].iloc[-1] - future['time_sec'].iloc[0]
        if coverage < self.config.event_duration * self.config.coverage_ratio:
            return

        diff = future['value'] - baseline

        should_end = False
        if self.current_event == 'accel':
            should_end = (diff <= self.config.event_threshold).all()
        elif self.current_event == 'decel':
            should_end = (diff >= -self.config.event_threshold).all()

        if should_end:
            seg = df[
                (df['time_sec'] >= self.event_start) &
                (df['time_sec'] <= time)
            ]
            self.events.append((
                self.event_start,
                time,
                seg['value'].min(),
                seg['value'].max(),
                self.current_event
            ))
            self.current_event = None
            self.event_start = None

    def calculate_vsr(self) -> int:
        """
        Вычисляет ВСР (вариабельность сердечного ритма) как среднюю амплитуду.

        Returns:
            Значение ВСР в bpm
        """
        # Берем только точки вне событий
        vsr_points = [
            (t, bpm) for t, bpm, is_event in self.vsr_buffer
            if not is_event
        ]

        if len(vsr_points) < 2:
            # Если недостаточно данных, возвращаем предыдущее значение
            if len(self.vsr_values) > 0:
                return self.vsr_values[-1]
            return 0

        vsr_array = np.array(vsr_points)
        seg_start = vsr_array[0, 0]
        seg_amplitudes = []

        while seg_start < vsr_array[-1, 0]:
            seg_end = seg_start + self.config.segment_sec
            seg_mask = (vsr_array[:, 0] >= seg_start) & (vsr_array[:, 0] < seg_end)
            seg_vals = vsr_array[seg_mask, 1]

            if len(seg_vals) > 0:
                seg_amplitudes.append(seg_vals.max() - seg_vals.min())

            seg_start = seg_end

        if seg_amplitudes:
            return round(np.mean(seg_amplitudes))
        elif len(self.vsr_values) > 0:
            return self.vsr_values[-1]
        else:
            return 0

    def evaluate_hypoxia_status(self, current_time: float) -> Tuple[str, str, str, str, str]:
        """
        Оценивает статус гипоксии на основе всех критериев.

        Args:
            current_time: Текущее время

        Returns:
            Кортеж: (общий_статус, baseline_status, vsr_status, decel_status, accel_status)
        """
        statuses = []

        # 1. Базальный ритм
        baseline_status = self._evaluate_baseline_status(current_time)
        baseline_status_str = self._status_to_string(baseline_status) if baseline_status is not None else 'normal'
        if baseline_status is not None:
            statuses.append(baseline_status)

        # 2. ВСР
        vsr_status = self._evaluate_vsr_status(current_time)
        vsr_status_str = self._status_to_string(vsr_status) if vsr_status is not None else 'normal'
        if vsr_status is not None:
            statuses.append(vsr_status)

        # 3. Децелерации (пока их нет - normal)
        decel_status = self._evaluate_deceleration_status(current_time)
        decel_status_str = self._status_to_string(decel_status) if decel_status is not None else 'normal'
        if decel_status is not None:
            statuses.append(decel_status)
        else:
            statuses.append(HypoxiaStatus.NORMAL)

        # 4. Акселерации (в первые 20 минут - normal)
        accel_status = self._evaluate_acceleration_status(current_time)
        accel_status_str = self._status_to_string(accel_status) if accel_status is not None else 'normal'
        if accel_status is not None:
            statuses.append(accel_status)
        else:
            statuses.append(HypoxiaStatus.NORMAL)

        # Если нет ни одного оцененного критерия, возвращаем normal
        if not statuses:
            overall_status = 'normal'
        else:
            # Возвращаем худший статус
            max_status = max(statuses)
            overall_status = self._status_to_string(max_status)

        return overall_status, baseline_status_str, vsr_status_str, decel_status_str, accel_status_str

    def _status_to_string(self, status: HypoxiaStatus) -> str:
        """Конвертирует HypoxiaStatus в строку."""
        if status == HypoxiaStatus.PATHOLOGICAL:
            return 'pathological'
        elif status == HypoxiaStatus.SUSPICIOUS:
            return 'suspicious'
        else:
            return 'normal'

    def _evaluate_baseline_status(self, current_time: float) -> Optional[HypoxiaStatus]:
        """Оценивает статус по базальному ритму."""
        if not self.baseline_history:
            return None

        current_baseline = self.baseline_history[-1]

        # Suspicious: хотя бы раз <110 или >160
        if (current_baseline < self.config.baseline_normal_min or
            current_baseline > self.config.baseline_normal_max):

            # Проверка на pathological: <100 или >180 непрерывно 10 минут
            window_start = current_time - self.config.baseline_path_duration

            # Проверяем, накопилось ли окно
            if self.time_history[0] <= window_start:
                # Окно накопилось, анализируем
                window_mask = np.array(self.time_history) >= window_start
                window_baselines = np.array(self.baseline_history)[window_mask]

                if len(window_baselines) > 0:
                    if (window_baselines < self.config.baseline_path_min).all() or \
                       (window_baselines > self.config.baseline_path_max).all():
                        return HypoxiaStatus.PATHOLOGICAL

            return HypoxiaStatus.SUSPICIOUS

        return HypoxiaStatus.NORMAL

    def _evaluate_vsr_status(self, current_time: float) -> Optional[HypoxiaStatus]:
        """Оценивает статус по ВСР."""
        if not self.vsr_history:
            return None

        current_vsr = self.vsr_history[-1]

        # Suspicious: хотя бы раз <5 или >25
        if (current_vsr < self.config.vsr_normal_min or
            current_vsr > self.config.vsr_normal_max):

            # Pathological: <5 непрерывно >50 мин ИЛИ >25 непрерывно >30 мин
            if current_vsr < self.config.vsr_normal_min:
                window_start = current_time - self.config.vsr_path_low_duration

                if self.time_history[0] <= window_start:
                    window_mask = np.array(self.time_history) >= window_start
                    window_vsr = np.array(self.vsr_history)[window_mask]

                    if len(window_vsr) > 0 and (window_vsr < self.config.vsr_normal_min).all():
                        return HypoxiaStatus.PATHOLOGICAL

            elif current_vsr > self.config.vsr_normal_max:
                window_start = current_time - self.config.vsr_path_high_duration

                if self.time_history[0] <= window_start:
                    window_mask = np.array(self.time_history) >= window_start
                    window_vsr = np.array(self.vsr_history)[window_mask]

                    if len(window_vsr) > 0 and (window_vsr > self.config.vsr_normal_max).all():
                        return HypoxiaStatus.PATHOLOGICAL

            return HypoxiaStatus.SUSPICIOUS

        return HypoxiaStatus.NORMAL

    def _evaluate_deceleration_status(self, current_time: float) -> Optional[HypoxiaStatus]:
        """Оценивает статус по децелерациям."""
        window_start = current_time - self.config.decel_window

        # Проверяем, накопилось ли окно 30 минут
        if self.time_history[0] > window_start:
            return None

        # Находим децелерации в окне 30 минут
        recent_decels = [
            event for event in self.events
            if event[4] == 'decel' and event[0] >= window_start
        ]

        if not recent_decels:
            return HypoxiaStatus.NORMAL

        # Suspicious: хотя бы 1 децелерация за 30 минут
        # Pathological: хотя бы 1 длительностью >3 мин ИЛИ 2 поздние децелерации
        long_decels = [
            event for event in recent_decels
            if (event[1] - event[0]) > self.config.decel_path_duration
        ]

        late_decels = [
            event for event in recent_decels
            if is_late_deceleration(event[0], self.df_uterus)
        ]

        if long_decels or len(late_decels) >= 2:
            return HypoxiaStatus.PATHOLOGICAL

        return HypoxiaStatus.SUSPICIOUS

    def _evaluate_acceleration_status(self, current_time: float) -> Optional[HypoxiaStatus]:
        """Оценивает статус по акселерациям."""
        # Для pathological проверяем окно 40 минут
        window_40_start = current_time - self.config.accel_path_window

        if self.time_history[0] <= window_40_start:
            # Окно 40 минут накопилось
            recent_accels_40 = [
                event for event in self.events
                if event[4] == 'accel' and event[0] >= window_40_start
            ]

            if len(recent_accels_40) == 0:
                return HypoxiaStatus.PATHOLOGICAL

        # Для normal/suspicious проверяем окно 20 минут
        window_20_start = current_time - self.config.accel_normal_window

        if self.time_history[0] > window_20_start:
            return None

        recent_accels_20 = [
            event for event in self.events
            if event[4] == 'accel' and event[0] >= window_20_start
        ]

        if len(recent_accels_20) >= self.config.accel_normal_count:
            return HypoxiaStatus.NORMAL
        else:
            return HypoxiaStatus.SUSPICIOUS

    def should_run_ml_prediction(self, current_time: float) -> bool:
        """
        Проверяет, нужно ли запускать ML прогнозирование.

        Args:
            current_time: Текущее время в секундах

        Returns:
            True если нужно запустить прогноз
        """
        if self.last_ml_prediction_time is None:
            return True

        time_since_last = current_time - self.last_ml_prediction_time
        return time_since_last >= self.config.ml_prediction_interval

    def run_ml_prediction(self, df_bpm: pd.DataFrame, current_time: float) -> Tuple[int, int, int]:
        """
        Запускает ML прогнозирование гипоксии.

        Args:
            df_bpm: DataFrame с данными ЧСС
            current_time: Текущее время

        Returns:
            Кортеж (prob_15, prob_30, prob_60) в процентах
        """
        if self.model_15 is None or self.model_30 is None or self.model_60 is None:
            return 0, 0, 0

        predictions = predict_hypoxia_all(
            df_bpm,
            self.df_uterus,
            current_time,
            self.model_15,
            self.model_30,
            self.model_60
        )

        if predictions is not None:
            self.last_ml_prediction_time = current_time
            # Конвертируем в проценты и округляем
            return (
                round(predictions[0] * 100),
                round(predictions[1] * 100),
                round(predictions[2] * 100)
            )
        else:
            # Недостаточно данных
            return 0, 0, 0

    def analyze(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Выполняет полный анализ данных CTG.

        Args:
            df: DataFrame с колонками 'time_sec' и 'value'

        Returns:
            DataFrame с добавленными колонками анализа
        """
        df = df.sort_values('time_sec').copy()

        # Вычисляем начальный baseline один раз
        if not self.initial_baseline_calculated:
            self.initial_baseline = self.calculate_initial_baseline(df)
            self.initial_baseline_calculated = True

        # Текущие ML прогнозы (обновляются каждую минуту)
        current_pred_15 = 0
        current_pred_30 = 0
        current_pred_60 = 0

        # Построчная обработка
        for idx, row in df.iterrows():
            time = row['time_sec']
            bpm_raw = row['value']

            # Фильтрация артефактов
            bpm = self.filter_artifact(time, bpm_raw)

            # Обновление буферов
            self.update_buffers(time, bpm)

            # Вычисление baseline
            baseline = self.calculate_baseline()
            self.baselines.append(round(baseline))

            # Поиск начала события
            self.check_event_start(time, bpm, baseline, df)

            # Поиск окончания события
            self.check_event_end(time, baseline, df)

            # Вычисление ВСР
            current_vsr = self.calculate_vsr()
            self.vsr_values.append(current_vsr)

            # Флаги событий
            self.acc_flags.append(self.current_event == 'accel')
            self.dec_flags.append(self.current_event == 'decel')

            # Обновление истории для статуса гипоксии
            self.time_history.append(time)
            self.baseline_history.append(round(baseline))
            self.vsr_history.append(current_vsr)

            # Оценка статуса гипоксии
            hypoxia_status, baseline_status, vsr_status, decel_status, accel_status = \
                self.evaluate_hypoxia_status(time)

            self.hypoxia_statuses.append(hypoxia_status)
            self.baseline_statuses.append(baseline_status)
            self.vsr_statuses.append(vsr_status)
            self.decel_statuses.append(decel_status)
            self.accel_statuses.append(accel_status)

            # ML прогнозирование (каждую минуту)
            if self.should_run_ml_prediction(time):
                current_pred_15, current_pred_30, current_pred_60 = \
                    self.run_ml_prediction(df[['time_sec', 'value']], time)

            # Сохраняем текущие прогнозы
            self.ml_predictions_15.append(current_pred_15)
            self.ml_predictions_30.append(current_pred_30)
            self.ml_predictions_60.append(current_pred_60)

        # Добавление результатов в DataFrame
        df = df.rename(columns={'time_sec': 'time', 'value': 'bpm'})
        df['bpm'] = df['bpm'].round()
        df['basal_rhythm'] = np.array(self.baselines)
        df['hrv'] = np.array(self.vsr_values)
        df['acceleration'] = self.acc_flags
        df['deceleration'] = self.dec_flags
        df['hypoxia'] = self.hypoxia_statuses
        df['basal_status'] = self.baseline_statuses
        df['hrv_status'] = self.vsr_statuses
        df['decel_status'] = self.decel_statuses
        df['accel_status'] = self.accel_statuses
        df['hypoxia_15'] = self.ml_predictions_15
        df['hypoxia_30'] = self.ml_predictions_30
        df['hypoxia_60'] = self.ml_predictions_60

        return df


def plot_ctg_analysis(
    df: pd.DataFrame,
    df_uterus: pd.DataFrame,
    events: List[Tuple[float, float, float, float, str]],
    figsize: Tuple[int, int] = (15, 14)
) -> None:
    """
    Визуализирует результаты анализа CTG.

    Args:
        df: DataFrame с результатами анализа
        df_uterus: DataFrame с данными маточных сокращений
        events: Список найденных событий
        figsize: Размер фигуры
    """
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=figsize, sharex=True)

    # График FHR и baseline
    ax1.plot(df['time'], df['bpm'], color='blue', label='FHR', linewidth=1)
    ax1.plot(df['time'], df['basal_rhythm'], color='green',
             label='Basal Rhythm', linewidth=1.5, alpha=0.7)

    # Отрисовка событий
    for start_t, end_t, vmin, vmax, etype in events:
        rect = Rectangle(
            (start_t, vmin - 2),
            end_t - start_t,
            vmax - vmin + 4,
            linewidth=1.5,
            edgecolor='red',
            facecolor='none'
        )
        ax1.add_patch(rect)

    ax1.set_ylabel('FHR (bpm)', fontsize=11)
    ax1.set_title('CTG: Basal Rhythm & Accel/Decel Events', fontsize=12, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # График ВСР
    ax2.plot(df['time'], df['hrv'], color='orange',
             label='HRV (mean amplitude)', linewidth=1)
    ax2.set_ylabel('HRV (bpm)', fontsize=11)
    ax2.legend(loc='best')
    ax2.set_title('FHR Variability', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # График маточных сокращений
    if df_uterus is not None and not df_uterus.empty:
        # Проверяем наличие нужных колонок
        time_col = 'time_sec' if 'time_sec' in df_uterus.columns else df_uterus.columns[0]
        value_col = 'value' if 'value' in df_uterus.columns else df_uterus.columns[1]

        ax3.plot(df_uterus[time_col], df_uterus[value_col],
                color='purple', label='Uterine Contractions', linewidth=1)
        ax3.fill_between(df_uterus[time_col], df_uterus[value_col],
                         alpha=0.3, color='purple')

    ax3.set_ylabel('TOCO', fontsize=11)
    ax3.legend(loc='best')
    ax3.set_title('Uterine Contractions', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # График статуса гипоксии
    status_map = {'normal': 0, 'suspicious': 1, 'pathological': 2}
    status_numeric = df['hypoxia'].map(status_map)

    colors = {'normal': 'green', 'suspicious': 'orange', 'pathological': 'red'}
    for status, color in colors.items():
        mask = df['hypoxia'] == status
        if mask.any():
            ax4.scatter(df.loc[mask, 'time'],
                       status_numeric[mask],
                       c=color, label=status.capitalize(),
                       s=1, alpha=0.6)

    ax4.set_ylabel('Hypoxia Status', fontsize=11)
    ax4.set_yticks([0, 1, 2])
    ax4.set_yticklabels(['Normal', 'Suspicious', 'Pathological'])
    ax4.legend(loc='best')
    ax4.set_title('Hypoxia Status (Rule-based)', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)

    # График ML прогнозов гипоксии
    ax5.plot(df['time'], df['hypoxia_15'], color='green',
             label='15 min prediction', linewidth=1.5, alpha=0.7)
    ax5.plot(df['time'], df['hypoxia_30'], color='orange',
             label='30 min prediction', linewidth=1.5, alpha=0.7)
    ax5.plot(df['time'], df['hypoxia_60'], color='red',
             label='60 min prediction', linewidth=1.5, alpha=0.7)
    ax5.set_ylabel('Hypoxia Risk (%)', fontsize=11)
    ax5.set_xlabel('Time (s)', fontsize=11)
    ax5.legend(loc='best')
    ax5.set_title('ML-based Hypoxia Predictions', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim([0, 100])

    plt.tight_layout()
    plt.show()


def main(
    df_bpm: pd.DataFrame,
    df_uterus: pd.DataFrame,
    model_15_path: str = "model_15.h5",
    model_30_path: str = "model_30.h5",
    model_60_path: str = "model_60.h5"
):
    """
    Основная функция для запуска анализа.

    Args:
        df_bpm: DataFrame с данными ЧСС (колонки 'time_sec' и 'value')
        df_uterus: DataFrame с данными маточных сокращений (опционально)
        model_15_path: Путь к модели прогноза на 15 минут
        model_30_path: Путь к модели прогноза на 30 минут
        model_60_path: Путь к модели прогноза на 60 минут

    Returns:
        DataFrame с результатами анализа
    """
    # Валидация данных
    required_columns = {'time_sec', 'value'}
    if not required_columns.issubset(df_bpm.columns):
        raise ValueError(f"DataFrame должен содержать колонки: {required_columns}")

    # Загрузка моделей прогнозирования
    model_15, model_30, model_60 = load_hypoxia_models(
        model_15_path, model_30_path, model_60_path
    )

    if model_15 is None or model_30 is None or model_60 is None:
        print("Внимание: Модели не загружены, ML прогнозы будут недоступны")

    # Создание анализатора и выполнение анализа
    analyzer = CTGAnalyzer(
        df_uterus=df_uterus,
        model_15=model_15,
        model_30=model_30,
        model_60=model_60
    )
    df_result = analyzer.analyze(df_bpm)

    # # Визуализация
    # plot_ctg_analysis(df_result, df_uterus, analyzer.events)

    # # Вывод результатов
    # print(df_result.head(20))
    # print(df_result.tail(20))


    return df_result


if __name__ == "__main__":
    # Пример использования
    df_bpm = pd.read_csv('2_bpm.csv')
    df_uterus = pd.read_csv('2_uterus.csv')

    df_bpm.columns = ["time_sec", "value"]
    df_uterus.columns = ["time_sec", "value"]

    df_result = main(df_bpm, df_uterus)
