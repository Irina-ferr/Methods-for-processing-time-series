# Запрограммировать метод оценки спектров Уэлча, оформив алгоритм в виде
# процедуры.
# На входе подпрограммы: массив данных, его длина, ширина окна, используемого
# при расчете периодограмм, сдвиг окна, используемого при расчете периодограмм,
# целочисленный код, определяющий тип оконного преобразования.
# На выходе подпрограммы: массив, содержащий оценку спектра мощности входного
# временного ряда, сделанную методом Уэлча.
# При реализации метода можно использовать подпрограммы, осуществляющие ДПФ DFT
# и оконное преобразование сигнала  Window_func

import math
import numpy as np
import matplotlib.pyplot as plt

def Window_func(signal, N, window_type):
    signal_modded = []
    if window_type == 1:  # Бартлетт
        for i in range(N):
            t_i = (i - N / 2) / N
            window = 1 - 2 * abs(t_i)
            signal_modded.append(signal[i] * window)
            
    elif window_type == 2:  # Хэмминг
        for i in range(N):
            window = 0.54 - 0.46 * math.cos(2 * math.pi * i / (N - 1))
            signal_modded.append(signal[i] * window)
            
    elif window_type == 3:  # Прямоугольное
        signal_modded = signal[:N]  # просто копируем сигнал
        
    else:
        raise ValueError('Не выбрано окно')

    return np.array(signal_modded)

def DFT(signal):
    N = len(signal)
    ah = np.zeros(N // 2 + 1)  # массив для коэффициентов a_h
    bh = np.zeros(N // 2 + 1)  # массив для коэффициентов b_h

    for h in range(N // 2 + 1):
        sum_sig1 = np.sum(signal * np.cos(2 * np.pi * h * np.arange(N) / N))
        sum_sig2 = np.sum(signal * np.sin(2 * np.pi * h * np.arange(N) / N))

        ah[h] = (2 / N) * sum_sig1
        bh[h] = (-2 / N) * sum_sig2

    return ah, bh

def Welch(signal, signal_length, window_width, window_shift, window_type):
    # определяем количество окон
    num_windows = (signal_length - window_width) // window_shift + 1
    spectrum_power = np.zeros(window_width // 2 + 1)  # массив для хранения оценок спектра мощности

    for i in range(num_windows):
        start_index = i * window_shift
        end_index = start_index + window_width
        
        # проверка выхода за границы массива
        if end_index > signal_length:
            break
        
        # извлечение текущего окна
        current_window = signal[start_index:end_index]
        
        # применение оконной функции
        windowed_signal = Window_func(current_window, window_width, window_type)
        
        # ДПФ для текущего окна
        ah, bh = DFT(windowed_signal)  
        
        # вычисление спектра мощности для текущего окна
        power_spectrum = ah**2 + bh**2
        
        # суммируем спектры мощности
        spectrum_power += power_spectrum

    # среднее значение спектров мощности по всем окнам
    spectrum_power /= num_windows
    
    return spectrum_power

# генерация тестового сигнала
fs = 1000  # частота дискретизации
T = 1.0    # время сигнала в секундах
t = np.linspace(0, T, int(fs * T), endpoint=False)  
signal = np.sin(2 * np.pi * 50 * t) + np.sin(2 * np.pi * 120 * t)  # сигнал с двумя частотами

# параметры для Уэлча
window_width = 256
window_shift = 128
window_type = 2  # Хэмминг

spectrum_power = Welch(signal, len(signal), window_width, window_shift, window_type)

# Графики
freqs = np.linspace(0, fs / 2, len(spectrum_power))
plt.plot(freqs, spectrum_power)
plt.title('Оценка спектра мощности методом Уэлча')
plt.xlabel('Частота (Гц)')
plt.ylabel('Спектр мощности')
plt.grid()
plt.show()
