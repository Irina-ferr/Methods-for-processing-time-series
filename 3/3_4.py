# Смоделировать временной ряд длительностью 2 секунды суммы синусоидального
# сигнала с амплитудой 1.2 и частотой 10 Гц и синусоидального сигнала с амплитудой
# 2 и частотой 20 Гц при частоте дискретизации 100 Гц. Отобразить созданный
# временной ряд графически. Используя созданную в п. 3.1 подпрограмму, построить и
# отобразить на графике амплитудный спектр созданной реализации. Значения на оси
# частот графика амплитудного спектра отображать в герцах.
import numpy as np
import matplotlib.pyplot as plt
import math

# функция для вычисления дискретного преобразования Фурье
def DFT(signal, N):
    ah = np.zeros(N // 2 + 1)  # массив для коэффициентов a_h
    bh = np.zeros(N // 2 + 1)  # массив для коэффициентов b_h

    for h in range(0, N // 2 + 1):
        sum_sig1 = 0.0
        sum_sig2 = 0.0
        for i in range(N):
            dt = (i + 1) / fs  # шаг времени
            fh = h / (N * dt)   # частота

            sum_sig1 += signal[i] * math.cos(2 * math.pi * fh * (i / fs))
            sum_sig2 += signal[i] * math.sin(2 * math.pi * fh * (i / fs))

        ah[h] = (2 / N) * sum_sig1
        bh[h] = (-2 / N) * sum_sig2

    return ah, bh

# параметры
fs = 100  # частота дискретизации (Гц)
duration = 2  # длительность сигнала (сек)
t = np.linspace(0, duration, int(fs * duration), endpoint=False)  # временные метки

# генерация синусоидальных сигналов
amplitude1 = 1.2
frequency1 = 10  # частота первого сигнала (Гц)
signal1 = amplitude1 * np.sin(2 * np.pi * frequency1 * t)

amplitude2 = 2
frequency2 = 20  # частота второго сигнала (Гц)
signal2 = amplitude2 * np.sin(2 * np.pi * frequency2 * t)

# сумма сигналов
combined_signal = signal1 + signal2

# вычисление коэффициентов ДПФ
N = len(combined_signal)
a_h, b_h = DFT(combined_signal, N)

# вычисление амплитудного спектра
amplitude_spectrum = np.sqrt(a_h**2 + b_h**2)

# построение графиков
plt.figure(figsize=(12, 6))

# график временного ряда
plt.subplot(2, 1, 1)
plt.plot(t, combined_signal)
plt.title('Сумма синусоидальных сигналов')
plt.xlabel('Время (с)')
plt.ylabel('Амплитуда')
plt.grid()

# график амплитудного спектра
frequencies = np.linspace(0, fs / 2, len(amplitude_spectrum))  # частоты для оси x
plt.subplot(2, 1, 2)
plt.stem(frequencies, amplitude_spectrum)
plt.title('Амплитудный спектр')
plt.xlabel('Частота (Гц)')
plt.ylabel('Амплитуда')
plt.xlim(0, fs / 2)  # ограничиваем ось x до половины частоты дискретизации
plt.grid()

plt.tight_layout()
plt.show()
