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
def DFT(signal,sampling_rate):
    N=len(signal)
    if (N%2==1):
        M=round(N/2)+1
    else:
        M=N/2
    M=int(M)    
    ah = np.zeros(M)  # массив для коэффициентов a_h
    bh = np.zeros(M)  # массив для коэффициентов b_h

    for h in range(0, M):
        sum_sig1 = 0.0
        sum_sig2 = 0.0
        fh = h/N*sampling_rate  # частота
        for i in range(N):
            
            sum_sig1 += signal[i] * np.cos(2 * np.pi * fh * (i / sampling_rate))
            sum_sig2 += signal[i] * np.sin(2 * np.pi * fh * (i / sampling_rate))

        ah[h] = (2 / N) * sum_sig1
        bh[h] = (-2 / N) * sum_sig2

    return ah, bh

# параметры
fs = 100  # частота дискретизации (Гц)
duration = 2  # длительность сигнала (сек)
t = np.linspace(0, duration, int(fs * duration))  # временные метки

# генерация синусоидальных сигналов
amplitude1 = 1.2
frequency1 = 10  # частота первого сигнала (Гц)
signal1 = amplitude1 * np.sin(2 * np.pi * frequency1 * t[:-1])

amplitude2 = 2
frequency2 = 20  # частота второго сигнала (Гц)
signal2 = amplitude2 * np.sin(2 * np.pi * frequency2 * t[:-1])

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
plt.plot(t[:-1], combined_signal)
plt.title('Сумма синусоидальных сигналов')
plt.xlabel('Время (с)')
plt.ylabel('Амплитуда')
plt.grid()

# график амплитудного спектра
frequencies = np.linspace(0, fs / 2, len(amplitude_spectrum))  # частоты для оси x
plt.subplot(2, 1, 2)
plt.stem(frequencies, amplitude_spectrum, markerfmt='' )
plt.title('Амплитудный спектр')
plt.xlabel('Частота (Гц)')
plt.ylabel('Амплитуда')
plt.xlim(0, fs / 2)  # ограничиваем ось x до половины частоты дискретизации
plt.grid()

plt.tight_layout()
plt.show()
