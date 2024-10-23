# Сгенерировать временной ряд из 5 периодов синусоиды с частотой 10 отсчетов на
# период. Отобразить реализацию графически. Используя созданную в п. 3.1
# подпрограмму, построить и отобразить на графике амплитудный спектр созданной
# реализации
import numpy as np
import matplotlib.pyplot as plt

# функция для вычисления дискретного преобразования Фурье
def DFT(signal, N):
    ah = np.zeros(N // 2 + 1)  # массив для коэффициентов a_h
    bh = np.zeros(N // 2 + 1)  # массив для коэффициентов b_h

    for h in range(0, N // 2 + 1):
        sum_sig1 = 0.0
        sum_sig2 = 0.0
        for i in range(N):
            dt = (i + 1) * (1 / sampling_rate)  # шаг времени
            fh = h / (N * dt)   # частота

            sum_sig1 += signal[i] * np.cos(2 * np.pi * fh * (i / sampling_rate))
            sum_sig2 += signal[i] * np.sin(2 * np.pi * fh * (i / sampling_rate))

        ah[h] = (2 / N) * sum_sig1
        bh[h] = (-2 / N) * sum_sig2

    return ah, bh

# параметры синусоиды
frequency = 1  # частота синусоиды (1 Гц)
sampling_rate = 10  # количество отсчетов на период
periods = 5  # количество периодов

# генерация временного ряда
N = periods * sampling_rate  # общее количество отсчетов
t = np.linspace(0, periods * (1 / frequency), N)  # временные метки
signal = np.sin(2 * np.pi * frequency * t)  # синусоидальный сигнал

# вычисление коэффициентов ДПФ
a_h, b_h = DFT(signal, N)

# вычисление амплитудного спектра
amplitude_spectrum = np.sqrt(a_h**2 + b_h**2)

# построение графиков
plt.figure(figsize=(12, 6))

# график временного ряда
plt.subplot(2, 1, 1)
plt.plot(t, signal)
plt.title('Синусоидальный сигнал')
plt.xlabel('Время (с)')
plt.ylabel('Амплитуда')
plt.grid()

# график амплитудного спектра
frequencies = np.linspace(0, sampling_rate / 2, len(amplitude_spectrum))  # частоты для оси x
plt.subplot(2, 1, 2)
plt.stem(frequencies, amplitude_spectrum)
plt.title('Амплитудный спектр')
plt.xlabel('Частота (Гц)')
plt.ylabel('Амплитуда')
plt.grid()

plt.tight_layout()
plt.show()
