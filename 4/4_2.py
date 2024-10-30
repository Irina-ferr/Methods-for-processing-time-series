import numpy as np
import matplotlib.pyplot as plt
import math

# функция для генерации гармонического сигнала
def generate_signal(N, A, T):
    n = np.arange(N)
    return A * np.cos(2 * np.pi * n / T)

# функция для применения окон
def Window_func(signal, N, window_type):
    signal_modded = []
    match window_type:
        case 1:  # Бартлетт
            for i in range(N):
                t_i = (i - N / 2) / N
                window = 1 - 2 * abs(t_i)
                signal_modded.append(signal[i] * window)
            return signal_modded

        case 2:  # Хэмминг
            for i in range(N):
                t_i = (i - N / 2) / N
                window = 0.46 * math.cos(2 * math.pi * t_i) + 0.54
                signal_modded.append(signal[i] * window)
            return signal_modded

        case 3:  # Прямоугольное
            window = [0] * N
            for i in range(N):
                if (i != 0) and (i != N - 1):
                    window[i] = 1
            signal_modded = [signal[i] * window[i] for i in range(N)]
            return signal_modded

        case _:
            return print('Не выбрано окно')

# Функция DFT
def DFT(signal, sampling_rate):
    N = len(signal)
    if N % 2 == 1:
        M = round(N / 2) + 1
    else:
        M = N // 2
    M = int(M)

    ah = np.zeros(M)  # массив для коэффициентов a_h
    bh = np.zeros(M)  # массив для коэффициентов b_h

    for h in range(M):
        sum_sig1 = 0.0
        sum_sig2 = 0.0
        fh = h / N * sampling_rate  # частота
        for i in range(N):
            sum_sig1 += signal[i] * np.cos(2 * np.pi * fh * (i / sampling_rate))
            sum_sig2 += signal[i] * np.sin(2 * np.pi * fh * (i / sampling_rate))

        ah[h] = (2 / N) * sum_sig1
        bh[h] = (-2 / N) * sum_sig2

    return ah, bh

# Параметры сигнала
A = 1          # амплитуда
T = 30         # период в дискретных отсчетах
N_values = [3000, 2999, 2992]  # длины реализаций
sampling_rate = 1  # Частота дискретизации

# Построение периодограмм для каждой длины и типа окна
# plt.figure(figsize=(15, 15))

for j, N in enumerate(N_values):
    signal = generate_signal(N, A, T)

    # Прямоугольное окно
    plt.subplot(3, 3, 3 * j + 1)
    windowed_signal = Window_func(signal, N, 3)  # Прямоугольное окно
    ah, bh = DFT(windowed_signal, sampling_rate)
    power_spectrum = ah**2 + bh**2
    plt.plot(power_spectrum, color='blue', label=f'N={N}, Прямоугольное')
    plt.title(f'Периодограмма (N={N}, Прямоугольное)')
    plt.xlabel('Частота')
    plt.ylabel('Плотность спектра мощности')
    plt.grid()

    # Окно Бартлетта
    plt.subplot(3, 3, 3 * j + 2)
    windowed_signal = Window_func(signal, N, 1)  # Окно Бартлетта
    ah, bh = DFT(windowed_signal, sampling_rate)
    power_spectrum = ah**2 + bh**2
    plt.plot(power_spectrum, color='green', label=f'N={N}, Бартлетта')
    plt.title(f'Периодограмма (N={N}, Бартлетта)')
    plt.xlabel('Частота')
    plt.ylabel('Плотность спектра мощности')
    plt.grid()

    # Окно Хэмминга
    plt.subplot(3, 3, 3 * j + 3)
    windowed_signal = Window_func(signal, N, 2)  # Окно Хэмминга
    ah, bh = DFT(windowed_signal, sampling_rate)
    power_spectrum = ah**2 + bh**2
    plt.plot(power_spectrum, color='red', label=f'N={N}, Хэмминга')
    plt.title(f'Периодограмма (N={N}, Хэмминга)')
    plt.xlabel('Частота')
    plt.ylabel('Плотность спектра мощности')
    plt.grid()

plt.tight_layout()
plt.show()