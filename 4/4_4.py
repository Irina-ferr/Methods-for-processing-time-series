# С помощью подпрограммы построить и
# сопоставить периодограммы гармонического сигнала при использовании прямоугольного окна, а также окон Бартлетта
# и Хэмминга для N_1=3000 и N_3=2992. Привести соответствующие графики, сделать
# выводы.
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
        case 1: # Бартлетт
            for i in range(N):
                t_i = (i - N / 2) / N
                window = 1 - 2 * abs(t_i)
                signal_modded.append(signal[i] * window)
            return signal_modded

        case 2: # Хэмминг
            for i in range(N):
                t_i = (i - N / 2) / N
                window = 0.46 * math.cos(2 * math.pi * t_i) + 0.54
                signal_modded.append(signal[i] * window)
            return signal_modded

        case 3: # Прямоугольное
            window = [0] * N
            for i in range(N):
                if (i != 0) and (i != N - 1):
                    window[i] = 1
            signal_modded = [signal[i] * window[i] for i in range(N)]
            return signal_modded

        case _:
            return print('Не выбрано окно')

# параметры сигнала
A = 1          # амплитуда
T = 30         # период в дискретных отсчетах
N_values = [3000, 2999, 2992] # длины реализаций

# построение периодограмм для каждой длины и типа окна
plt.figure(figsize=(15, 15))

for j, N in enumerate(N_values):
    signal = generate_signal(N, A, T)

    # Прямоугольное окно
    plt.subplot(3, 3, 3 * j + 1)
    windowed_signal = Window_func(signal, N, 3) # Прямоугольное окно
    plt.psd(windowed_signal, 256, 1, color='blue', label=f'N={N}, Прямоугольное')
    plt.title(f'Периодограмма (N={N}, Прямоугольное)')
    plt.xlabel('Частота')
    plt.ylabel('Плотность спектра мощности')
    plt.grid()

    # Окно Бартлетта
    plt.subplot(3, 3, 3 * j + 2)
    windowed_signal = Window_func(signal, N, 1) # Окно Бартлетта
    plt.psd(windowed_signal, 256, 1, color='green', label=f'N={N}, Бартлетта')
    plt.title(f'Периодограмма (N={N}, Бартлетта)')
    plt.xlabel('Частота')
    plt.ylabel('Плотность спектра мощности')
    plt.grid()

    # Окно Хэмминга
    plt.subplot(3, 3, 3 * j + 3)
    windowed_signal = Window_func(signal, N, 2) # Окно Хэмминга
    plt.psd(windowed_signal, 256, 1, color='red', label=f'N={N}, Хэмминга')
    plt.title(f'Периодограмма (N={N}, Хэмминга)')
    plt.xlabel('Частота')
    plt.ylabel('Плотность спектра мощности')
    plt.grid()

plt.tight_layout()
plt.show()
