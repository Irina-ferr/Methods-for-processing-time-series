# Запрограммировать обратное ДПФ, оформив его в виде процедуры.
# На входе подпрограммы: глобальные массивы коэффициентов
# ah и bh: Ah и Bh, целое M – количество гармоник.
# На выходе подпрограммы: массив, содержащий временной ряд.
import numpy as np
import matplotlib.pyplot as plt


def re_DFT(a_h, b_h, M, N):
    """
    вычисляет временной ряд на основе коэффициентов a_h и b_h.
    
    на вход: 
        a_h: массив коэффициентов a_h (косинусные компоненты).
        b_h: массив коэффициентов b_h (синусные компоненты).
        M: количество гармоник.
        N: количество точек времени.

    на выход:
        signal: массив, содержащий временной ряд.
    """
    signal = np.zeros(N)
    
    for n in range(N):
        for k in range(M):
            signal[n] += a_h[k] * np.cos(2 * np.pi * k * n / N) + b_h[k] * np.sin(2 * np.pi * k * n / N)
    
    return signal

# параметры сигнала
N = 100  # количество точек времени
M = N // 2 + 1 if N % 2 != 0 else N // 2  # количество гармоник

# пример коэффициентов 
a_h = np.array([1, 0.5, 0.25])  # пример амплитуд косинусов
b_h = np.array([0, 0.5, 0.25])   # пример амплитуд синусов

# убедитесь, что длина a_h и b_h соответствует M
a_h = np.pad(a_h, (0, M - len(a_h)), 'constant')
b_h = np.pad(b_h, (0, M - len(b_h)), 'constant')

# получение временного ряда
signal = re_DFT(a_h, b_h, M, N)

# визуализация временного ряда
plt.plot(signal)
plt.title('временной ряд из обратного дпф')
plt.xlabel('время')
plt.ylabel('амплитуда')
plt.grid()
plt.show()

