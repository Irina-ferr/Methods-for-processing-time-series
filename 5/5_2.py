# Запрограммировать метод оценки спектров Даньелла, оформив алгоритм в виде
# процедуры.
# На входе подпрограммы: массив данных, его длина, ширина окна усреднения в
# частотной области.
# На выходе подпрограммы: массив, содержащий оценку спектра мощности входного
# временного ряда, сделанную методом Даньелла.
# При реализации метода можно использовать подпрограмму, осуществляющую ДПФ,
# созданную при выполнении п. 3.1.
import numpy as np
import matplotlib.pyplot as plt

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

def daniell(signal, window_width, sampling_rate):
    N = len(signal)
    padded_signal = np.pad(signal, (0, max(0, window_width - N % window_width)), 'constant')
    num_windows = len(padded_signal) // window_width
    spectrum = np.zeros(window_width // 2)  # Инициализируем массив для спектра

    for i in range(num_windows):
        windowed_signal = padded_signal[i * window_width: (i + 1) * window_width]
        ah, bh = DFT(windowed_signal, sampling_rate)
        power_spectrum = ah**2 + bh**2  # Оценка спектра мощности
        spectrum += power_spectrum

    # Усредняем спектр по всем окнам
    spectrum /= num_windows

    return spectrum

# Пример использования

# Генерируем тестовый сигнал (синусоида с шумом)
fs = 1000  # Частота дискретизации
t = np.arange(0, 1, 1/fs)
signal = 0.5 * np.sin(2 * np.pi * 50 * t) + 0.1 * np.random.randn(len(t))  # 50 Гц с шумом

# Параметры
window_width = 256  # Ширина окна усреднения

# Рассчитываем спектр по методу Даньелла
spectrum = daniell(signal, window_width, fs)

# Визуализация спектра
freqs = np.arange(window_width // 2) * (fs / window_width)

plt.plot(freqs, spectrum)
plt.title("Оценка спектра мощности методом Даньелла")
plt.xlabel("Частота (Гц)")
plt.ylabel("Спектр мощности")
plt.grid()
plt.show()