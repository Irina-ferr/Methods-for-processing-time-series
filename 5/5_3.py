# С помощью подпрограммы, созданной при выполнения п. 5.2 оценить спектр
# реализации белого шума из файла “noise.txt” при различных
# значениях ширины окна усреднения. 
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

def daniell_spectrum(signal, window_width, sampling_rate):
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

# Загрузка белого шума из файла
signal = np.loadtxt('noise.txt')

# Параметры
sampling_rate = 1000  # Частота дискретизации
window_widths = [64, 128, 256, 512]  # Ширины окон усреднения

# Визуализация результатов
plt.figure(figsize=(10, 8))

for window_width in window_widths:
    spectrum = daniell_spectrum(signal, window_width, sampling_rate)
    freqs = np.arange(len(spectrum)) * (sampling_rate / len(spectrum))
    
    plt.plot(freqs, spectrum, label=f'Width = {window_width}')

plt.title("Оценка спектра мощности методом Даньелла")
plt.xlabel("Частота (Гц)")
plt.ylabel("Спектр мощности")
plt.legend()
plt.grid()
plt.show()