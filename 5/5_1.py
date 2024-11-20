# Рассчитать периодограмму реализации из файла “lfp.txt”
# (Fsamp=1 кГц) и построить графики функций спектральной плотности в линейном и
# логарифмическом масштабах.

import numpy as np
import matplotlib.pyplot as plt
import math

# Функция DFT
def DFT(signal, sampling_rate):
    print("*")
    N = len(signal)
    if N % 2 == 1:
        M = round(N / 2) + 1
    else:
        M = N // 2
    M = int(M)
    print("*")
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

# Загрузка данных из файла
file_path = 'lfp.txt'
data = np.loadtxt('lfp.txt')
q=len(data)
q=q//4
data = data [:q]
# Параметры
sampling_rate = 1000  # Частота дискретизации

# Применяем DFT
ah, bh = DFT(data, sampling_rate)

# Расчет спектральной плотности
power_spectrum = np.sqrt(ah**2 + bh**2)

# Применяем логарифмическое преобразование к спектральной плотности
# Добавляем небольшую величину для избегания логарифма от нуля

power_spectrum_log = 20 * np.log10(power_spectrum/np.max(power_spectrum) + 1e-10)

# Частоты
freqs = np.arange(len(power_spectrum)) * (sampling_rate / len(data))

# Построение графиков

# Линейный масштаб
plt.subplot(1, 2, 1)
plt.plot(freqs[:len(power_spectrum)], power_spectrum)
plt.title('Спектральная плотность (линейный масштаб)')
plt.xlabel('Частота (Гц)')
plt.ylabel('Спектральная плотность')
plt.grid()

# Логарифмический масштаб
plt.subplot(1, 2, 2)
plt.plot(freqs[:len(power_spectrum_log)], power_spectrum_log)
plt.title('Спектральная плотность (логарифмический масштаб)')
plt.xlabel('Частота (Гц)')
plt.ylabel('Спектральная плотность (дБ)')
plt.grid()

plt.tight_layout()
plt.show()
