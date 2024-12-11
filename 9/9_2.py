# Используя рассчитанный в лекции 9 БИХ-фильтр (9.9) очистить от 50 Гц сетевой
# наводки временной ряд электрокардиограммы {x_i} из
# 'ECG_50Hz.txt'. Частота дискретизации ряда 500 Гц.
# Отобразить исходную – {x_i} и отфильтрованную – {y_i} реализации на одном
# графике. Рассчитать и отобразить графически в логарифмическом масштабе спектры
# мощности реализаций {x_i}, {y_i}. Рассчитать и отобразить на графике в
# логарифмическом и линейном масштабах АЧХ используемого фильтр
import numpy as np
import matplotlib.pyplot as plt
import math

def DFT(signal):
    num_samples = len(signal)
    ah = np.zeros(int(num_samples / 2))
    bh = np.zeros(int(num_samples / 2))

    for freq_index in range(int(num_samples / 2)):
        for sample_index in range(num_samples):
            ah[freq_index] += signal[sample_index] * np.cos(2 * np.pi * sample_index * freq_index / num_samples)
            bh[freq_index] -= signal[sample_index] * np.sin(2 * np.pi * sample_index * freq_index / num_samples)

    return ah, bh

def compute_spectrum(signal):
    Ah, Bh = DFT(signal)
    spectrum = np.sqrt((Ah * (2 / len(signal)))**2 + (Bh * (2 / len(signal)))**2)
    return spectrum

# данные
ecg_signal = np.loadtxt('ECG_50Hz.txt')
ecg_signal=ecg_signal[:2000]
filtered_signal = []

# фильтрация сигнала
for index in range(len(ecg_signal)):
    if index == 0:
        filtered_signal.append(ecg_signal[0])
    elif index == 1:
        filtered_signal.append(ecg_signal[1] - 1.61804 * ecg_signal[0] + 1.51638 * filtered_signal[0])
    elif index == 2:
        filtered_signal.append(ecg_signal[2] - 1.61804 * ecg_signal[1] + ecg_signal[0] + 1.51638 * filtered_signal[1] - 0.87830 * filtered_signal[0])
    else:
        filtered_signal.append(ecg_signal[index] - 1.61804 * ecg_signal[index - 1] + ecg_signal[index - 2] + 1.51638 * filtered_signal[index - 1] - 0.87830 * filtered_signal[index - 2])

# импульсная характеристика
bh = [1, -1.61804, 1] + [0] * 1000
ah = [1, -1.51638, 0.87830] + [0] * 1000
impulse_response = []
for i in range(1000):
    if i == 0:
        impulse_response.append(bh[0] / ah[0])
    else:
        p = sum(impulse_response[i - k] * ah[k] for k in range(1, i + 1))
        impulse_response.append((1 / ah[0]) * (bh[i] - p))

# график сигнала
plt.subplot(1, 3, 1)
plt.plot(np.arange(len(ecg_signal)), ecg_signal)
plt.title('Сигнал x_i')
plt.xlabel('Индекс')
plt.ylabel('Амплитуда')
plt.grid()

plt.subplot(1, 3, 2)
plt.plot(np.arange(len(impulse_response)), impulse_response)
plt.title('Импульсная характеристика h_i')
plt.xlabel('Индекс')
plt.ylabel('Амплитуда')
plt.grid()

plt.subplot(1, 3, 3)
plt.plot(np.arange(len(filtered_signal)), filtered_signal)
plt.title('Свёртка y_i')
plt.xlabel('Индекс')
plt.ylabel('Амплитуда')
plt.grid()

plt.show()

# частотный спектр
sampling_frequency = 500
frequency = np.arange(0, sampling_frequency / 2, sampling_frequency / len(ecg_signal))

spectrum_ecg = compute_spectrum(ecg_signal)
spectrum_filtered = compute_spectrum(filtered_signal)
spectrum_impulse = compute_spectrum(impulse_response)

# графики спектров
plt.subplot(1, 4, 1)
plt.plot(frequency, [10 * math.log10(power / max(spectrum_ecg)) for power in spectrum_ecg])
plt.title('Спектр мощности логарифмический x_i')
plt.xlabel('Частота (Гц)')
plt.ylabel('Мощность (дБ)')
plt.grid()

plt.subplot(1, 4, 2)
plt.plot(frequency, [10 * math.log10(power / max(spectrum_filtered)) for power in spectrum_filtered])
plt.title('Спектр мощности логарифмический y_i')
plt.xlabel('Частота (Гц)')
plt.ylabel('Мощность (дБ)')
plt.grid()

plt.subplot(1, 4, 3)
plt.plot(frequency[:500], spectrum_impulse)
plt.title('АЧХ h_i')
plt.xlabel('Частота (Гц)')
plt.ylabel('Амплитуда')
plt.grid()

plt.subplot(1, 4, 4)
plt.plot(frequency[:500], [10 * math.log10(power / max(spectrum_impulse)) for power in spectrum_impulse])
plt.title('АЧХ логарифмический h_i')
plt.xlabel('Частота (Гц)')
plt.ylabel('Амплитуда (дБ)')
plt.grid()

plt.show()
