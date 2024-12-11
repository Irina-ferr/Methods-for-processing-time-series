# С помощью КИХ ФНЧ, коэффициенты которого хранятся в текстовом файле
# 'FIR_LPF_Fco=1.5kHz_Fsamp=8kHz.txt', очистить от
# высокочастотных шумов реализацию {x_i} из 'xi_9.txt'”'.
# Частота дискретизации сигнала 8 кГц. Отобразить исходную – {x_i}  и
# отфильтрованную – {y_i} реализации на одном графике. Оценить и отобразить
# графически в логарифмическом масштабе спектры мощности реализаций {x_i} , {y_i} i y .
# Рассчитать и отобразить на графиках в логарифмическом и линейном масштабах
# АЧХ используемого фильтра.
import numpy as np
import matplotlib.pyplot as plt
import math

def DFT(signal):
    num_samples = len(signal)
    ah = np.zeros(int(num_samples / 2))
    bh = np.zeros(int(num_samples / 2))

    for freq_index in range(0, int(num_samples / 2)):
        for sample_index in range(num_samples):
            ah[freq_index] += (signal[sample_index] * np.cos(2 * np.pi * sample_index * freq_index / num_samples))
            bh[freq_index] -= (signal[sample_index] * np.sin(2 * np.pi * sample_index * freq_index / num_samples))
    
    return ah, bh

# для спектрра частот
def compute_spectrum(data):
    N = len(data)
    spectrum_magnitude = []
    A_coeffs, B_coeffs = DFT(data)  # Использование функции DFT
    for k in range(N // 2):
        spectrum_magnitude.append(np.sqrt((A_coeffs[k] * (2 / N)) ** 2 + (B_coeffs[k] * (2 / N)) ** 2))
    return spectrum_magnitude
def convolution(signal, signal_len, impulse_response, impulse_response_len, index):
    if index < 0 or index >= signal_len + impulse_response_len - 1:
        raise ValueError("Индекс выходит за пределы допустимого диапазона")
    result = 0
    for j in range(impulse_response_len):
        if 0 <= index - j < signal_len:  # Проверка границ
            result += signal[index - j] * impulse_response[j]
    return result

signal_xi = np.loadtxt('xi_9.txt')  # сигнал
impulse_response_hi = np.loadtxt('FIR_LPF_Fco=1.5kHz_Fsamp=8kHz.txt')  # коэфф фильтра

# длина сигнала
len_signal_xi = len(signal_xi)
len_impulse_response_hi = len(impulse_response_hi)

# свертка
convolved_output_yi = []
for i in range(len_signal_xi + len_impulse_response_hi - 1):
    convolved_output_yi.append(convolution(signal_xi, len_signal_xi, impulse_response_hi, len_impulse_response_hi, i))

# графики
plt.subplot(1, 3, 1)
plt.plot(np.arange(len_signal_xi), signal_xi, color="mediumseagreen")
plt.title('Сигнал x_i')
plt.ylabel('Амплитуда')
plt.grid()

plt.subplot(1, 3, 2)
plt.plot(np.arange(len_impulse_response_hi), impulse_response_hi, color="aquamarine")
plt.title('Импульсная характеристика h_i')
plt.ylabel('Амплитуда')
plt.grid()

plt.subplot(1, 3, 3)
plt.plot(np.arange(len(convolved_output_yi)), convolved_output_yi, color="darkviolet")
plt.title('Результат свертки y_i')
plt.ylabel('Амплитуда')
plt.grid()

plt.show()




# частота дискретизации
fs = 8000

# спектры
spectrum_hi = compute_spectrum(impulse_response_hi)
spectrum_xi = compute_spectrum(signal_xi)
spectrum_yi = compute_spectrum(convolved_output_yi)

# оси частот
freq_bins_xi = np.arange(len(spectrum_xi)) * (fs / (len(signal_xi) * 2))
freq_bins_yi = np.arange(len(spectrum_yi)) * (fs / (len(convolved_output_yi) * 2))
freq_bins_hi = np.arange(len(spectrum_hi)) * (fs / (len(impulse_response_hi) * 2))

# графики спектров
plt.subplot(1, 4, 1)
plt.plot(freq_bins_xi, [10 * math.log10(p / max(spectrum_xi)) for p in spectrum_xi], color="darkviolet")
plt.title('Лог спектр мощности x_i')
plt.xlabel('Частота (Гц)')
plt.ylabel('Амплитуда (дБ)')
plt.grid()

plt.subplot(1, 4, 2)
plt.plot(freq_bins_yi, [10 * math.log10(p / max(spectrum_yi)) for p in spectrum_yi], color="darkviolet")
plt.title('Лог спектр мощности y_i')
plt.xlabel('Частота (Гц)')
plt.ylabel('Амплитуда (дБ)')
plt.grid()

plt.subplot(1, 4, 3)
plt.plot(freq_bins_hi, spectrum_hi, color="darkviolet")
plt.title('Амплитудная характеристика h_i')
plt.xlabel('Частота (Гц)')
plt.ylabel('Амплитуда')
plt.grid()

plt.subplot(1, 4, 4)
plt.plot(freq_bins_hi, [10 * math.log10(p / max(spectrum_hi)) for p in spectrum_hi], color="darkviolet")
plt.title('Лог амплитудная характеристика h_i')
plt.xlabel('Частота (Гц)')
plt.ylabel('Амплитуда (дБ)')
plt.grid()

plt.show()
