import numpy as np
import matplotlib.pyplot as plt
import math as m
# расчет импульсной характеристики
def calc_impulse_responce(numerator_coefficients,denominator_coefficients, lenght):
    for n in range(lenght):  # 1000 значений
        if n == 0:
            impulse_response.append(numerator_coefficients[0] / denominator_coefficients[0])
        else:
            previous_sum = 0
            for k in range(1, n + 1):
                if k < len(denominator_coefficients):
                    previous_sum += impulse_response[n - k] * denominator_coefficients[k]
            impulse_response.append((1 / denominator_coefficients[0]) * (numerator_coefficients[n] - previous_sum))
    return impulse_response

# АЧХ 
def DFT(signal):
    num_samples = len(signal)
    ah = np.zeros(int(num_samples / 2))
    bh = np.zeros(int(num_samples / 2))

    for freq_index in range(0, int(num_samples / 2)):
        for sample_index in range(num_samples):
            ah[freq_index] += (signal[sample_index] * np.cos(2 * np.pi * sample_index * freq_index / num_samples))
            bh[freq_index] -= (signal[sample_index] * np.sin(2 * np.pi * sample_index * freq_index / num_samples))
    
    return ah, bh

def calc_amplitude_response(impulse_response):
    num_samples = len(impulse_response)
    amplitude_spectrum = []
    ah, bh = DFT(impulse_response)

    for freq_index in range(num_samples // 2):
        amplitude = m.sqrt((ah[freq_index] * (2 / num_samples)) ** 2 + (bh[freq_index] * (2 / num_samples)) ** 2)
        amplitude_spectrum.append(amplitude)

    return amplitude_spectrum

# коэффициенты фильтра
numerator_coefficients = [0.01,0.03,0.09,0.2,0.2,0.2,0.2,0.09,0.03,0.01]+[0]*1000  # коэффициенты числитель
denominator_coefficients = [1]+[0]*1000  # коэффициенты знаменатель


impulse_response = []  # импульсная характеристика

impulse_response=calc_impulse_responce(numerator_coefficients,denominator_coefficients, lenght=1000)

# импульсная график
plt.subplot(1, 2, 1)
plt.plot(np.arange(1000), impulse_response)
plt.title('Импульсная характеристика')
# plt.xlabel('Номер выборки')
plt.ylabel('Амплитуда')
plt.grid()
plt.tight_layout()
print(impulse_response)
# АЧХ рассчет
amplitude_response = calc_amplitude_response(impulse_response)
frequencies = np.linspace(0, 0.5, len(amplitude_response))

# АЧХ график
plt.subplot(1, 2, 2)
plt.plot(frequencies, amplitude_response)
plt.title('АЧХ фильтра')
plt.xlabel('Частота')
plt.ylabel('Амплитуда')
plt.grid()
plt.tight_layout()

plt.show()