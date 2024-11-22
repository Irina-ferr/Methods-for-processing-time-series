# Запрограммировать генерацию временной реализации гармонического сигнала с
# периодом 30 дискретных отсчетов длиной 3000 отсчетов. Сопоставить оценки
# спектра мощности этой реализации, сделанные с помощью построения
# периодограммы и полученные при использовании метода Даньелла. 
import numpy as np
import matplotlib.pyplot as plt

# параметры сигнала
sampling_rate = 100  # Частота дискретизации
period = 30          # Период в дискретных отсчетах
signal_length = 3000  # Длина сигнала
frequency = sampling_rate / period  # Частота

# генерация гармонического сигнала
t = np.arange(signal_length)
A = 1  # амплитуда
signal = A * np.cos(2 * np.pi * frequency * t / sampling_rate)

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
    ah, bh = DFT(signal, sampling_rate)
    power_spectrum = ah**2 + bh**2
    N=len(power_spectrum)
    spectrum=[]
    for i in range (N-1):
        if((i<window_width) or(i>(N-window_width-1))):
            spectrum.append(power_spectrum[i])
            
        else:
            window=power_spectrum[i-(window_width//2):i+(window_width//2)+1]
            daniell_sp=sum(window)/len(window)
            spectrum.append(daniell_sp)
    return spectrum

frequencies_dft = np.arange(len(signal)) * (sampling_rate / len(signal))
spectrum_dft = np.zeros(len(frequencies_dft) // 2)

ah, bh = DFT(signal, sampling_rate)
spectrum_dft[:len(ah)] = ah**2 + bh**2  # оценка спектра мощности

# оценка спектра мощности методом Даньелла
window_width = 100  # ширина окна для метода Даньелла
spectrum_daniell = daniell_spectrum(signal, window_width, sampling_rate)

# визуализация результатов
plt.figure(figsize=(12, 6))

# спектр с использованием DFT
plt.subplot(1, 2, 1)
plt.plot(frequencies_dft[:signal_length // 2], spectrum_dft[:signal_length // 2])
plt.title('Спектр с использованием DFT')
plt.xlabel('Частота (Гц)')
plt.ylabel('Спектр мощности')
plt.grid()

# спектр методом Даньелла
plt.subplot(1, 2, 2)
plt.plot(spectrum_daniell)
plt.title('Спектр методом Даньелла')
plt.xlabel('Частота (Гц)')
plt.ylabel('Спектр мощности')
plt.grid()

plt.tight_layout()
plt.show()
