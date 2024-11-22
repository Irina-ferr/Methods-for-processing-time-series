# С помощью подпрограммы, созданной при выполнения п. 5.2 оценить спектр
# реализации белого шума из файла “Noise.txt” при различных
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

# загрузка белого шума из файла
signal = np.loadtxt('noise.txt')

# параметры
sampling_rate = 250  # Частота дискретизации
window_widths = [63, 127, 255, 511]  # Ширины окон усреднения

# визуализация результатов
plt.figure(figsize=(10, 8))

for window_width in window_widths:
    spectrum = daniell_spectrum(signal, window_width, sampling_rate)
    for i in range(len(spectrum)):
        spectrum[i]=spectrum[i]*(sampling_rate/len(spectrum))
    freqs = np.arange(len(spectrum)) * (sampling_rate / len(spectrum))
    
    plt.plot(spectrum, label=f'Width = {window_width}')

plt.title("Оценка спектра мощности методом Даньелла")
plt.xlabel("Частота (Гц)")
plt.ylabel("Спектр мощности")
plt.legend()
plt.grid()
plt.show()