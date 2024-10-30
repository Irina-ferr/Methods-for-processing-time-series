# С помощью построения графика периодограммы изучить спектральный состав ряда
# “X2000Hz.txt” (Частота дискретизация сигнала 2000 Гц).
# Проредить ряд, перевыбрав его до частоты 200 Гц (т.е. создать новый ряд, выбрав из
# исходного каждый 10 отсчет). Изучить периодограмму прореженного сигнала с
# частотой 200 Гц и сопоставить ее с периодограммой исходного сигнала. Объяснить
# наблюдающиеся эффекты.
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

data = np.loadtxt('X2000Hz.txt')

time = data[:, 0]  
signal = data[:, 1]  

# частота дискретизации
fs = 2000  # Гц

# построение периодограммы исходного сигнала с использованием DFT
ah, bh = DFT(signal, fs)
frequencies = np.arange(len(ah)) * fs / len(signal)
fig,ax=plt.subplots(nrows=2,sharex=True)
ax[0].plot(frequencies, np.sqrt(ah**2 + bh**2), color='blue')
ax[0].set_title('Периодограмма исходного сигнала (2000 Гц)')
ax[0].set_xlabel('Частота (Гц)')
ax[0].set_ylabel('Спектральная мощность')
ax[0].set_xlim(0, fs / 2)

t_frequency = fs/2
frequencies_mask = frequencies < t_frequency
ah[~frequencies_mask] = 0
bh[~frequencies_mask] = 0

downsampled_signal = signal[::10]
fs_new = fs / 10 

ah_downsampled, bh_downsampled = DFT(downsampled_signal, fs_new)
frequencies_new = np.arange(len(ah_downsampled)) * fs_new / len(downsampled_signal)

ax[1].plot(frequencies_new,np.sqrt(ah_downsampled**2 + bh_downsampled**2), color='red')
ax[1].set_title('Периодограмма сигнала (200 Гц)')
ax[1].set_xlabel('Частота (Гц)')
ax[1].set_ylabel('Спектральная мощность')
ax[1].set_xlim(0, fs_new / 2)

plt.show()