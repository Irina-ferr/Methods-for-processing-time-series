# Запрограммировать метод оценки спектров Даньелла, оформив алгоритм в виде
# процедуры.
# На входе подпрограммы: массив данных, его длина, ширина окна усреднения в
# частотной области.
# На выходе подпрограммы: массив, содержащий оценку спектра мощности входного
# временного ряда, сделанную методом Даньелла.
# При реализации метода можно использовать подпрограмму, осуществляющую ДПФ,
# созданную при выполнении п. 3.1.
import numpy as np
import math
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

# Генерируем тестовый сигнал (синусоида)

A = 5 # амплитуда
F = 30 # частота
dt = 0.01 # шаг между измерениями
fs= 1/dt
t = np.linspace(0,20,int(20/dt)) # временная ось
signal = A*np.sin(2*math.pi*F*t) # создание сигнала длина 2000

# Параметры
window_width = 101  # Ширина окна усреднения

# Рассчитываем спектр по методу Даньелла
spectrum = daniell(signal, window_width, fs)
for i in range(len(spectrum)):
    spectrum[i]=spectrum[i]*(fs/len(spectrum))
plt.plot(spectrum)
plt.title("Оценка спектра мощности методом Даньелла")
plt.xlabel("Частота (Гц)")
plt.ylabel("Спектр мощности")
plt.grid()
plt.show()