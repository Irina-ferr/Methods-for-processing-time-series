# Используя созданную в п. 3.2 подпрограмму, рассчитать обратное ДПФ по
# имеющемуся амплитудному спектру созданного в п. 3.4 сигнала. Отобразить на
# одном графике исходный временной ряд (сумму 2 гармонических составляющих),
# ряд, полученный в результате обратного ДПФ, разницу между этими рядами.
import numpy as np
import matplotlib.pyplot as plt

def re_DFT(a_h, b_h, M, N):
    """
    вычисляет временной ряд на основе коэффициентов a_h и b_h.
    
    на вход: 
        a_h: массив коэффициентов a_h (косинусные компоненты).
        b_h: массив коэффициентов b_h (синусные компоненты).
        M: количество гармоник.
        N: количество точек времени.

    на выход:
        signal: массив, содержащий временной ряд.
    """
    signal = np.zeros(N)  # создаем массив нулей для временного ряда
    
    for n in range(N):  # перебираем все точки времени
        for k in range(M):  # перебираем все гармоники
            # добавляем вклад от каждой гармоники
            signal[n] += a_h[k] * np.cos(2 * np.pi * k * n / N) + b_h[k] * np.sin(2 * np.pi * k * n / N)
    
    return signal  # возвращаем восстановленный сигнал

def DFT(signal,sampling_rate):
    N=len(signal)
    if (N%2==1):
        M=round(N/2)+1
    else:
        M=N/2
    M=int(M)    
    ah = np.zeros(M)  # массив для коэффициентов a_h
    bh = np.zeros(M)  # массив для коэффициентов b_h

    for h in range(0, M):
        sum_sig1 = 0.0
        sum_sig2 = 0.0
        fh = h/N*sampling_rate  # частота
        for i in range(N):
            
            sum_sig1 += signal[i] * np.cos(2 * np.pi * fh * (i / sampling_rate))
            sum_sig2 += signal[i] * np.sin(2 * np.pi * fh * (i / sampling_rate))

        ah[h] = (2 / N) * sum_sig1
        bh[h] = (-2 / N) * sum_sig2

    return ah, bh
# параметры
fs = 100  # частота дискретизации (Гц)
duration = 2  # длительность сигнала (сек)
t = np.linspace(0, duration, int(fs * duration), endpoint=False)  # временные метки

# генерация синусоидальных сигналов
amplitude1 = 1.2
frequency1 = 10  # частота первого сигнала (Гц)
signal1 = amplitude1 * np.sin(2 * np.pi * frequency1 * t)  # первый сигнал

amplitude2 = 2
frequency2 = 20  # частота второго сигнала (Гц)
signal2 = amplitude2 * np.sin(2 * np.pi * frequency2 * t)  # второй сигнал

# сумма сигналов
combined_signal = signal1 + signal2  # комбинированный сигнал
N=len(combined_signal)
a_h,b_h=DFT(combined_signal,N)
if (N%2==1):
    M=round(N/2)+1
else:
    M=N/2
M=int(M) 
# восстановление сигнала с помощью обратного дпф
reconstructed_signal = re_DFT(a_h, b_h, M, N)
reconstructed_signal=reconstructed_signal*(-1)
# разница между исходным и восстановленным сигналом
difference_signal = combined_signal - reconstructed_signal  # разница между сигналами

# визуализация результатов
plt.figure(figsize=(12, 8))

# исходный сигнал
plt.subplot(3, 1, 1)
plt.plot(t, combined_signal, label='исходный сигнал', color='blue')  # график исходного сигнала
plt.title('исходный сигнал')
plt.xlabel('время (с)')
plt.ylabel('амплитуда')
plt.grid()
plt.legend()

# восстановленный сигнал
plt.subplot(3, 1, 2)
plt.plot(t, reconstructed_signal, label='восстановленный сигнал', color='orange')  # график восстановленного сигнала
plt.title('восстановленный сигнал после обратного дпф')
plt.xlabel('время (с)')
plt.ylabel('амплитуда')
plt.grid()
plt.legend()

# разница между сигналами
plt.subplot(3, 1, 3)
plt.plot(t, difference_signal, label='разница', color='red')  # график разницы между сигналами
plt.title('разница между исходным и восстановленным сигналами')
plt.xlabel('время (с)')
plt.ylabel('амплитуда')
plt.grid()
plt.legend()

plt.tight_layout()  # автоматическая настройка макета графиков
plt.show()  # отображение графиков
