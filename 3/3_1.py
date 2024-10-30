# Запрограммировать ДПФ, оформив его расчет в виде процедуры.
# На входе подпрограммы: массив, содержащий временной ряд, целое N – длина
# анализируемого участка ряда.
# На выходе подпрограммы: массивы коэффициентов
# ah и bh: Ah и Bh соответственно
import numpy as np
import matplotlib.pyplot as plt
import math
# функция для вычисления дискретного преобразования Фурье
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

# загрузка сигнала из файла
signal = np.loadtxt('Sin.txt')

plt.plot(signal)
plt.title('временной ряд из обратного дпф')
plt.xlabel('время')
plt.ylabel('амплитуда')
plt.grid()
plt.show()

# вычисление коэффициентов дпф
a_h, b_h = DFT(signal, len(signal))

# вывод результатов
print("коэффициенты a_h:", a_h)
print("коэффициенты b_h:", b_h)
