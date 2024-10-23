# Запрограммировать ДПФ, оформив его расчет в виде процедуры.
# На входе подпрограммы: массив, содержащий временной ряд, целое N – длина
# анализируемого участка ряда.
# На выходе подпрограммы: массивы коэффициентов
# ah и bh: Ah и Bh соответственно
import numpy as np
import math
import matplotlib.pyplot as plt

# функция для расчета коэффициентов дпф
def DFT(signal, N):
    """
    вычисляет коэффициенты a_h и b_h для дискретного преобразования фурье (дпф) сигнала.
    
    на вход: 
        signal: входной массив сигналов (временной ряд).
        N: длина анализируемого участка ряда

    на выход:
        ah: массив коэффициентов a_h (косинусные компоненты).
        bh: массив коэффициентов b_h (синусные компоненты).
    """
    
    ah = np.zeros(N // 2 + 1)  # массив для коэффициентов a_h
    bh = np.zeros(N // 2 + 1)  # массив для коэффициентов b_h

    for h in range(0, N // 2 + 1):
        sum_sig1 = 0.0
        sum_sig2 = 0.0
        for i in range(N):
            dt = (i + 1) * 0.1  # предполагаем, что шаг времени равен 0.1
            fh = h / (N * dt)   # частота

            sum_sig1 += signal[i] * math.cos(2 * math.pi * fh * (i * 0.1))
            sum_sig2 += signal[i] * math.sin(2 * math.pi * fh * (i * 0.1))

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
