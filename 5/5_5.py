# Запрограммировать метод оценки спектров Уэлча, оформив алгоритм в виде
# процедуры.
# На входе подпрограммы: массив данных, его длина, ширина окна, используемого
# при расчете периодограмм, сдвиг окна, используемого при расчете периодограмм,
# целочисленный код, определяющий тип оконного преобразования.
# На выходе подпрограммы: массив, содержащий оценку спектра мощности входного
# временного ряда, сделанную методом Уэлча.
# При реализации метода можно использовать подпрограммы, осуществляющие ДПФ DFT
# и оконное преобразование сигнала  Window_func

import math
import numpy as np
import matplotlib.pyplot as plt

def Window_func(signal, N, window_type):
    
    signal_modded = []
    match window_type:
        case 1: #Бартлетт
            for i in range (N):
                t_i=(i-N/2)/N
                window=1-2*abs(t_i)
                signal_modded.append(signal[i]*window)
            return signal_modded

        case 2: #Хэмминг
            for i in range (N):
                t_i=(i-N/2)/N
                window=0.46*math.cos(2*math.pi*t_i)+0.54
                signal_modded.append(signal[i]*window)
            return signal_modded

        case 3: #Прямоугольное
            window = [1] * N
            for i in range (N):
                if (i == 0) and (i == N-1):
                    window[i] = 0
            signal_modded = [signal[i] * window[i] for i in range(N)]
            return signal_modded

        case _:
            return print ('Не выбрано окно')

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

