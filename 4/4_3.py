# Запрограммировать, оформив в виде процедуры, оконные преобразования Бартлетта,
# Хэмминга и прямоугольное окно.
# На входе подпрограммы: массив данных, длина массива, целочисленный код,
# задающий тип используемого окна.
# На выходе подпрограммы: массив, содержащий результат умножения исходного
# временного ряда на заданное окно. 
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

# генерация синусоиды
A = 10 # амплитуда первой синусоиды
F1 = 0.1 # частота первой синусоиды
F2 = 0.4 # частота второй синусоиды
dt = 0.01 # шаг между измерениями
t = np.linspace(0,20,int(20/dt)) # временная ось
signal = 5*np.sin(2*math.pi*F1*t) + 0.1*np.sin(2*math.pi*F2*t) # создание сигнала с двумя синусоидами
N=len(signal)
signal_1=[5] * N
plt.subplot(4, 1, 1)
plt.plot(t,signal,color = 'blue', linewidth = 2)

plt.title("Исходный сигнал")

plt.subplot(4, 1, 2)
plt.plot(t,Window_func(signal_1,N,1),color = 'orange', linewidth = 1)
plt.plot(t,Window_func(signal,N,1),color = 'green', linewidth = 1)
plt.title("Окно Бартлетта")


plt.subplot(4, 1, 3)
plt.plot(t,Window_func(signal_1,N,2),color = 'orange', linewidth = 1)
plt.plot(t,Window_func(signal,N,2),color = 'green', linewidth = 1)
plt.title("Окно Хэмминга")


plt.subplot(4, 1, 4) 
plt.plot(t,Window_func(signal_1,N,3),color = 'orange', linewidth = 1)
plt.plot(t,Window_func(signal,N,3),color = 'green', linewidth = 1)

plt.title("Прямоугольное окно")

plt.tight_layout() # автоматическая подгонка подписей и графиков

plt.show()
