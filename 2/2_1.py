# Запрограммировать расчет кросскорреляционной функции
# Оформить алгоритм в виде функции  На входе подпрограммы: 
# массивы сигналов X и Y, величина сдвига тау , длина анализируемых участков реализаций N.  
# На выходе подпрограммы: кросскорреляционная функция 
# Используя созданную подпрограмму рассчитать и построить график автокорреляционной функции
# временного ряда “Sin.txt” для целых тау от  0 до 1000 
import numpy as np
import matplotlib.pyplot as plt
import math as m


def mean (data):
    st_index=0
    lenght=len(data)
    tot_sum=0.0 
    for i in range(st_index, st_index+lenght):
        tot_sum += data [i]   
    
    emp = tot_sum/lenght
   
    return emp

def variance(data):
    start_index=0
    length=len(data)

    # извлекаем участок данных для анализа
    segment = data[start_index:start_index + length]
    
    # среднее значение
    mean = sum(segment) / length
    
    # дисперсия
    variance = sum((x - mean) ** 2 for x in segment) / length
    return variance

def cross_correlation(X, Y, tau, N):
    # проверяем, является ли сдвиг отрицательным
    if tau < 0:
        # для отрицательного сдвига выбираем сегменты из X и Y
        X_segment = X[-tau:N]
        Y_segment = Y[0:N + tau]
    else:
        # для положительного сдвига выбираем сегменты из X и Y
        X_segment = X[0:N - tau]
        Y_segment = Y[tau:N]

    # вычисляем средние значения для сегментов #поменять функцию на эмп среднее
    mean_X = mean(X_segment) 
    mean_Y = mean(Y_segment)

    # вычисляем стандартные отклонения для сегментов эмп дисперсия
    std_X = variance(X_segment) #var
    std_Y = variance(Y_segment)

    # вычисляем ковариацию между сегментами
    covariance = np.sum((X_segment - mean_X) * (Y_segment - mean_Y))
    # вычисляем корреляцию
    correlation = covariance / ((len(X_segment) - 1) * m.sqrt (std_X * std_Y) )

    return correlation

def load_data(filename):
    # загружаем данные из файла
    return np.loadtxt(filename)

# загрузка данных из файла "Sin.txt"
signal = load_data("Sin.txt")

# определяем параметры
N = len(signal)  # длина сигнала
taus = range(0, 1001)  # диапазон значений сдвига от 0 до 1000
autocorr = []  # список для хранения значений автокорреляции

# вычисление автокорреляционной функции для каждого значения сдвига
for tau in taus:
    autocorr.append(cross_correlation(signal, signal, tau, N))

# построение графика автокорреляционной функции
plt.figure(figsize=(10, 6))
plt.plot(taus, autocorr)
plt.title('Автокорреляционная функция')  # заголовок графика
plt.xlabel('Сдвиг (tau)')  # подпись оси X
plt.ylabel('Автокорреляция')  # подпись оси Y
plt.grid()  # включаем сетку на графике
plt.show()  # отображаем график
