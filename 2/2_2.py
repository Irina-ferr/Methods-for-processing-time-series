# Используя подпрограмму, созданную при выполнении п. 2.1 рассчитать и 
# отобразить графически АКФ реализации белого шума 
# “White noise.txt” для целых тау от  0 до 200
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

# загрузка данных белого шума
white_noise = np.loadtxt("White noise.txt")

# параметры
N = len(white_noise)
taus = range(0, 201)  # изменяем диапазон до 200
autocorr_white_noise = []

# вычисление автокорреляционной функции для белого шума
for tau in taus:
    autocorr_white_noise.append(cross_correlation(white_noise, white_noise, tau, N))

# построение графика
plt.figure(figsize=(10, 6))
plt.plot(taus, autocorr_white_noise)
plt.title('Автокорреляционная функция белого шума')
plt.xlabel('Сдвиг (tau)')
plt.ylabel('Автокорреляция')
plt.grid()
plt.show()
