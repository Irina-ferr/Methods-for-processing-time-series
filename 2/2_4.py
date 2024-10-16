# Рассчитывая АКФ сигнала “Noise+Sin.txt” 
# с помощью подпрограммы, созданной при выполнении п. 2.1, 
# определить период гармонической составляющей сигнала, 
# скрытой аддитивным шумом.  (Ответ: 20 дискретных отсчетов) 
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

def find_peaks(data, threshold=0.09, min_distance=5):
    peaks = []
    for i in range(min_distance, len(data) - min_distance):
        if (data[i] > data[i - 1]) and (data[i] > data[i + 1]) and (data[i] > threshold):
            if not peaks or (i - peaks[-1] >= min_distance):
                peaks.append(i)
    peaks.append(len(data)-1)        
    return peaks

# загрузка данных
signal = np.loadtxt("Noise+Sin.txt")

# Параметры
N = len(signal)
taus = range(0, 1001)
autocorr = []

# вычисление автокорреляционной функции
for tau in taus:
    autocorr.append(cross_correlation(signal, signal, tau, N))

# поиск пиков в автокорреляционной функции
peaks = find_peaks(autocorr)

# определение периодов
periods = np.diff(peaks)

# вывод результатов
print("Пики автокорреляционной функции находятся на:", peaks)
print("Определенные периоды:", periods)
per= sum(periods) / len(periods)
print("Средний период:", per)

# Построение графика
plt.figure(figsize=(10, 6))
plt.plot(taus, autocorr)
plt.plot(peaks, np.array(autocorr)[peaks], "x")  # отображаем пики
plt.title('Автокорреляционная функция')
plt.xlabel('Сдвиг (tau)')
plt.ylabel('Автокорреляция')
plt.grid()
plt.show()
