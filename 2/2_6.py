# При использовании современных диагностических показателей состояния сердечнососудистой системы 
# человека важно корректно идентифицировать моменты R-пиков (высокоамплитудных короткие положительные
#  зубцы при регистрации кардиосигнала во II стандартном отведении) электрокардиограммы (ЭКГ). 
# Одним из методов решения этой задачи является сопоставление в скользящем окне участка 
# исследуемого сигнала с участком «эталонной» ЭКГ с помощью расчета ККФ.  
# В задаче требуется осуществить поиск R-пиков временного ряда ЭКГ участком с помощью оценки 
# в скользящем окне ККФ этого ряда и «эталонной» записи одного R-пика “ECG_sample_R.txt”. 
# Отобразить  графически результаты обработки, построив временной ряд ЭКГ с помеченными вершинами 
# Rпиков и рассчитываемую в скользящем окне величину ККФ. 
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

def sliding_window_correlation(ECG, ECG_sample_R, window_size):
    correlations = []
    for i in range(len(ECG) - window_size + 1):
        segment = ECG[i:i + window_size]
        correlation = cross_correlation(segment, ECG_sample_R, 0, len(ECG_sample_R))
        correlations.append(correlation)
    return correlations

# параметры
ECG = np.loadtxt("ECG.txt")
ECG_sample_R = np.loadtxt("ECG_sample_R.txt")
window_size = len(ECG_sample_R)
threshold = 0.5  # порог для определения пиков

# вычисление скользящей корреляции
correlations = sliding_window_correlation(ECG, ECG_sample_R, window_size)

# поиск пиков
peaks = np.where(np.array(correlations) > threshold)[0]

# визуализация 
plt.figure(figsize=(12, 6))

# for i in range (len(peaks)):
#     peaks[i]=peaks[i]+15
# peaks=peaks[::]
# график ЭКГ
plt.subplot(2, 1, 1)
plt.plot(ECG, label='ЭКГ')
plt.scatter(peaks, ECG[peaks], color='red', label='R-пики')
plt.title('ЭКГ с R-пиками')
plt.ylabel('Амплитуда')
plt.legend()

# график корреляции
plt.subplot(2, 1, 2)
plt.plot(correlations, label='Кросскорреляция', color='orange')
plt.axhline(y=threshold, color='r', linestyle='--', label='Порог')
plt.title('ККФ в скользящем окне')
plt.ylabel('Корр коэф')
plt.legend()

plt.tight_layout()
plt.show()

# plt.plot(ECG_sample_R)
# plt.show()
# в эталонной записи R пик
# сравнивать эталонную запись от 0 до смещения длина экг-эталон
# когда ккф большая отмечать пик 