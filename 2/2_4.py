# Рассчитывая АКФ сигнала “Noise+Sin.txt” 
# с помощью подпрограммы, созданной при выполнении п. 2.1, 
# определить период гармонической составляющей сигнала, 
# скрытой аддитивным шумом.  (Ответ: 20 дискретных отсчетов) 
import numpy as np
import matplotlib.pyplot as plt

def cross_correlation(X, Y, tau, N):
    if tau < 0:
        X_segment = X[-tau:N]
        Y_segment = Y[0:N + tau]
    else:
        X_segment = X[0:N - tau]
        Y_segment = Y[tau:N]

    mean_X = np.mean(X_segment)
    mean_Y = np.mean(Y_segment)

    std_X = np.std(X_segment)
    std_Y = np.std(Y_segment)

    covariance = np.sum((X_segment - mean_X) * (Y_segment - mean_Y))
    correlation = covariance / ((len(X_segment) - 1) * std_X * std_Y)

    return correlation

def find_peaks(data, threshold=0.09, min_distance=5):
    peaks = []
    for i in range(min_distance, len(data) - min_distance):
        if (data[i] > data[i - 1]) and (data[i] > data[i + 1]) and (data[i] > threshold):
            if not peaks or (i - peaks[-1] >= min_distance):
                peaks.append(i)
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
print("Средний период:", round(per))

# Построение графика
plt.figure(figsize=(10, 6))
plt.plot(taus, autocorr)
plt.plot(peaks, np.array(autocorr)[peaks], "x")  # отображаем пики
plt.title('Автокорреляционная функция')
plt.xlabel('Сдвиг (tau)')
plt.ylabel('Автокорреляция')
plt.grid()
plt.show()
