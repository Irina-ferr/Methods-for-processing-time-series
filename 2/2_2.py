# Используя подпрограмму, созданную при выполнении п. 2.1 рассчитать и 
# отобразить графически АКФ реализации белого шума 
# “White noise.txt” для целых тау от  0 до 200
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
