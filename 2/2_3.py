# Используя подпрограмму, созданную при выполнении п. 2.1 
# рассчитать и отобразить графически АКФ реализации цветного шума 
# “Red noise.txt” для целых тау от  0 до 200
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

# загрузка данных
signal_red_noise = np.loadtxt("Red noise.txt")

# параметры
N_red_noise = len(signal_red_noise)
taus_red_noise = range(0, 201)  # от 0 до 200
autocorr_red_noise = []

# вычисление автокорреляционной функции для красного шума
for tau in taus_red_noise:
    autocorr_red_noise.append(cross_correlation(signal_red_noise, signal_red_noise, tau, N_red_noise))

# построение графика
plt.figure(figsize=(10, 6))
plt.plot(taus_red_noise, autocorr_red_noise)
plt.title('Автокорреляционная функция для красного шума')
plt.xlabel('Сдвиг (tau)')
plt.ylabel('Автокорреляция')
plt.grid()
plt.show()
