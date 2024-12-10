# Иллюстрация свойства (6.9) свертки. Запрограммировать свертку {y_i} временного
# ряда суммы 3 гармонических функций {x_i} из “xi_3sin.txt” и
# импульсной характеристики частотно-избирательного фильтра {h_i} из файла
# “hi_BandPass100Hz_1kHz.txt”. Рассчитать и отобразить
# графически: |X( f )| , |H( f )| , |Y( f )| , |X ( f ) *Y( f )| где Y( f ) , X ( f ) и H( f ) – Фурье-
# образы {x_i}, {h_i} и {y_i} ( i i i y  h  x ), соответственно
import numpy as np
import matplotlib.pyplot as plt

def convolution_at_time(x, x_length, h, h_length, i):
    if i < 0 or i >= x_length + h_length - 1:
        raise ValueError("Индекс i выходит за пределы допустимого диапазона.")

    y_i = 0.0

    for j in range(h_length):
        x_index = i - j
        
        if 0 <= x_index < x_length:
            y_i += x[x_index] * h[j]

    return y_i

def DFT(signal):
    N = len(signal)
    ah = np.zeros(N // 2 + 1)  # массив для коэффициентов a_h
    bh = np.zeros(N // 2 + 1)  # массив для коэффициентов b_h

    for h in range(N // 2 + 1):
        sum_sig1 = np.sum(signal * np.cos(2 * np.pi * h * np.arange(N) / N))
        sum_sig2 = np.sum(signal * np.sin(2 * np.pi * h * np.arange(N) / N))

        ah[h] = (2 / N) * sum_sig1
        bh[h] = (-2 / N) * sum_sig2

    return ah, bh

# Чтение данных из файлов
x = np.loadtxt("xi_3sin.txt")
h = np.loadtxt("hi_BandPass100Hz_1kHz.txt")

# Длина сигналов
x_length = len(x)
h_length = len(h)

# Рассчет свертки
y_length = x_length 
y = np.zeros(y_length)

for i in range(y_length):
    y[i] = convolution_at_time(x, x_length, h, h_length, i)

# Выполнение ДПФ
ah_x, bh_x = DFT(x)
ah_h, bh_h = DFT(h)
ah_y, bh_y = DFT(y)

# Расчет амплитуд
X_f = np.sqrt(ah_x**2 + bh_x**2)

H_f = np.sqrt(ah_h**2 + bh_h**2)

Y_f = np.sqrt(ah_y**2 + bh_y**2)

# Расчет произведения X(f) и Y(f)
XY_f = X_f * Y_f[:len(X_f)]

# Частота для графиков
freqs_x = np.linspace(0, 0.5, len(X_f))  
freqs_h = np.linspace(0, 0.5, len(H_f))
freqs_y = np.linspace(0, 0.5, len(Y_f))
freqs_xy = freqs_x  

# Построение графиков
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(freqs_x, X_f)
plt.title('|X(f)|')
plt.xlabel('Частота')
plt.ylabel('Амплитуда')

plt.subplot(2, 2, 2)
plt.plot(freqs_h, H_f)
plt.title('|H(f)|')
plt.xlabel('Частота')
plt.ylabel('Амплитуда')

plt.subplot(2, 2, 3)
plt.plot(freqs_y, Y_f)
plt.title('|Y(f)|')
plt.xlabel('Частота')
plt.ylabel('Амплитуда')

plt.subplot(2, 2, 4)
plt.plot(freqs_xy, XY_f)
plt.title('|X(f) * Y(f)|')
plt.xlabel('Частота')
plt.ylabel('Амплитуда')

plt.tight_layout()
plt.show()
