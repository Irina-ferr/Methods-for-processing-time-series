import numpy as np
import matplotlib.pyplot as plt

def DFT(signal, sampling_rate):
    # Функция для вычисления дискретного преобразования Фурье (ДПФ)
    N = len(signal)  # Длина входного сигнала
    if N % 2 == 1:
        M = round(N / 2) + 1  # Если длина нечетная, округляем
    else:
        M = N // 2  # Если четная, просто делим на 2
    M = int(M)
    
    # Массивы для коэффициентов a_h и b_h
    ah = np.zeros(M)  
    bh = np.zeros(M)  

    # Вычисление коэффициентов ДПФ
    for h in range(M):
        sum_sig1 = 0.0
        sum_sig2 = 0.0
        fh = h / N * sampling_rate  # Частота для данного h
        for i in range(N):
            sum_sig1 += signal[i] * np.cos(2 * np.pi * fh * (i / sampling_rate))  # Сумма для a_h
            sum_sig2 += signal[i] * np.sin(2 * np.pi * fh * (i / sampling_rate))  # Сумма для b_h

        ah[h] = (2 / N) * sum_sig1  # Нормировка для a_h
        bh[h] = (-2 / N) * sum_sig2  # Нормировка для b_h

    return ah, bh

# Загрузка данных из файла
data = np.loadtxt('X2000Hz.txt')

# Извлечение времени и сигнала из данных
time = data[:, 0]  
signal = data[:, 1]  

# Частота дискретизации
fs = 2000  # Гц

# Построение периодограммы исходного сигнала с использованием DFT
ah, bh = DFT(signal, fs)  # Вычисление коэффициентов ДПФ
frequencies = np.arange(len(ah)) * fs / len(signal)  # Определение частот

# Создание графиков для отображения периодограммы
fig, ax = plt.subplots(nrows=2, sharex=True)

# Периодограмма исходного сигнала
ax[0].plot(frequencies, np.sqrt(ah**2 + bh**2), color='blue')  # Спектральная мощность
ax[0].set_title('Периодограмма исходного сигнала (2000 Гц)')
ax[0].set_xlabel('Частота (Гц)')
ax[0].set_ylabel('Спектральная мощность')
ax[0].set_xlim(0, fs / 2)  # Ограничение по оси X до половины частоты дискретизации

# Маска для фильтрации частот выше половины частоты дискретизации
t_frequency = fs / 2  
frequencies_mask = frequencies < t_frequency
ah[~frequencies_mask] = 0  # Обнуление коэффициентов вне диапазона
bh[~frequencies_mask] = 0  

# Прореживание сигнала, выбирая каждый 10-й отсчет
downsampled_signal = signal[::10]
fs_new = fs / 10  # Новая частота дискретизации (200 Гц)

# Построение периодограммы прореженного сигнала
ah_downsampled, bh_downsampled = DFT(downsampled_signal, fs_new)  # Вычисление ДПФ для нового сигнала
frequencies_new = np.arange(len(ah_downsampled)) * fs_new / len(downsampled_signal)  # Частоты нового сигнала

# Периодограмма прореженного сигнала
ax[1].plot(frequencies_new, np.sqrt(ah_downsampled**2 + bh_downsampled**2), color='red')  
ax[1].set_title('Периодограмма сигнала (200 Гц)')
ax[1].set_xlabel('Частота (Гц)')
ax[1].set_ylabel('Спектральная мощность')
ax[1].set_xlim(0, fs_new / 2)  # Ограничение по оси X до половины новой частоты дискретизации

plt.show()  # Отображение графиков
