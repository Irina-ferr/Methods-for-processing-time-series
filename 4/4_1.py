# С помощью построения графика периодограммы изучить спектральный состав ряда
# “X2000Hz.txt” (Частота дискретизация сигнала 2000 Гц).
# Проредить ряд, перевыбрав его до частоты 200 Гц (т.е. создать новый ряд, выбрав из
# исходного каждый 10 отсчет). Изучить периодограмму прореженного сигнала с
# частотой 200 Гц и сопоставить ее с периодограммой исходного сигнала. Объяснить
# наблюдающиеся эффекты.
import numpy as np
import matplotlib.pyplot as plt

# Загрузка данных
data = np.loadtxt('X2000Hz.txt')

# Предполагаем, что первый столбец - это время, второй - значения сигнала
time = data[:, 0]  # Временные метки
signal = data[:, 1]  # Значения сигнала

# Частота дискретизации
fs = 2000  # Гц

# Построение периодограммы исходного сигнала
plt.subplot(2,1,1)
plt.psd(signal, NFFT=1024, Fs=fs, color='blue')
plt.title('Периодограмма исходного сигнала (2000 Гц)')
plt.xlabel('Частота (Гц)')
plt.ylabel('Спектральная мощность')
plt.grid()

# Прореживание сигнала
downsampled_signal = signal[::10]  # Выбираем каждый 10-й отсчет
fs_new = fs / 10  # Новая частота дискретизации (200 Гц)

# Построение периодограммы прореженного сигнала
plt.subplot(2,1,2)
plt.psd(downsampled_signal, NFFT=1024, Fs=fs_new, color='red')
plt.title('Периодограмма прореженного сигнала (200 Гц)')
plt.xlabel('Частота (Гц)')
plt.ylabel('Спектральная мощность')
plt.grid()
plt.legend()
plt.tight_layout()  # автоматическая настройка макета графиков
plt.show()
