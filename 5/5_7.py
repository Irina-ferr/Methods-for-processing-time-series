# Запрограммировать генерацию временной реализации гармонического сигнала с
# периодом 30 дискретных отсчетов длиной 3000 отсчетов. Сопоставить оценки
# спектра мощности этой реализации, сделанные с помощью метода Даньелла и
# метода Уэлча. Ширину окна для построения периодограммы задать равной 300
# отсчетов, перекрытие при сдвиге окон задать равным 50% ширины окна.
import numpy as np
import matplotlib.pyplot as plt

def generate_harmonic_signal(period, length, amplitude=1.0, phase=0):
    t = np.arange(length)
    signal = amplitude * np.sin(2 * np.pi * t / period + phase)
    return signal

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

def daniell_spectrum(signal, window_width):
    ah, bh = DFT(signal)
    power_spectrum = ah**2 + bh**2
    N=len(power_spectrum)
    spectrum=[]
    for i in range (N-1):
        if((i<window_width) or(i>(N-window_width-1))):
            spectrum.append(power_spectrum[i])
            
        else:
            window=power_spectrum[i-(window_width//2):i+(window_width//2)+1]
            daniell_sp=sum(window)/len(window)
            spectrum.append(daniell_sp)
    return spectrum

# Функция для оконной функции
def Window_func(signal, N, window_type):
    signal_modded = []
    if window_type == 1:  # Бартлетт
        for i in range(N):
            t_i = (i - N / 2) / N
            window = 1 - 2 * abs(t_i)
            signal_modded.append(signal[i] * window)
    elif window_type == 2:  # Хэмминг
        for i in range(N):
            window = 0.54 - 0.46 * np.cos(2 * np.pi * i / (N - 1))
            signal_modded.append(signal[i] * window)
    elif window_type == 3:  # Прямоугольное
        signal_modded = signal[:N]  # просто копируем сигнал
    else:
        raise ValueError('Не выбрано окно')

    return np.array(signal_modded)

def Welch(signal, signal_length, window_width, window_shift, window_type):
    num_windows = (signal_length - window_width) // window_shift + 1
    spectrum_power = np.zeros(window_width // 2 + 1)

    for i in range(num_windows):
        start_index = i * window_shift
        end_index = start_index + window_width
        
        if end_index > signal_length:
            break
        
        current_window = signal[start_index:end_index]
        windowed_signal = Window_func(current_window, window_width, window_type)
        
        ah, bh = DFT(windowed_signal)
        
        power_spectrum = ah**2 + bh**2
        
        spectrum_power += power_spectrum

    spectrum_power /= num_windows
    
    return spectrum_power


# Параметры сигнала
period = 30  # Период в дискретных отсчетах
length = 3000  # Длина сигнала
amplitude = 1.0
phase = 0

# Генерация сигнала
signal = generate_harmonic_signal(period, length, amplitude, phase)

# Параметры для методов
window_size = 600  # ширина окна
overlap = window_size // 2  # сдвиг
window_type = 2  #хэмминг

# Оценка спектра мощности
spectrum_daniell = daniell_spectrum(signal, window_size)
spectrum_welch = Welch(signal, length, window_size, overlap, window_type)

# Частоты для графиков
frequencies_daniell = np.linspace(0, len(signal) / (2 * period), len(spectrum_daniell))
frequencies_welch = np.linspace(0, len(signal) / (2 * period), len(spectrum_welch))


# # График метода Даньелла
# plt.subplot(2, 1, 1)
# plt.plot(frequencies_daniell, spectrum_daniell)
# plt.title('Спектр мощности (Метод Даньелла)')
# plt.xlabel('Частота (Гц)')
# plt.ylabel('Спектр мощности')
# plt.grid()

# # График метода Уэлча
# plt.subplot(2, 1, 2)
# plt.plot(frequencies_welch, spectrum_welch)
# plt.title('Спектр мощности (Метод Уэлча)')
# plt.xlabel('Частота (Гц)')
# plt.ylabel('Спектр мощности')
# plt.grid()

# plt.tight_layout()
# plt.show()
# Определение пределов оси Y
y_min = min(np.min(spectrum_daniell), np.min(spectrum_welch))
y_max = max(np.max(spectrum_daniell), np.max(spectrum_welch))

# График метода Даньелла
plt.subplot(2, 1, 1)
plt.plot(frequencies_daniell, spectrum_daniell)
plt.title('Спектр мощности (Метод Даньелла)')
plt.xlabel('Частота (Гц)')
plt.ylabel('Спектр мощности')
plt.grid()
plt.ylim(y_min, y_max)  # Установка одинаковых пределов по оси Y

# График метода Уэлча
plt.subplot(2, 1, 2)
plt.plot(frequencies_welch, spectrum_welch)
plt.title('Спектр мощности (Метод Уэлча)')
plt.xlabel('Частота (Гц)')
plt.ylabel('Спектр мощности')
plt.grid()
plt.ylim(y_min, y_max)  # Установка одинаковых пределов по оси Y

plt.tight_layout()
plt.show()
