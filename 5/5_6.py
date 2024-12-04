# Применить метод Уэлча для оценки спектра мощности реализации процесса белого
# шума “Noise.txt”, частота дискретизации 10 Гц. При выборе
# параметров метода Уэлча обеспечить спектральное разрешение не хуже 0.1 Гц.
import numpy as np
import matplotlib.pyplot as plt

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

# Функция для вычисления ДПФ
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

# Функция метода Уэлча
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

# Шаг 1: Загрузка данных
signal = np.loadtxt("Noise.txt")  # Загрузка данных из файла
fs = 10  # Частота дискретизации в Гц

# Шаг 2: Определение параметров метода Уэлча
window_width = 100  # Длина окна (N)
window_shift = 50   # Шаг сдвига окна
window_type = 2     # Хэмминг

# Шаг 3: Оценка спектра мощности методом Уэлча
spectrum_power = Welch(signal, len(signal), window_width, window_shift, window_type)

# Шаг 4: Построение графика
freqs = np.linspace(0, fs / 2, len(spectrum_power))
plt.plot(freqs, spectrum_power)
plt.title('Оценка спектра мощности методом Уэлча для белого шума')
plt.xlabel('Частота (Гц)')
plt.ylabel('Спектр мощности')
plt.grid()
plt.show()
