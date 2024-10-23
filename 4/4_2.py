# Запрограммировать генерацию временной реализации гармонического сигнала с
# периодом 30 дискретных отсчетов. Сопоставить периодограммы реализаций такого
# сигнала длиной N_1=3000, N_2=2999 и N_3=2992. Объяснить наблюдающиеся эффекты.
import numpy as np
import matplotlib.pyplot as plt

# функция для генерации гармонического сигнала
def generate_signal(N, A, T):
    n = np.arange(N)
    return A * np.cos(2 * np.pi * n / T)

# параметры сигнала
A = 1          # амплитуда
T = 30         # период в дискретных отсчетах
N_values = [3000, 2999, 2992]  # длины реализаций

# построение периодограмм для каждой длины
plt.figure(figsize=(15, 10))

for i, N in enumerate(N_values):
    signal = generate_signal(N, A, T)
    
    # построение периодограммы
    plt.subplot(3, 1, i + 1)
    plt.psd(signal, x=256, y=1, color='blue', label=f'N={N}')
    plt.title(f'Периодограмма гармонического сигнала (N={N})')
    plt.xlabel('Частота')
    plt.ylabel('Плотность спектра мощности')
    plt.grid()

plt.tight_layout()
plt.show()
