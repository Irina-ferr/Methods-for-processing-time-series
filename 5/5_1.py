# Рассчитать периодограмму реализации из файла “lfp.txt”
# (Fsamp=1 кГц) и построить графики функций спектральной плотности в линейном и
# логарифмическом масштабах.
import numpy as np
import matplotlib.pyplot as plt
def Welch(signal, N):
    
    return
# Загрузка данных из файла
lfp_data = np.loadtxt("lfp.txt") 

# Частота дискретизации
Fsamp = 1000 # Гц

# Вычисление периодограммы с помощью функции `welch`
f, Pxx_den = Welch(lfp_data, Fsamp)

# Построение графиков
plt.figure(figsize=(10, 6))
Pxx_den_log=Pxx_den
# Линейный масштаб
plt.subplot(2, 1, 1)
plt.plot(f, Pxx_den)
plt.title('Спектральная плотность мощности (линейный масштаб)')
plt.xlabel('Частота (Гц)')
plt.ylabel('Плотность спектра мощности')
plt.grid()

# Логарифмический масштаб
plt.subplot(2, 1, 2)
plt.semilogy(f, Pxx_den_log)
plt.title('Спектральная плотность мощности (логарифмический масштаб)')
plt.xlabel('Частота (Гц)')
plt.ylabel('Плотность спектра мощности')
plt.grid()

plt.tight_layout()
plt.show()
