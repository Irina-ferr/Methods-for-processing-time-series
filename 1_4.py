import numpy as np
import matplotlib.pyplot as plt

# Параметры
N = 10000  # Общее количество отсчетов
mean = 1   # Среднее значение
std_dev = 1  # Стандартное отклонение (дисперсия = std_dev^2)

# Генерация нормально распределенных случайных отсчетов
data = np.random.normal(loc=mean, scale=std_dev, size=N)

# Длины обрабатываемых участков
lengths = np.arange(1, N + 1)

# Инициализация массивов для хранения оценок среднего и дисперсии
means = np.zeros(N)
variances = np.zeros(N)

# Вычисление среднего и дисперсии для каждого участка длиной от 1 до N
for L in lengths:
    means[L - 1] = np.mean(data[:L])
    variances[L - 1] = np.var(data[:L])

# Построение графиков
plt.figure(figsize=(14, 6))

# График среднего
plt.subplot(1, 2, 1)
plt.plot(lengths, means, label='Среднее', color='blue')
plt.axhline(y=mean, color='red', linestyle='--', label='Истинное среднее')
plt.title('Зависимость среднего от длины участка')
plt.xlabel('Длина участка (L)')
plt.ylabel('Среднее значение')
plt.legend()
plt.grid()

# График дисперсии
plt.subplot(1, 2, 2)
plt.plot(lengths, variances, label='Дисперсия', color='green')
plt.axhline(y=std_dev**2, color='orange', linestyle='--', label='Истинная дисперсия')
plt.title('Зависимость дисперсии от длины участка')
plt.xlabel('Длина участка (L)')
plt.ylabel('Дисперсия')
plt.legend()
plt.grid()

# Показать графики
plt.tight_layout()
plt.show()
