import numpy as np
import matplotlib.pyplot as plt

# параметры
N = 10000  # общ количество отсчетов
mean = 1   # среднее значение
std_dev = 1  # стандартное отклонение (дисперсия = std_dev^2)

# генерация нормально распределенных случайных отсчетов
data = np.random.normal(loc=mean, scale=std_dev, size=N)

# длины обрабатываемых участков
lengths = np.arange(1, N + 1)

# инициализация массивов для хранения оценок среднего и дисперсии
means = np.zeros(N)
variances = np.zeros(N)

# вычисление среднего и дисперсии для каждого участка длиной от 1 до N
for L in lengths:
    means[L - 1] = np.mean(data[:L])
    variances[L - 1] = np.var(data[:L])

# построение графиков
plt.figure(figsize=(14, 6))

# график среднего
plt.subplot(1, 2, 1)
plt.plot(lengths, means, label='Среднее', color='blue')
plt.axhline(y=mean, color='red', linestyle='--', label='Истинное среднее')
plt.title('Зависимость среднего от длины участка')
plt.xlabel('Длина участка (L)')
plt.ylabel('Среднее значение')
plt.legend()
plt.grid()

# график дисперсии
plt.subplot(1, 2, 2)
plt.plot(lengths, variances, label='Дисперсия', color='green')
plt.axhline(y=std_dev**2, color='orange', linestyle='--', label='Истинная дисперсия')
plt.title('Зависимость дисперсии от длины участка')
plt.xlabel('Длина участка (L)')
plt.ylabel('Дисперсия')
plt.legend()
plt.grid()

# показать графики
plt.tight_layout()
plt.show()
