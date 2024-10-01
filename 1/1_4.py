import numpy as np
import matplotlib.pyplot as plt
def emp_sum (data, st_index, lenght):
    
    tot_sum=0.0 
    for i in range(st_index, st_index+lenght):
        tot_sum += data [i]   
    
    emp = tot_sum/lenght
   
    return emp

def calculate_variance(data, start_index, length):

    # извлекаем участок данных для анализа
    segment = data[start_index:start_index + length]
    
    # среднее значение
    mean = sum(segment) / length
    
    # дисперсия
    variance = sum((x - mean) ** 2 for x in segment) / length
    
    return variance
# параметры
N = 10000  # общ количество отсчетов
mean = 1   # среднее значение
std_dev = 1  # стандартное отклонение (дисперсия = std_dev^2)

# генерация нормально распределенных случайных отсчетов
data = np.random.normal(loc=mean, scale=std_dev, size=N)

# длины обрабатываемых участков
lengths = np.arange(100, 10000 + 1,100)

# инициализация массивов для хранения оценок среднего и дисперсии
means = np.zeros(100)
variances = np.zeros(100)

# вычисление среднего и дисперсии для каждого участка длиной от 1 до N
for i,L in enumerate (lengths):
    means[i] = np.mean(data[:L])
    # means[i] = calculate_variance(data[:L],0,L)
    variances[i] = emp_sum(data[:L],0,L)

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
