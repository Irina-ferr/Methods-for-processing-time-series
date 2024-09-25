# запрограммировать оценку дисперсии по участку реализации, оформив алгоритм в виде функции.
import numpy as np

def calculate_variance(data, start_index, length):

    # извлекаем участок данных для анализа
    segment = data[start_index:start_index + length]
    
    # среднее значение
    mean = sum(segment) / length
    
    # дисперсия
    variance = sum((x - mean) ** 2 for x in segment) / length
    
    return variance


data = np.loadtxt ('mean1.txt')
print (data)
st_index = int (input ("индекс первого элемента массива "))
lenght = int (input ("длина анализируемого участка "))
variance = calculate_variance(data, st_index, lenght)
print(f"Дисперсия: {variance}")

