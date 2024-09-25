# Запрограммировать оценку эмпирического среднего по участку временного ряда 
# методом расчета частичных сумм, оформив алгоритм в виде функции.
# Вход и выход подпрограммы: аналогично п. 1.1. 
import numpy as np

def compute_partial_sums(data):
  
    partial_sums = [0] * (len(data) + 1)
    for i in range(1, len(data) + 1):
        partial_sums[i] = partial_sums[i - 1] + data[i - 1]
    return partial_sums

def emp_mean_partial_sums(data, start_index, length):
    
    # вычисление частичных сумм
    partial_sums = compute_partial_sums(data)
    
    # вычисление суммы и эмпирического среднего
    total_sum = partial_sums[start_index + length] - partial_sums[start_index]
    mean = total_sum / length
    
    return mean

data = np.loadtxt  ('mean2.txt')
print (data)
st_index = int (input ("индекс первого элемента массива "))
lenght = int (input ("длина анализируемого участка "))
mean= emp_mean_partial_sums(data, st_index, lenght)
print (f"Эмпирическое среднее {mean:.2f}")






