# Запрограммировать оценку эмпирического среднего по участку временного ряда 
# методом расчета частичных сумм, оформив алгоритм в виде функции.
# Вход и выход подпрограммы: аналогично п. 1.1. 
import numpy as np

def emp_mean_partial_sums(data, start_index, length):
    m= int(np.log2(lenght))
    for j in range (1,m+1):
        for i in range (0,int((lenght/pow(2,j)))):
            data[i]= (data[2*i]+data[(2*i)+1]) / 2 
    return data[0]
   
data = np.loadtxt  ('mean2.txt')
print (data)
st_index = int (input ("индекс первого элемента массива "))
lenght = int (input ("длина анализируемого участка "))
m = np.log2(lenght)
mean = emp_mean_partial_sums(data, st_index, lenght)
print (f"Эмпирическое среднее {mean:.2f}")






