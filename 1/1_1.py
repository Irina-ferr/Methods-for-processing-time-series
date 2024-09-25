# 1.1	Запрограммировать оценку эмпирического среднего по участку временного ряда методом суммирования с накоплением, оформив алгоритм в виде функции.
# 
# На входе подпрограммы: массив элементов вещественного типа, индекс первого элемента массива и длина анализируемого участка в отсчетах. 
# Тест подпрограммы производить, загрузив данные из файлов “Практическое занятие 1/mean1.txt” и “Практическое занятие 1/mean2.txt”. 

import numpy as np
def emp_sum (data, st_index, lenght):
    
    tot_sum=0.0 
    for i in range(st_index, st_index+lenght):
        tot_sum += data [i]   
    
    emp = tot_sum/lenght
   
    return emp



data1 = np.loadtxt  ('mean1.txt')
print (data1)

st_index = int (input ("индекс первого элемента массива "))
lenght = int (input ("длина анализируемого участка "))
mean = emp_sum (data1, st_index, lenght)
print (f"Эмпирическое среднее {mean}")


data2 = np.loadtxt  ('mean2.txt')
print (data2)
st_index = int (input ("индекс первого элемента массива "))
lenght = int (input ("длина анализируемого участка "))
mean= emp_sum (data2, st_index, lenght)
print (f"Эмпирическое среднее {mean}")






