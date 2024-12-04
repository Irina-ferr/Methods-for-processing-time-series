# Используя подпрограмму (п. 6.1) рассчитать свертку реализации {x_i } из файла
# “xi.txt” с импульсной характеристикой {h_i } из файла
# “hi.txt”. Отобразить на графиках {x_i }, {h_i } и {y_1 }, где
# y_i=h_i*x_i
import numpy as np
import matplotlib.pyplot as plt

def convolution_at_time(x, x_length, h, h_length, i):
    if i < 0 or i >= x_length + h_length - 1:
        raise ValueError("Индекс i выходит за пределы допустимого диапазона.")

    y_i = 0.0

    for j in range(h_length):
        x_index = i - j
        
        if 0 <= x_index < x_length:
            y_i += x[x_index] * h[j]

    return y_i

# Функция для чтения данных из файла
def read_data_from_file(filename):
    data = []
    with open(filename, 'r') as file:
        # Пропускаем заголовок
        next(file)
        for line in file:
            parts = line.strip().split()
            if len(parts) == 2:  # Убедимся, что есть два элемента
                data.append(float(parts[1]))  # Считываем только x_i или h_i
    return np.array(data)

# Чтение данных из файлов
x = read_data_from_file("xi.txt")
h = read_data_from_file("hi.txt")

# Длины массивов
x_length = len(x)
h_length = len(h)

# Длина результата свертки
y_length = x_length + h_length - 1
y = np.zeros(y_length)

# Вычисление свертки
for i in range(y_length):
    y[i] = convolution_at_time(x, x_length, h, h_length, i)

# Убираем последние две точки свертки
y = y[:-2]  # Удаляем последние две точки

# Обновляем длину результата после удаления точек
y_length -= 2

# Построение графиков
plt.figure(figsize=(12, 8))

# График x
plt.subplot(3, 1, 1)
plt.stem(np.arange(x_length), x, basefmt=" ", linefmt='b-', markerfmt='bo')
plt.title('Исходная последовательность x[n]')
plt.xlabel('n')
plt.ylabel('Амплитуда')
plt.xticks(np.arange(x_length))
plt.grid(axis='y')

# График h
plt.subplot(3, 1, 2)
plt.stem(np.arange(h_length), h, basefmt=" ", linefmt='orange', markerfmt='ro')
plt.title('Импульсная характеристика h[n]')
plt.xlabel('n')
plt.ylabel('Амплитуда')
plt.xticks(np.arange(h_length))
plt.grid(axis='y')

# График y
plt.subplot(3, 1, 3)
plt.stem(np.arange(y_length), y, basefmt=" ", linefmt='g-', markerfmt='go')
plt.title('Результат свертки y[n]')
plt.xlabel('n')
plt.ylabel('Амплитуда')
plt.xticks(np.arange(y_length))
plt.grid(axis='y')

# Показать графики
plt.tight_layout()
plt.show()