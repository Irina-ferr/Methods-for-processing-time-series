# Передаточная функция цифрового фильтра имеет вид: 
#     H(z)=0.01*z**9+0.03*z**8+0.09*z**7+0.20*z**6+0.20*z**5+0.20*z**4+0.20*z**3+0.20*z**4+0.20*z**3+0.09*z**2+0.03*z**1+0.01 / z**9
# Рассчитать методом деления в столбик и отобразить графически 1000 значений
# импульсной характеристики. Рассчитать и изобразить графически АЧХ этого
# фильтра в линейном масштабе. К какому классу относится линейная система,
# имеющая такую передаточную характеристику? Какой тип у анализируемого
# фильтра? Какая частота среза?
def calculate_impulse_response(denominator_coefficients, numerator_coefficients):
    # Коэффициенты знаменателя
    a = denominator_coefficients
    n = len(a)
    
    # Коэффициенты числителя
    b = numerator_coefficients
    
    # Инициализация импульсной характеристики
    impulse_response = [b[0] / a[0]]
    
    # Вычисление импульсной характеристики методом деления в столбик
    for i in range(1, n):
        subh = 0
        for j in range(1, i + 1):
            subh += a[j] * impulse_response[i - j]
        impulse_response.append((b[i] - subh) * (1 / a[0]))
    
    return impulse_response

# Пример использования функции с заданными коэффициентами
numerator_coefficients = [0.01, 0.03, 0.09, 0.20, 0.20, 0.20, 0.20, 0.20, 0.09, 0.03, 0.01]
denominator_coefficients = [1] + [0] * 9  # z^9 в знаменателе

# Получение импульсной характеристики
impulse_response = calculate_impulse_response(denominator_coefficients, numerator_coefficients)

# Дополнение до 1000 значений
impulse_response += [0] * (1000 - len(impulse_response))

# Графическое отображение импульсной характеристики
import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.stem(impulse_response)
plt.title('Импульсная характеристика фильтра')
plt.xlabel('n')
plt.ylabel('h[n]')
plt.grid()
plt.show()
