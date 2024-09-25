# Запрограммировать расчет кросскорреляционной функции
# Оформить алгоритм в виде функции  На входе подпрограммы: 
# массивы сигналов X и Y, величина сдвига тау , длина анализируемых участков реализаций N.  
# На выходе подпрограммы: кросскорреляционная функция 
# Используя созданную подпрограмму рассчитать и построить график автокорреляционной
# временного ряда “Sin.txt” для целых тау от  0 до 1000 
import numpy as np
import matplotlib.pyplot as plt

def cross_correlation(X, Y, tau, N):
  """
  Рассчитывает кросскорреляционную функцию для двух сигналов.

  Args:
      X: Первый сигнал 
      Y: Второй сигнал 
      tau: 
      N: Длина анализируемых участков реализаций

  Returns:
      Кросскорреляционная функция (float).
  """

  if len(X) < N + tau or len(Y) < N + tau:
    raise ValueError("Длина сигналов должна быть больше N + tau.")

  # вычисление ККФ
  rxy = np.sum((X[:N] * Y[tau:N+tau])) / N

  return rxy

# загрузка данных из файла
data = np.loadtxt("Sin.txt")

# вычисление АКФ для различных значений tau
taus = np.arange(0, 1001)
autocorrelation = [cross_correlation(data, data, tau, len(data)) for tau in taus]

# построение графика АКФ
plt.figure(figsize=(10, 6))
plt.plot(taus, autocorrelation)
plt.title("Автокорреляционная функция сигнала")
plt.xlabel("Сдвиг (tau)")
plt.ylabel("Автокорреляция")
plt.grid(True)
plt.show()