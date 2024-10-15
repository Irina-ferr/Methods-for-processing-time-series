# Из-за несоблюдения особенностей технологического процесса внутри 
# объемной стальной детали образовался дефект. 
# Излучатель ультразвукового дефектоскопа генерирует гармонические импульсы с частотой 5 МГц. 
# Тестирующий сигнал отражается от дефекта и регистрируется приемником дефектоскопа. 
# Излученный и принятый сигнал были оцифрованы с частотой дискретизации 100 МГц 
# и сохранены в файлы 
# “UStransmitter.txt” и “USreceiver.txt” 
# (принятый после отражения от дефекта сигнал значительно затух и зашумлен) соответственно. 
# Необходимо определить, на какой глубине, расположен дефект, полагая, 
# что скорость звука в стали используемой марки 5 км/c.  
import numpy as np
import matplotlib.pyplot as plt

def cross_correlation(X, Y, tau):
    """вычисление кросс-корреляции между двумя сигналами."""
    N = len(X)
    if tau < 0:
        X_segment = X[-tau:N]
        Y_segment = Y[0:N + tau]
    else:
        X_segment = X[0:N - tau]
        Y_segment = Y[tau:N]

    mean_X = np.mean(X_segment)
    mean_Y = np.mean(Y_segment)

    std_X = np.std(X_segment)
    std_Y = np.std(Y_segment)

    covariance = np.sum((X_segment - mean_X) * (Y_segment - mean_Y))
    correlation = covariance / ((len(X_segment) - 1) * std_X * std_Y)

    return correlation

def find_time_delay(transmitter_signal, receiver_signal, max_tau):
    """поиск временной задержки между излученным и принятым сигналом."""
    correlations = [cross_correlation(receiver_signal, transmitter_signal, tau) for tau in range(max_tau)]
    
    # находим индекс максимальной корреляции
    time_delay_index = np.argmax(correlations)
    
    return time_delay_index

# загрузка данных
transmitter_signal = np.loadtxt("UStransmitter.txt")
receiver_signal = np.loadtxt("USreceiver.txt")

# параметры
sampling_frequency = 100e6  # Частота дискретизации 100 МГц
speed_of_sound = 5000  # скорость звука в км/с
max_tau = 2000  # максимальное значение для задержки

# поиск временной задержки
time_delay_index = find_time_delay(transmitter_signal, receiver_signal, max_tau)

# расчет времени задержки в секундах
time_delay_seconds = time_delay_index / sampling_frequency

# расчет глубины дефекта = скорость звука × время / 2
depth = (speed_of_sound * time_delay_seconds) / 2  # делим на 2, так как сигнал идет туда и обратно

# вывод результатов
print(f"Временная задержка: {time_delay_seconds:.8f} секунд")
print(f"Глубина дефекта: {depth:.2f} метров")

# график кросс-корреляции
plt.figure(figsize=(10, 6))
plt.plot(range(max_tau), [cross_correlation(receiver_signal, transmitter_signal, tau) for tau in range(max_tau)])
plt.title('Кросс-корреляция между излученным и принятым сигналом')
plt.xlabel('Задержка (такты)')
plt.ylabel('Кросс-корреляция')
plt.grid()
plt.show()
