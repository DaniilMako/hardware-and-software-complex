# Семинар 4 от 26.9.23
import numpy as np  # для работы с матрицами

# 1. Отсортировать значения массива по частоте встречания.
a = np.random.randint(5, size=10)
print('Исходный массив', a)
unique_elements, frequency = np.unique(a, return_counts=True)
sorted_by_freq = unique_elements[np.argsort(frequency)[::-1]]
print('Отсортирован по частоте:', sorted_by_freq, '\n')

# 2. Дана картинка высоты h, ширины w, тип данных np.uint8 (от 0 до 255). Найти количество уникальных цветов.
h, w = np.random.randint(1, 8, dtype=np.uint8), np.random.randint(1, 8, dtype=np.uint8)
print('h:', h, '   w:', w)
image = np.random.randint(256, size=(h, w))
print('image:\n', image)
print('Уникальные цвета:\n', np.unique(image))
print('Количество уникальных цветов:', len(np.unique(image)), '\n')

# 3. Написать функцию, вычисляющую плавающее среднее вектора
import matplotlib.pyplot as plt  # для работы с графиком


def moving_average(vector, window):
    res = np.empty(len(vector))  # пустой массив длинной массива vector
    for i in range(len(vector)):
        res[i] = sum(vector[i:i + window]) / window
        # print('vector[i:i+window] =', vector[i:i+window])
    print('vector = ', *vector)
    print('res = ', *res)
    plt.plot(vector)  # график по vector
    plt.plot(res)  # график по res
    plt.show()  # вывод графика


vector = np.random.randint(-2, 2, size=30)
window = 3
moving_average(vector, window)

# 4. Дана матрица (n, 3). Вывести те тройки чисел, которые являются длинами сторон треугольника
n = 4
matrix = np.random.randint(10, size=(n, 3))
print('\nmatrix:\n', matrix)

for row in matrix:
    if row[0] + row[1] > row[2] and row[0] + row[2] > row[1] and row[1] + row[2] > row[0]:
        print('Треугольник:', row)
