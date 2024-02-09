# Семинар 12 от 28.11.23
# 1. Написать функцию, переводящую изображение в матрицу столбцов - im2col(). На вход функция принимает изображение и размер свёртки, возвращает столбцы.
import cv2
import numpy as np

print('\n>>>>>>>>>>>>>>>> ЗАДАНИЕ 1 >>>>>>>>>>>>>>>>\n')

img = cv2.imread('frog32.jpg')
img = img[28:, 28:, :]  # берется срез для удобной проверки, но если не брать, то лучше закомментить print'ы матриц
print('img.shape ', img.shape)
print('img:\n', img, sep='\n')

# def my_func():  # для 3 задания

def moving(arr, len_p):
    a = []
    for i in range(3):
        for j in range(i, len_p, 3):
            a = np.append(a, arr[j])
    # print(a)
    return a


def im2col(image, kernel):
    image_height, image_width, num_channels = image.shape
    kernel_height, kernel_width = kernel.shape

    output_height = image_height - kernel_height + 1
    output_width = image_width - kernel_width + 1

    col_matrix = np.zeros((output_height * output_width, kernel_height * kernel_width * num_channels))

    for w in range(output_width):
        for h in range(output_height):
            patch = image[h:h + kernel_height, w:w + kernel_width, :].reshape(-1)
            # print(patch, '\n')  # список вида [220 222 223 209 213 218 206 208 216 187 194 203] содержит сразу 3 канала по n пикселей
            # в данном примере 4 пикселя х 3 = 12 в одном таком patch
            patch = moving(patch,
                           len(patch))  # функция для упорядочивания значений (переставляет значения так, чтобы сначала шли n пикселей одного канала, затем 2го и 3го)
            col_matrix[h * output_width + w, :] = patch
    col_matrix = list(zip(*col_matrix))
    col_matrix = np.array(col_matrix)
    return col_matrix


core = np.array([[1, 2],
                 [3, 4]])

res = im2col(img, core)

print('\nres.shape:', res.shape)
print('res:\n', res)

# 2. Написать функцию свёртки, которая работает без циклов. Вместо циклов, она использует im2col(), для перевода изображения в набор столбцов.
print('\n>>>>>>>>>>>>>>>> ЗАДАНИЕ 2 >>>>>>>>>>>>>>>>\n')


def convolution(image, core2):
    vector = core2.reshape(1, -1)  # преобразование ядра в вектор
    print(vector)
    print(np.tile(vector, 3))
    vector = np.tile(vector, 3)  # увеличени длины вектора для корректоного умножения матрицы на вектор
    vector = vector.reshape(1, -1)
    print(vector.shape, res.shape)
    return vector.dot(res)  # используется res из предыдущей задачи, т.к. матрица одна и та же


res2 = convolution(img, core)
print(res2.shape)
print('convolution:\n', res2)

# 3. Сравнить результаты с torch.nn.Conv2d
print('\n>>>>>>>>>>>>>>>> ЗАДАНИЕ 3 >>>>>>>>>>>>>>>>\n')
import torch
from PIL import Image
import torchvision.transforms as T


def func():
    # (in_channels, out_channels, kernel_size)
    conv = torch.nn.Conv2d(3, 3, (5 * 5))
    # Read input image
    img = Image.open('frog32.jpg')  # RGB
    # convert image to torch tensor
    img = T.ToTensor()(img)
    print("img size:", img.size())
    # CNN operation on image
    img = conv(img)
    # convert image to PIL(pillow library) image
    # it is the pillow supported library to handle image
    # img = T.ToPILImage()(img)
    # disply image
    # img.show()

# func()

# import time
#
# start_time = time.time()
# my_func()
# end_time = time.time()
# print("Время работы моей реализации:\n", end_time - start_time)
# #  0.08945703506469727
# start_time = time.time()
# func()
# end_time = time.time()
# print("Время работы torch.nn.Conv2d:\n", end_time - start_time)
# #  0.01331782341003418
