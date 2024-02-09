# Семинар 12 от 28.11.23
# Задания
# 1. Написать функцию, переводящую изображение в матрицу столбцов - im2col(). На вход функция принимает изображение и размер свёртки, возвращает столбцы.
print('\n>>>>>>>>>>>>>>>> ЗАДАНИЕ 1 >>>>>>>>>>>>>>>>\n')
import cv2
import numpy as np


img = cv2.imread('frog32.jpg')
img = img[28:, 28:, :]
print('img.shape ', img.shape)

print('img:\n', img, sep='\n')


def my_func():  # для 3 задачи
    pass


def copy_n_remove(arr, len_p):
    a = []
    for i in range(3):
        for j in range(i, len_p, 3):
            a = np.append(a, arr[j])
    print(a)
    return a


def im2col(image, kernel):
    image_height, image_width, num_channels = image.shape
    kernel_height, kernel_width = kernel.shape
    # print('kernel_height', kernel_height, 'kernel_width', kernel_width)
    output_height = image_height - kernel_height + 1
    output_width = image_width - kernel_width + 1
    # print('output_height', output_height, 'output_width',  output_width)
    col_matrix = np.zeros((output_height * output_width, kernel_height * kernel_width * num_channels))

    for w in range(output_width):
        for h in range(output_height):
            patch = image[h:h + kernel_height, w:w + kernel_width, :].reshape(-1)
            print(patch, '\n')
            patch = copy_n_remove(patch, len(patch))
            col_matrix[h * output_width + w, :] = patch
    col_matrix = list(zip(*col_matrix))
    col_matrix = np.array(col_matrix)
    return col_matrix


# def im2col(image, core, stride=1, pas=0):
#     rows_core, cols_core = core.shape
#     new_image = []
#     # print('len(image.shape) ', len(image.shape))
#     if len(image.shape) == 3:
#         rows, cols, depth = image.shape
#         # print(rows, cols, depth)
#         print(cols - cols_core + 1, rows - rows_core + 1, depth)
#         for d in range(depth):
#             for i in range(cols - cols_core):
#                 for j in range(rows - rows_core):
#                     col = image[i:i + rows_core, j:j + cols_core, :].flatten()
#                     print(image[i:i + rows_core, j:j + cols_core, :].flatten(), '\n')
#                     new_image.append(col)
#     else:
#         rows, cols = image.shape
#         print(rows - rows_core + 1, cols - cols_core + 1)
#         for i in range(rows - rows_core + 1):
#             for j in range(cols - cols_core + 1):
#                 print(image[i:i + rows_core, j:j + cols_core].flatten('F'), '\n')
#                 col = image[i:i + rows_core, j:j + cols_core].flatten()
#                 # print(col)
#                 new_image.append(col)
#     # print(*new_image, sep='\n')
#     new_image = list(zip(*new_image))
#     # print(*new_image, sep='\n')
#     new_image = np.array(new_image)
#     return new_image


img2 = np.array([[1, 2, 3, 4, 5],
                 [6, 7, 8, 9, 10],
                 [11, 12, 13, 14, 15],
                 [16, 17, 18, 19, 20]])

core = np.array([[1, 2],
                 [3, 4]])
# x, y = core.shape
res = im2col(img, core)

print('res.shape ', res.shape)
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
# # 0.006010293960571289
# start_time = time.time()
# func()
# end_time = time.time()
# print("Время работы torch.nn.Conv2d:\n", end_time - start_time)
# # 0.009354352951049805
