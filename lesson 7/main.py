# 1. Для датасета Nails segmentation создать генератор, который на каждой итерации возвращает пару списков из заданного количества (аргумент функции) изображений и масок к ним (итератор должен перемешивать примеры).
# 2. Добавить в генератор случайную аугментацию (каждая применяется случайно). После преобразований все изображения должны иметь одинаковый размер. Обратите внимание, что большинство преобразований должны применяться одинаково к изображению и маске
# A. Поворот на случайный угол
# B. Отражение по вертикали, горизонтали
# C. Вырезание части изображения
# D. Размытие

import os
import cv2
import pandas as pd
import numpy as np


# ЗАДАЧА 2
def random_aug(img, lbl, rnd):  # rnd - случайное значение от 0 до 3
    print(rnd)
    img = cv2.imread('images/' + img)  # считываю изображение
    lbl = cv2.imread('labels/' + lbl)  # считываю маску
    if rnd == 0:  # A. Поворот на случайный угол
        height, width = img.shape[:2]
        center = (width / 2, height / 2)
        rotate = cv2.getRotationMatrix2D(center, np.random.randint(0, 360), 1.0)
        img = cv2.warpAffine(src=img, M=rotate, dsize=(width, height))
        lbl = cv2.warpAffine(src=lbl, M=rotate, dsize=(width, height))
    if rnd == 1:  # B. Отражение по вертикали, горизонтали
        img = img[::-1]  # сначала поворот по вертикали
        lbl = lbl[::-1]
        img = cv2.flip(img, 1)  # потом - по горизонтали
        lbl = cv2.flip(lbl, 1)
    if rnd == 2:  # C. Вырезание части изображения
        np.random.seed(43)
        bbox = np.random.randint(0, 128, size=4)  # x, y, w, h
        img = img[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
        lbl = lbl[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
    if rnd == 3:  # D. Размытие
        img = cv2.blur(img, (2, 2))
        lbl = cv2.blur(lbl, (2, 2))
    img_lbl = np.concatenate((img, lbl), axis=1)
    cv2.imshow('image and label', img_lbl)
    cv2.waitKey(0)
    return img, lbl


# ЗАДАЧА 1
def generator(num):  # num - количество случайных строк из таблицы
    df_new = df.sample(num)
    print(df_new)
    lst_images = list(df_new['images'])
    lst_labels = list(df_new['labels'])
    print('lst_images:', lst_images, 'lst_labels:', lst_labels,
          sep='\n')  # вывожу/возвращаю пару списков из заданного количества
    for i in range(len(lst_images)):  # случайная аугментация
        lst_images[i], lst_labels[i] = random_aug(lst_images[i], lst_labels[i], np.random.randint(0, 4))
    return lst_images, lst_labels


list_images = os.listdir('images')
list_labels = os.listdir('labels')
d = {'images': list_images, 'labels': list_labels}
df = pd.DataFrame(data=d)  # 1й столбец - изображение, 2й - маска, 3й - флаг (было ли уже это изображение)

list_images, list_labels = generator(5)
