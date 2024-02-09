# Семинар 6 от 10.10.23
import numpy as np  # нужен только для конкатенации
import cv2
import os  # для работы с файлами

# 1. Для данных Nails segmentation объедините пары изображение-маска (список файлов получить с помощью библиотеки os название парных изображений совпадают)
# 2. Выведите по очереди пары с помощью OpenCV эти пары (переключение по нажатию клавиши)
for file in os.listdir('images'):
    a = cv2.imread('images/' + file)  # считываем изображение
    b = cv2.imread('labels/' + file)  # считываем маску
    ab = np.concatenate((a, b), axis=1)  # объединяем в одно окно
    cv2.imshow('Tasks 1 и 2', ab)
    cv2.waitKey(0)

# 3. Выделите контуры на масках и отрисуйте их на изображениях
for file in os.listdir('images'):
    image = cv2.imread('images/' + file)
    label = cv2.imread('labels/' + file)
    _, thresh = cv2.threshold(label, 200, 255, cv2.THRESH_BINARY)  # обрабатываем изображения

    gray = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)  # преобразуем изображение в оттенки серого
    contours, _ = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # ищем контуры

    image = cv2.drawContours(image, contours, -1, (255, 0, 0), 2)  # рисуем контуры
    cv2.imshow('Task 3', image)
    cv2.waitKey(0)

# 4. Воспроизведите любой видеофайл с помощью OpenCV в градациях серого
video = cv2.VideoCapture('video.mp4')  # считываем видео
while True:
    _, img = video.read()
    gray_video = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # используем фильтр серого
    cv2.imshow("Task 4", gray_video)
    cv2.waitKey(10)
