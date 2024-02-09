# Лабораторная 6.1
# Программа должна реализовывать следующий функционал:
# 1 Покадровое получение видеопотока с камеры. Использовать камеру ноутбука, вебкамеру или записать видео файл с вебкамеры товарища и использовать его.
# 2 Реализовать обнаружение движения в видеопотоке: попарно сравнивать текущий и предыдущий кадры. (Если вы сможете в более сложный алгоритм, устойчивый к шумам вебкамеры - будет совсем хорошо)
# 3 По мере проигрывания видео в отдельном окне отрисовывать двухцветную карту с результатом: красное - есть движение, зелёное - нет движения
# 4 Добавить таймер, по которому включается и выключается обнаружение движения. О текущем режиме программы сообщать текстом с краю изображения: “Красный свет” - движение обнаруживается, “Зелёный свет” - движение не обнаруживается.
import cv2
import numpy as np
import time

camera = cv2.VideoCapture(0)
_, frame1 = camera.read()
_, frame2 = camera.read()

frame_color = np.zeros((150, 150, 3), dtype=np.uint8)
green = (0, 255, 0)
red = (0, 0, 255)
end_time = 0
prepare_time = 4
what_color = False
# fr_green =
frame_color_size = 150
start_time = time.time()
# start_interval_time = 0
interval_time = 0
# color = green
flag = True
while True:
    interval_time = time.time() - start_time
    # cv2.putText(frame1, 'Movement recorded', (10, 30), cv2.FONT_HERSHEY_TRIPLEX, 1, green, 2)
    # ------------------------------------------------------------------------------------------------------------------
    diff = cv2.absdiff(frame1, frame2)  # распознание движения - нахождение разницы между двумя кадрами
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)  # серый цвет

    blur = cv2.GaussianBlur(gray, (5, 5), 0)  # фильтрация лишних контуров

    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    # _, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # нахождение контуров
    cv2.drawContours(frame1, contours, -1, (0, 255, 0))  # отрисовка найденных контуров
    # ------------------------------------------------------------------------------------------------------------------
    # while True:
    #     # cv2.rectangle(frame_color, (0, 0), (frame_color_size - 1, frame_color_size - 1), white_color, -1)
    #     if len(contours) > 0:
    #         cv2.putText(frame1, 'Movement recorded', (10, 30), cv2.FONT_HERSHEY_TRIPLEX, 1, green, 2)
    #         # cv2.imshow('status', fr_green)
    #     else:
    #         cv2.putText(frame1, 'There is no movement', (10, 30), cv2.FONT_HERSHEY_TRIPLEX, 1, red, 2)
    #         # cv2.imshow('status', fr_red)

    # 3
    if len(contours) > 0:
        # cv2.putText(frame1, 'Movement recorded', (10, 30), cv2.FONT_HERSHEY_TRIPLEX, 1, green, 2)
        # cv2.imshow('status', fr_green)
        # frame_color = cv2.rectangle(frame_color, (1, 1), (150, 150), green, 200)
        color_res = red
    else:
        # cv2.putText(frame1, 'There is no movement', (10, 30), cv2.FONT_HERSHEY_TRIPLEX, 1, red, 2)
        # frame_color = cv2.rectangle(frame_color, (1, 1), (150, 150), red, 200)
        # cv2.imshow('status', fr_red)
        color_res = green

    if interval_time > 0 and flag is True:
        # color = red
        cv2.putText(frame1, 'GREEN', (10, 30), cv2.FONT_HERSHEY_TRIPLEX, 1, green, 2)
        # start_time += interval_time
        # interval_time = 0
        # interval_time = time.time()
    if interval_time > 4:
        flag = False
        cv2.putText(frame1, 'RED', (10, 30), cv2.FONT_HERSHEY_TRIPLEX, 1, red, 2)
        # color = green
    if interval_time > 8:
        flag = True
        start_time += interval_time
    frame_color = cv2.rectangle(frame_color, (1, 1), (150, 150), color_res, 200)
    cv2.imshow('FRAME_COLOR', frame_color)
    # cv2.waitKey(300)
    print("start_time:", start_time, '       interval_time:', interval_time)
    # ------------------------------------------------------------------------------------------------------------------
    cv2.imshow('Frame', frame1)
    frame1 = frame2
    _, frame2 = camera.read()

    if cv2.waitKey(20) == 27:
        break

camera.release()
cv2.destroyAllWindows()
