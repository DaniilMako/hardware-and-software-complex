# Лабораторная 7.1
# Ловлю призраков необходимо реализовать через поиск и сопоставление ключевых точек на изображениях. Алгоритм должен состоять из следующих шагов:
# - Загрузка изображения, на котором необходимо осуществлять поиск;
# - Загрузка изображения(ий) призраков;
# - Обнаружить на них ключевые точки и вычислить для них любые понравившиеся вам дескрипторы SIFT, SURF, ORB;
# - Сопоставить точки шаблона (призрака) с точками изображения через Brute-Force Matching или FLANN Matching и найти какой области соответстветствует призрак;
# - Найти гомографию используя алгоритм RANSAC. Выделить призрака на изображение рамкой.
# Ключевые слова для поиска в Google и документации OpenCV: findHomography, RANSAC, SIFT_Create, FlannBasedMatcher.
# ЛР 7.1: нужно поймать одного призрака
import cv2
import numpy as np


def find_ghost(image_with_ghosts, ghost_image):
    image_with_ghosts = cv2.imread(image_with_ghosts)
    ghost_image = cv2.imread(ghost_image)

    # Обнаружить на них ключевые точки и вычислить для них любые понравившиеся вам дескрипторы SIFT, SURF, ORB;
    sift = cv2.SIFT_create()
    # Нахождение ключевых точек и дескрипторов для основного изображения
    image_kp, image_dp = sift.detectAndCompute(image_with_ghosts, None)
    # Нахождение ключевых точек и дескрипторов для изображения призрака
    ghost_kp, ghost_dp = sift.detectAndCompute(ghost_image, None)

    # Сопоставление точек шаблона (призрака) с точками изображения через Brute-Force Matching
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
    matches = bf.match(ghost_dp, image_dp)
    matches = np.array(matches)

    # Извлечение точек призрака и соответствующих точек изображения
    ghost_pts = np.array([ghost_kp[m.queryIdx].pt for m in matches])
    image_pts = np.array([image_kp[m.trainIdx].pt for m in matches])
    # Нахождение гомографии с помощью алгоритма RANSAC
    homography, _ = cv2.findHomography(ghost_pts, image_pts, cv2.RANSAC, 5.0)

    # Применение гомографии к координатам углов призрака
    ghost_corners = np.float32([[0, 0], [0, ghost_image.shape[0]], [ghost_image.shape[1], ghost_image.shape[0]],
                                [ghost_image.shape[1], 0]]).reshape(-1, 1, 2)  # координаты углов призрака
    # преобразование углов
    transformed_corners = cv2.perspectiveTransform(ghost_corners, homography)

    # Отображение рамки вокруг призрака на искомом изображении
    image_with_ghosts = cv2.polylines(image_with_ghosts, [np.int32(transformed_corners)], True, (0, 255, 0), 4)

    image_with_ghosts = cv2.resize(image_with_ghosts, (1000, 500))
    cv2.imshow("Ghosts", image_with_ghosts)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


image_path = 'lab7.png'
ghost_image_paths = 'pampkin_ghost.png'
find_ghost(image_path, ghost_image_paths)
