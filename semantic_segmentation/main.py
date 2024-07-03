# from ultralytics import YOLO
#
# # Load a model
# model = YOLO('yolov8n.yaml')  # build a new model from YAML
# model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
# model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights
#
# # Train the model
# results = model.train(data='coco128.yaml', epochs=100, imgsz=640)


from ultralytics import YOLO
import cv2

# обучение
model = YOLO('yolov8n-seg.pt', task='detect')
model.info()
print(model.names)
results = model.train(data='"C:\Users\Makovey\PycharmProjects\PAC 1\data.yaml"', epochs=2, imgsz=640)
# конец обучения

# import cv2
#
# model = YOLO('TOSTER.pt')  # подгружаю свою модель
# # проверка на пикче
# res = model.track('тостер1.jpg', persist=True, conf=0.25)
# # res = model.predict('тостер1.jpg')
# img = res[0].plot()
# # print(res[0])
# # print(res[0].boxes)
# while True:
#     cv2.imshow('image', img)
#     if cv2.waitKey(1) & 0xFF == ord('a'):
#         break



import numpy as np
# model = YOLO('GOVNO.pt')
#
# # load video
# video_path = 'toaster_test.mp4'
# cap = cv2.VideoCapture(video_path)
#
# ret = True
# # read frames
#
# # Определите параметры видео
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fps = int(cap.get(cv2.CAP_PROP_FPS))
#
# # Создайте объект VideoWriter
# out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
#
# # i = 0
# # while ret:
# #     ret, frame = cap.read()
# #
# #     if (i % 3 == 0):
# #         i = 0
# #         if ret:
# #             results = model.track(frame, persist=True)
# #
# #             # plot results
# #             # cv2.rectangle
# #             # cv2.putText
# #
# #             image = results[0].plot()
# #             out.write(image)
# #
# #             cv2.imshow('frame', image)
# #             if cv2.waitKey(1) & 0xFF == ord('a'):
# #                 break
# #         else:
# #             break
# #     i += 1
# #
# # cap.release()
# # out.release()
