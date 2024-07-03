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

# # обучение
model = YOLO('yolov8n.pt')
# # Display model information (optional)
# model.info()
# # Train the model on the COCO8 example dataset for 100 epochs
# results = model.train(data='data.yaml', epochs=10, imgsz=640)
# # конец обучения

model = YOLO('best.pt')  # подгружаю свою модель
# проверка на пикче
res = model.track('plain.png', persist=True)
img = res[0].plot()
while True:
    cv2.imshow('plain', img)
    if cv2.waitKey(1) & 0xFF == ord('a'):
        break
