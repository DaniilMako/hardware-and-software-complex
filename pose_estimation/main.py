from ultralytics import YOLO
import cv2

model = YOLO('yolov8n-pose.pt')  # подгружаю модель
# проверка на пикче
img = cv2.imread('stand1.jpg')
img = cv2.resize(img, (640, 640))
# res = model.track(img, persist=True, conf=0.25)
res = model.predict(img, conf=0.25)
img = res[0].plot()
print()
# print(res)
# print(res[0].boxes)

while True:
    cv2.imshow('image', img)
    if cv2.waitKey(1) & 0xFF == ord('a'):
        break

from ultralytics import YOLO

# Load a model
# model = YOLO('yolov8n-pose.yaml')  # build a new model from YAML
# model = YOLO('yolov8n-pose.pt')  # load a pretrained model (recommended for training)
# model = YOLO('yolov8n-pose.yaml').load('yolov8n-pose.pt')  # build from YAML and transfer weights

# Train the model
# results = model.train(data='coco8-pose.yaml', epochs=100, imgsz=640)
