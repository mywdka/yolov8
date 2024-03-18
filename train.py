from ultralytics import YOLO

model = YOLO("yolov8n.yaml")  # build a new model from scratch
model.train(data="config.yaml", epochs=100)  # train the model