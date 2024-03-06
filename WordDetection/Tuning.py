from ultralytics import YOLO

model = YOLO('./yolov8x.pt')

model.tune(data='base.yaml', epochs=30, iterations=30, optimizer='AdamW', plots=False, save=False, val=False)
