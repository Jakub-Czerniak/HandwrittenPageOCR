from ultralytics import YOLO

model = YOLO('./yolov8x.pt')

model.tune(data='./base.yaml', epochs=30, iterations=1, optimizer='AdamW', plots=False, save=False, val=False)
