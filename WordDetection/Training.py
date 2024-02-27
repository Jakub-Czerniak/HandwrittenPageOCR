from ultralytics import YOLO

model = YOLO('yolov8x.pt')

#python train.py --img 1088 --batch 8 --epochs 150 --data ./data/ocr_handwriting_data.yaml --weights [string]::Empty --cfg yolov5s.yaml --name yolov5s_results
results = model.train(data='base.yaml', epochs=1, imgsz=1088, batch=8)
