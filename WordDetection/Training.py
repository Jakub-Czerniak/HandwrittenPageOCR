from ultralytics import YOLO

model = YOLO('yolov8x.pt')

#python train.py --img 1088 --batch 8 --epochs 150 --data ./data/ocr_handwriting_data.yaml --weights [string]::Empty --cfg yolov5s.yaml --name yolov5s_results
results = model.train(data='augmented.yaml', epochs=200, imgsz=1088, batch=8, hsv_h=0, hsv_s=0, hsv_v=0,
                      degrees=0, translate=0, scale=0, shear=0, perspective=0, flipud=0, fliplr=0,
                      mosaic=0, mixup=0, copy_paste=0, erasing=0)

results = model.train(data='preprocessed.yaml', epochs=200, imgsz=1088, batch=8, hsv_h=0, hsv_s=0, hsv_v=0,
                      degrees=0, translate=0, scale=0, shear=0, perspective=0, flipud=0, fliplr=0,
                      mosaic=0, mixup=0, copy_paste=0, erasing=0)

results = model.train(data='base.yaml', epochs=200, imgsz=1088, batch=8, hsv_h=0, hsv_s=0, hsv_v=0,
                      degrees=0, translate=0, scale=0, shear=0, perspective=0, flipud=0, fliplr=0,
                      mosaic=0, mixup=0, copy_paste=0, erasing=0)
