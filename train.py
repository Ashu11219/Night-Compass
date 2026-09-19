from ultralytics import YOLO

model = YOLO('yolov8n.pt')
model.train(data='configs/thermal.yaml', epochs=5, device=0, imgsz=640, batch=16)