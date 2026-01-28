from ultralytics import YOLO
model = YOLO("yolov8n.pt")
# Train the model with MPS
results = model.train(data="coco8.yaml", epochs=100, imgsz=640, device="mps")