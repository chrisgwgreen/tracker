from ultralytics import YOLO

# Load a YOLO26n PyTorch model
model = YOLO("yolo26n.pt")

# Run inference
results = model("tcp://127.0.0.1:8888")