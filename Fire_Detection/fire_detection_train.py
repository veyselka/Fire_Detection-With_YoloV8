import os
from ultralytics import YOLO

# data.yaml dosyasının tam yolu
data_yaml_path = os.path.join(os.path.dirname(__file__), "data.yaml")

# YOLOv8 modelini yükle (istenirse yolov9'a güncelleyebilirsin)
model = YOLO("yolov8n.pt")  # veya YOLO("yolov9n.pt") eğer varsa

# Modeli eğit
model.train(
    data=data_yaml_path,
    epochs=50,
    imgsz=640,
    batch=16,
    name="fire-detection-v9"
)
