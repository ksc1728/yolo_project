from ultralytics import YOLO
import warnings

# Build a new model from scratch
model = YOLO("yolov8n.yaml")

# Load a pretrained model (recommended for training)
model = YOLO("yolov8n.pt")

# Train the model
model.train(data="dataset/config.yaml", epochs=15)

warnings.filterwarnings('ignore')

