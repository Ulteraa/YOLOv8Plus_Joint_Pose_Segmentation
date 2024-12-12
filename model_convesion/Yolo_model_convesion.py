# from monai.utils import export
# from onnx_graphsurgeon import export_onnx
from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2
model_path = 'runs/pose_segment/train/weights/last.pt'
model = YOLO(model_path, task='pose_segment') # load a custom trained model
model.export(format="engine", imgsz=(1088, 1920))  # creates 'yolov8n.torchscript'
