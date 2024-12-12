import torch
from ultralytics import YOLO

def train_yolo(data_config, model_path='yolov8x-seg.pt', epochs=1, batch_size=4, nms_iou_threshold=0.25):
    model_path = 'yolov8-seg.yaml'
    torch.cuda.empty_cache()
    # Load the model and move it to the specified device
    model = YOLO(model_path, task='pose_segment')
    for name, param in model.model.named_parameters():
        print(name, param.shape)
        param.requires_grad = True

    # # Define the backbone layers to freeze
    # backbone_layers = [
    #     "model.0.", "model.1.", "model.2.", "model.3.",
    #     "model.4.", "model.5.", "model.6.", "model.7.",
    #     "model.8.", "model.9."
    # ]
    #
    # # Freeze the backbone layers
    # for name, param in model.model.named_parameters():
    #     if any(name.startswith(layer) for layer in backbone_layers):
    #         param.requires_grad = False
    #         print(f'Layer frozen: {name}')

    # Verify frozen layers
    for name, param in model.model.named_parameters():
        if not param.requires_grad:
            print(f'Layer frozen: {name}')

    # Train the model
    model.train(data=data_config, epochs=epochs, batch=batch_size, imgsz=1920, rect=True, freeze=True)

# Create YOLO data configuration file
data_config_path = "segmentation.yaml"

# Train YOLO model
train_yolo(data_config=data_config_path)
