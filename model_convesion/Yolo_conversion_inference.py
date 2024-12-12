import matplotlib.pyplot as plt
from ultralytics import YOLO
import cv2
import  os
# Load the exported TorchScript model
torchscript_model = YOLO('runs/pose_segment/train/weights/last.engine', task='pose_segment')
images_dir = '/home/fariborz_taherkhani/Downloads/yolo_conversion/test_images'
image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

# Loop through each image
for image_file in image_files:
    image_path = os.path.join(images_dir, image_file)
    img = cv2.imread(image_path )
    results = torchscript_model(img, imgsz=(1088, 1920), task='pose')
    result = results[0]

    annotated_img = result.plot()

    # Display the original and annotated images
    plt.figure(figsize=(12, 8))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(img)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("YOLO Predictions")
    plt.imshow(annotated_img)
    plt.axis('off')

    plt.show()


# #torchscript_model = YOLO("runs/pose_segment/train/weights/last.torchscript", task='pose_segment')
# img = '/home/fariborz_taherkhani/Downloads/yolo_conversion/test_images/left_650482_20231010T014533.jpg'
# # # Run inference
# img = cv2.imread(img)
# results = torchscript_model(img, imgsz=(1088, 1920))
# result = results[0]
# # masks = result.masks.xy
# # image_shape = img.shape[:2]
# # for i in range(len(masks)):
# #     mask_np = masks[i]
# #     plot_polygon_mask( mask_np, image_shape)
# #
# #
# #
# #
# #     plt.show()
#
#
#
# # Visualize the results
# # Load the original image
#
#
# # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# #
# #
# # # Resize the image
# # img = cv2.resize(img, new_size)
#
# # Get the results and draw them on the image
# annotated_img = result.plot()
#
# # Display the original and annotated images
# plt.figure(figsize=(12, 8))
# plt.subplot(1, 2, 1)
# plt.title("Original Image")
# plt.imshow(img)
# plt.axis('off')
#
# plt.subplot(1, 2, 2)
# plt.title("YOLO Predictions")
# plt.imshow(annotated_img)
# plt.axis('off')
#
# plt.show()
#
# # Define paths