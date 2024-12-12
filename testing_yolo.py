import os
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

def test_yolo_on_images(model_path, images_dir, conf_threshold=0.5):
    # Load the trained model
    model = YOLO(model_path)

    # Get the list of image files in the directory
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

    # Loop through each image
    for image_file in image_files:
        try:
            image_path = os.path.join(images_dir, image_file)


            # Perform inference on the image with custom thresholds
            # results = model(image_path, conf=conf_threshold, iou=iou_threshold)
            results = model.predict(image_path, conf=conf_threshold, iou=0.3)

            # Since results is a list, access the first (and only) item
            result = results[0]

            # Visualize the results
            # Load the original image
            img = cv2.imread(image_path)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            #
            #
            # # Resize the image
            # img = cv2.resize(img, new_size)

            # Get the results and draw them on the image
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
        except:
            print('some input issue!')

# Define paths
model_path = "/home/fariborz_taherkhani/Joint_Pose_Segment_Yolo-master/runs/pose_segment/train/weights/best.pt"  # Update this path based on your training run
images_dir = "/home/fariborz_taherkhani/Joint_Pose_Segment_Yolo-master/DataSet/train/images"  # Update this to the directory containing your images

# Run the test with custom thresholds
test_yolo_on_images(model_path=model_path, images_dir=images_dir, conf_threshold=0.5)
