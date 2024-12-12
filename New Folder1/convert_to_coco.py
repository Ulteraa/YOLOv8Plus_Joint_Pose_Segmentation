import json
# address='/home/fariborz_taherkhani/Joint_Yolo/val_annotations_by_image_json/000000000885.jpg.json'
# from PIL import Image, ImageDraw
#
# file = open(address, 'r')
#
# data = json.load(file)
# print('finished')

# import json
# import os
#
# # Paths
# input_json_path = "coco/annotations/person_keypoints_train2017.json"  # Input COCO JSON file
# output_dir = "train_annotations_yolo_format"  # Directory to save YOLO format text files
#
# # Create the output directory if it doesn't exist
# os.makedirs(output_dir, exist_ok=True)
#
# # Load the COCO annotation file
# with open(input_json_path, "r") as f:
#     coco_data = json.load(f)
#
# # Create a mapping from image ID to image metadata
# image_metadata = {img["id"]: {"file_name": img["file_name"], "height": img["height"], "width": img["width"]}
#                   for img in coco_data["images"]}
#
# # Helper function to normalize a value
# def normalize(value, scale):
#     return value / scale
#
# # Convert and save annotations for each image
# for annotation in coco_data["annotations"]:
#     image_id = annotation["image_id"]
#     metadata = image_metadata[image_id]
#     img_width, img_height = metadata["width"], metadata["height"]
#
#     # Convert bounding box from [x, y, w, h] to [center_x, center_y, normalized_width, normalized_height]
#     bbox = annotation["bbox"]
#     x, y, w, h = bbox
#     center_x = normalize(x + w / 2, img_width)
#     center_y = normalize(y + h / 2, img_height)
#     norm_w = normalize(w, img_width)
#     norm_h = normalize(h, img_height)
#     yolo_bbox = f"{center_x} {center_y} {norm_w} {norm_h}"
#
#     # Convert keypoints (normalize and retain visibility)
#     keypoints = annotation.get("keypoints", [])
#     yolo_keypoints = []
#     for i in range(0, len(keypoints), 3):
#         kp_x = normalize(keypoints[i], img_width)
#         kp_y = normalize(keypoints[i + 1], img_height)
#         visibility = keypoints[i + 2]
#         yolo_keypoints.append(f"{kp_x} {kp_y} {visibility}")
#     yolo_keypoints = " ".join(yolo_keypoints)
#
#     # Convert segmentation points (normalize coordinates)
#     segmentation = annotation.get("segmentation", [])
#     yolo_segmentation = []
#     for seg in segmentation:
#         for i in range(0, len(seg), 2):
#             try:
#                 seg_x = normalize(float(seg[i]), img_width)
#                 seg_y = normalize(float(seg[i + 1]), img_height)
#                 yolo_segmentation.append(f"{seg_x} {seg_y}")
#             except (ValueError, TypeError):
#                 print(f"Skipping invalid segmentation point: {seg[i]}, {seg[i + 1]} in image ID {image_id}")
#     yolo_segmentation = " ".join(yolo_segmentation)
#
#     # yolo_segmentation = []
#     # for seg in segmentation:
#     #     for i in range(0, len(seg), 2):
#     #         seg_x = normalize(seg[i], img_width)
#     #         seg_y = normalize(seg[i + 1], img_height)
#     #         yolo_segmentation.append(f"{seg_x} {seg_y}")
#     # yolo_segmentation = " ".join(yolo_segmentation)
#
#     # Create the YOLO format string
#     annotation_id = annotation["id"]
#     yolo_annotation = f"id: {annotation_id} bbox: {yolo_bbox} keypoints: {yolo_keypoints} segmentations: {yolo_segmentation}"
#
#     # Remove file extension from image name
#     file_name_no_ext = os.path.splitext(metadata["file_name"])[0]
#
#     # Define the output file path
#     output_file_path = os.path.join(output_dir, file_name_no_ext + ".txt")
#
#     # Save the annotation in YOLO format to the text file
#     with open(output_file_path, "a") as f:  # Use 'a' to append annotations for multiple objects in the same image
#         f.write(yolo_annotation + "\n")
#
#     print(f"Saved annotation for image ID {image_id} to {output_file_path}")



import json
import os

# Paths
input_json_path = "coco/annotations/person_keypoints_val2017.json"  # Input COCO JSON file
output_dir = "val_annotations_yolo_format"  # Directory to save YOLO format text files

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load the COCO annotation file
with open(input_json_path, "r") as f:
    coco_data = json.load(f)

# Create a mapping from image ID to image metadata
image_metadata = {img["id"]: {"file_name": img["file_name"], "height": img["height"], "width": img["width"]}
                  for img in coco_data["images"]}

# Helper function to normalize a value
def normalize(value, scale):
    return value / scale

# Convert and save annotations for each image
for annotation in coco_data["annotations"]:
    try:
        image_id = annotation["image_id"]
        metadata = image_metadata[image_id]
        img_width, img_height = metadata["width"], metadata["height"]

        # Convert bounding box from [x, y, w, h] to [center_x, center_y, normalized_width, normalized_height]
        bbox = annotation["bbox"]
        x, y, w, h = bbox
        center_x = normalize(x + w / 2, img_width)
        center_y = normalize(y + h / 2, img_height)
        norm_w = normalize(w, img_width)
        norm_h = normalize(h, img_height)
        yolo_bbox = f"{center_x} {center_y} {norm_w} {norm_h}"

        # Convert keypoints (normalize and retain visibility)
        keypoints = annotation.get("keypoints", [])
        yolo_keypoints = []
        for i in range(0, len(keypoints), 3):
            kp_x = normalize(float(keypoints[i]), img_width)
            kp_y = normalize(float(keypoints[i + 1]), img_height)
            visibility = keypoints[i + 2]
            yolo_keypoints.append(f"{kp_x} {kp_y} {visibility}")
        yolo_keypoints = " ".join(yolo_keypoints)

        # Convert segmentation points (normalize coordinates)
        segmentation = annotation.get("segmentation", [])
        yolo_segmentation = []
        for seg in segmentation:
            for i in range(0, len(seg), 2):
                seg_x = normalize(float(seg[i]), img_width)
                seg_y = normalize(float(seg[i + 1]), img_height)
                yolo_segmentation.append(f"{seg_x} {seg_y}")
        yolo_segmentation = " ".join(yolo_segmentation)

        # Create the YOLO format string
        annotation_id = annotation["id"]
        yolo_annotation = f"id: {annotation_id} bbox: {yolo_bbox} keypoints: {yolo_keypoints} segmentations: {yolo_segmentation}"

        # Remove file extension from image name
        file_name_no_ext = os.path.splitext(metadata["file_name"])[0]

        # Define the output file path
        output_file_path = os.path.join(output_dir, file_name_no_ext + ".txt")

        # Save the annotation in YOLO format to the text file
        with open(output_file_path, "a") as f:  # Use 'a' to append annotations for multiple objects in the same image
            f.write(yolo_annotation + "\n")

        print(f"Saved annotation for image ID {image_id} to {output_file_path}")

    except (ValueError, TypeError, KeyError) as e:
        print(f"Skipping annotation for image ID {annotation.get('image_id')} due to error: {e}")
