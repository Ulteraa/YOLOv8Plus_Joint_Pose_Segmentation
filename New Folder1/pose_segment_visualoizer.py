import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from matplotlib import pyplot as plt

# Define paths to the image and annotation folders
image_folder = "Pose_Segmetn_YOLO_Dasets/images"  # path of the folder containing images
annotation_folder = "Pose_Segmetn_YOLO_Dasets/labels"  # path of the folder containing annotation text files

# Constants for the image size

# Helper functions

def mask_visualize(image, mask, boxes, labels, keypoints):
    fontsize = 18
    f, ax = plt.subplots(2, 1, figsize=(8, 8))
    ax[0].imshow(image)

    # Draw bounding boxes on the image
    for box, label, keypoint in zip(boxes, labels, keypoints):
        x_min, y_min, x_max, y_max = box
        rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, fill=False, edgecolor='red', linewidth=2)
        ax[0].add_patch(rect)
        ax[0].text(x_min, y_min, label, fontsize=fontsize, color='white', bbox=dict(facecolor='red', alpha=0.5))
        for kp in keypoint:
            kp_x, kp_y, visibility = kp  # Unpack keypoint values
            if visibility > 0:  # Only draw visible keypoints (you can customize this check)
                circle = plt.Circle((kp_x, kp_y), radius=5, color='blue',
                                    fill=True)  # Adjust radius and color as needed
                ax[0].add_patch(circle)

    ax[1].imshow(mask)
    plt.show()


def parse_annotation_line(line):
    try:
        # Split the line based on spaces
        components = line.split()

        # Ensure the line contains the expected components
        if 'id:' not in components or 'bbox:' not in components:
            raise ValueError(f"Invalid format in line: {line}")

        # Extract class ID (assuming it's after 'id:')
        class_id_idx = components.index('id:') + 1
        class_id = components[class_id_idx]

        # Extract bbox (assuming it's after 'bbox:')
        bbox_idx = components.index('bbox:') + 1
        bbox = list(map(float, components[bbox_idx:bbox_idx + 4]))  # Get bbox in (x_center, y_center, width, height)

        # Extract keypoints (if present)
        keypoints = []
        if 'keypoints:' in components:
            kp_idx = components.index('keypoints:') + 1
            kp_end = components.index('segmentations:') if 'segmentations:' in components else len(components)
            keypoints = list(map(float, components[kp_idx:kp_end]))

        # Extract segmentation (if present)
        segmentations = []
        if 'segmentations:' in components:
            seg_idx = components.index('segmentations:') + 1
            segmentations = list(map(float, components[seg_idx:]))

        return class_id, bbox, keypoints, segmentations

    except (IndexError, ValueError) as e:
        print(f"Error parsing line: {line}. Error: {e}")
        return None, None, None, None  # Returning None to prevent crashes



def generate_points(annotation_path=''):
    labels = []
    bounding_boxes = []
    segmentations = []
    keypoints =[]

    with open(annotation_path, "r") as file:
        for line in file:
            class_id, bbox, kypnts, segmentation = parse_annotation_line(line)
            if class_id is None:  # Skip lines that couldn't be parsed
                continue

            labels.append(class_id)

            # Convert bbox from normalized YOLO format to pixel values
            x_center, y_center, width, height = bbox
            x_min = int((x_center - width / 2) * IMAGE_WIDTH)
            y_min = int((y_center - height / 2) * IMAGE_HEIGHT)
            x_max = int((x_center + width / 2) * IMAGE_WIDTH)
            y_max = int((y_center + height / 2) * IMAGE_HEIGHT)
            bounding_boxes.append((x_min, y_min, x_max, y_max))
            keypoints_list = [[kypnts[i]*IMAGE_WIDTH, kypnts[i + 1]*IMAGE_HEIGHT, kypnts[i + 2]] for i in
                              range(0, len(kypnts), 3)]
            keypoints.append(keypoints_list)

            # Convert segmentation points to pixel values
            scaled_seg = [(segmentation[i] * IMAGE_WIDTH if i % 2 == 0 else segmentation[i] * IMAGE_HEIGHT) for i in range(len(segmentation))]
            segmentations.append(scaled_seg)

    return labels, segmentations, bounding_boxes,  keypoints


def convert_boundary_to_mask_array(labels, points, show=0):
    mask = Image.new("L", (IMAGE_WIDTH, IMAGE_HEIGHT), 0)
    draw = ImageDraw.Draw(mask)
    for i, boundary_coords in enumerate(points):
        draw.polygon(boundary_coords, fill=1)
        centroid_x = sum(x for x, _ in zip(boundary_coords[::2], boundary_coords[1::2])) / len(boundary_coords[::2])
        centroid_y = sum(y for _, y in zip(boundary_coords[::2], boundary_coords[1::2])) / len(boundary_coords[::2])
        centroid = (int(centroid_x), int(centroid_y))
        text = str(labels[i])
        font = ImageFont.load_default()
        text_w, text_h = draw.textsize(text, font=font)
        text_pos = (centroid[0] - text_w / 2, centroid[1] - text_h / 2)
        draw.text(text_pos, text, font=font, fill='black')
    mask_array = np.array(mask) * 255
    return mask_array

def generate_mask(annotation_path='', show=0):
    labels, segmentations, bounding_boxes, keypoints = generate_points(annotation_path)
    mask_array = convert_boundary_to_mask_array(labels, segmentations, show)
    return mask_array, bounding_boxes, labels, keypoints

# Process all images and corresponding annotations
image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.jpg')])
annotation_files = sorted([f for f in os.listdir(annotation_folder) if f.endswith('.txt')])

for image_file, annotation_file in zip(image_files, annotation_files):
    image_path = os.path.join(image_folder, image_file)
    annotation_path = os.path.join(annotation_folder, annotation_file)

    # Open and resize the image
    img = Image.open(image_path)
    IMAGE_WIDTH, IMAGE_HEIGHT = img.size
    # img = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT))

    # Generate the mask and bounding boxes
    mask_array, bounding_boxes, labels, keypoints = generate_mask(annotation_path=annotation_path, show=0)

    # Visualize the image with bounding boxes and mask
    mask_visualize(np.array(img), mask_array, bounding_boxes, labels, keypoints)
