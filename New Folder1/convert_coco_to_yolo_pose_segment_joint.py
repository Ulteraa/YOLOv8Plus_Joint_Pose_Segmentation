import json
import os

def convert_bbox_to_yolo_format(bbox, img_width, img_height):
    x_min, y_min, width, height = bbox
    center_x = (x_min + width / 2) / img_width
    center_y = (y_min + height / 2) / img_height
    width /= img_width
    height /= img_height
    return center_x, center_y, width, height


def convert_keypoints_to_yolo_format(keypoints, img_width, img_height):
    yolo_keypoints = []
    for i in range(0, len(keypoints), 3):
        x = keypoints[i] / img_width
        y = keypoints[i + 1] / img_height
        visibility = 2  # Assuming visibility is provided
        yolo_keypoints.append(f"{x} {y} {visibility}")
    return yolo_keypoints


def convert_json_to_yolo(json_file_path, output_dir):
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    img_width = data['images'][0]['width']
    img_height = data['images'][0]['height']

    output_lines = []

    for obj in data['annotations']:
        class_id = obj['quality']  # or whatever key contains the class information
        bbox = obj['bbox']  # or however the bbox is stored
        keypoints = obj['keypoints'][0] # or however the keypoints are stored

        center_x, center_y, width, height = convert_bbox_to_yolo_format(bbox, img_width, img_height)
        yolo_keypoints = convert_keypoints_to_yolo_format(keypoints, img_width, img_height)

        yolo_format_line = f"{class_id} {center_x} {center_y} {width} {height} " + " ".join(yolo_keypoints)
        output_lines.append(yolo_format_line)

    output_file_path = os.path.join(output_dir, os.path.splitext(os.path.basename(json_file_path))[0] + '.txt')

    with open(output_file_path, 'w') as out_f:
        out_f.write("\n".join(output_lines))


def process_directory(json_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file_name in os.listdir(json_dir):
        if file_name.endswith('.json'):
            json_file_path = os.path.join(json_dir, file_name)
            convert_json_to_yolo(json_file_path, output_dir)


# Example usage:
json_directory = 'totall_combine_keypoints'
output_directory = 'Yolo_format_Keypoint'
process_directory(json_directory, output_directory)







import os
import json

def convert_to_yolo_format(json_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop over all JSON files in the folder
    for json_file in os.listdir(json_folder):
        if json_file.endswith(".json"):
            with open(os.path.join(json_folder, json_file), 'r') as f:
                data = json.load(f)

            for img_data in data['images']:
                img_id = img_data['id']
                img_width = img_data['width']
                img_height = img_data['height']
                annotations = [ann for ann in data['annotations'] if ann['image_id'] == img_id]

                output_txt_file = os.path.join(output_folder, f"{os.path.splitext(img_data['file_name'])[0]}.txt")

                with open(output_txt_file, 'w') as txt_file:
                    for ann in annotations:
                        class_id = ann['quality']  # or whatever key contains the class information
                        bbox = ann['bbox']  # or however the bbox is stored
                        keypoints = ann['keypoints'][0]  # or however the keypoints are stored

                        center_x, center_y, width, height = convert_bbox_to_yolo_format(bbox, img_width, img_height)
                        yolo_keypoints = convert_keypoints_to_yolo_format(keypoints, img_width, img_height)







                        # Get class ID (note: class IDs are 0-indexed in YOLO, but 1-indexed in COCO)
                        class_id = ann['category_id'] - 1  # convert to 0-indexed

                        # Get bounding box in [x_center, y_center, width, height] and normalize it
                        bbox = ann['bbox']
                        x_center = (bbox[0] + bbox[2] / 2) / img_width
                        y_center = (bbox[1] + bbox[3] / 2) / img_height
                        width = bbox[2] / img_width
                        height = bbox[3] / img_height

                        # Get keypoints if available
                        keypoints = []
                        if 'keypoints' in ann and ann['keypoints']:
                            for kp in ann['keypoints']:
                                kp_x = kp[0] / img_width
                                kp_y = kp[1] / img_height
                                visibility = 1 if kp[2] > 0 else 0  # YOLO style visibility
                                keypoints.extend([kp_x, kp_y, visibility])

                        # Get segmentation if available
                        segmentations = []
                        if 'segmentation' in ann and ann['segmentation']:
                            for seg in ann['segmentation']:
                                seg_normalized = [s / img_width if i % 2 == 0 else s / img_height for i, s in
                                                  enumerate(seg)]
                                segmentations.extend(seg_normalized)

                        # Create the YOLO format string
                        yolo_format = f"id: {class_id} bbox: {x_center} {y_center} {width} {height} keypoints: {' '.join(map(str, keypoints))} segmentations: {' '.join(map(str, segmentations))}\n"

                        # Write to the corresponding text file
                        txt_file.write(yolo_format)


# Usage
json_folder = '/path/to/your/json/folder'
output_folder = '/path/to/your/output/txt/folder'
convert_to_yolo_format(json_folder, output_folder)
