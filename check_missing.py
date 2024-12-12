import os

# Define the label folder path
label_folder = "/home/fariborz_taherkhani/Joint_Pose_Segment_Yolo-master/Pose_Segmetn_YOLO_Dasets/test/labels"  # Replace with your folder path


# Validation function
def validate_annotation(file_path):
    with open(file_path, "r") as file:
        content = file.read()

    errors = []

    # Validate "id:"
    if "id:" not in content or content.split("id:")[1].split()[0].strip() != "1":
        errors.append("Invalid or missing 'id' field")

    # Validate "bbox:"
    if "bbox:" in content:
        bbox_values = content.split("bbox:")[1].split("keypoints:")[0].strip().split()
        if len(bbox_values) != 4 or not all(is_float(v) for v in bbox_values):
            errors.append("Invalid 'bbox' field")
    else:
        errors.append("Missing 'bbox' field")

    # Validate "keypoints:"
    if "keypoints:" in content:
        keypoints_values = content.split("keypoints:")[1].split("segmentations:")[0].strip().split()
        if len(keypoints_values) != 51 or not all(is_float(v) for v in keypoints_values):
            errors.append("Invalid 'keypoints' field")
    else:
        errors.append("Missing 'keypoints' field")

    # Validate "segmentation:"
    if "segmentations:" in content:
        segmentation_values = content.split("segmentations:")[1].strip().split()
        if len(segmentation_values) < 3:
            errors.append("Invalid 'segmentations' field")
    else:
        errors.append("Missing 'segmentations' field")

    return errors


# Helper function to check if a value is a valid float
def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


# Process all text files in the folder
for label_file in os.listdir(label_folder):
    if label_file.endswith(".txt"):
        file_path = os.path.join(label_folder, label_file)
        errors = validate_annotation(file_path)

        # If there are errors, print them
        if errors:
            print(f"Errors in file: {label_file}")
            for error in errors:
                print(f"  - {error}")

print("Validation completed!")
