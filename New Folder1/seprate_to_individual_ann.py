# import json
# import os
#
# # Paths
# input_json_path = "coco/annotations/person_keypoints_val2017.json"  # Path to the input JSON file
# output_dir = "annotations_by_image"  # Directory to save individual text files
#
# # Create the output directory if it doesn't exist
# os.makedirs(output_dir, exist_ok=True)
#
# # Load the COCO annotation file
# with open(input_json_path, "r") as f:
#     coco_data = json.load(f)
#
# # Group annotations by image ID
# annotations_by_image = {}
# for annotation in coco_data["annotations"]:
#     image_id = annotation["image_id"]
#     if image_id not in annotations_by_image:
#         annotations_by_image[image_id] = []
#     annotations_by_image[image_id].append(annotation)
#
# # Map image IDs to file names
# image_id_to_file = {img["id"]: img["file_name"] for img in coco_data["images"]}
#
# # Save each image's annotations to a separate text file
# for image_id, annotations in annotations_by_image.items():
#     file_name = image_id_to_file.get(image_id, f"{image_id}.txt")  # Use file name or image ID as the text file name
#     output_file_path = os.path.join(output_dir, file_name.replace("/", "_") + ".txt")  # Replace slashes for safety
#
#     # Write annotations to the text file
#     with open(output_file_path, "w") as f:
#         json.dump(annotations, f, indent=2)
#
#     print(f"Saved annotations for image ID {image_id} to {output_file_path}")

# import json
# import os
#
# # Paths
# input_json_path = "coco/annotations/person_keypoints_val2017.json"  # Path to the input COCO JSON file
# output_dir = "val_annotations_by_image_json"  # Directory to save individual JSON files
#
# # Create the output directory if it doesn't exist
# os.makedirs(output_dir, exist_ok=True)
#
# # Load the COCO annotation file
# with open(input_json_path, "r") as f:
#     coco_data = json.load(f)
#
# # Group annotations by image ID
# annotations_by_image = {}
# for annotation in coco_data["annotations"]:
#     image_id = annotation["image_id"]
#     if image_id not in annotations_by_image:
#         annotations_by_image[image_id] = []
#     annotations_by_image[image_id].append(annotation)
#
# # Map image IDs to file names
# image_id_to_file = {img["id"]: img["file_name"] for img in coco_data["images"]}
#
# # Save each image's annotations to a separate JSON file
# for image_id, annotations in annotations_by_image.items():
#     # Get the corresponding file name for the image
#     file_name = image_id_to_file.get(image_id, f"{image_id}.json")
#     output_file_path = os.path.join(output_dir, file_name.replace("/", "_") + ".json")  # Replace slashes for safety
#
#     # Create a JSON structure containing the annotations
#     output_data = {
#         "image_id": image_id,
#         "file_name": file_name,
#         "annotations": annotations
#     }
#
#     # Write annotations to the JSON file
#     with open(output_file_path, "w") as f:
#         json.dump(output_data, f, indent=2)
#
#     print(f"Saved annotations for image ID {image_id} to {output_file_path}")


import json
import os

# Paths
input_json_path = "coco/annotations/person_keypoints_train2017.json"  # Path to the input COCO JSON file
output_dir = "train_annotations_by_image_json"  # Directory to save individual JSON files

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load the COCO annotation file
with open(input_json_path, "r") as f:
    coco_data = json.load(f)

# Create a mapping from image ID to image metadata
image_metadata = {img["id"]: {"file_name": img["file_name"], "height": img["height"], "width": img["width"]}
                  for img in coco_data["images"]}

# Group annotations by image ID
annotations_by_image = {}
for annotation in coco_data["annotations"]:
    image_id = annotation["image_id"]
    if image_id not in annotations_by_image:
        annotations_by_image[image_id] = []
    annotations_by_image[image_id].append(annotation)

# Save each image's annotations to a separate JSON file
for image_id, annotations in annotations_by_image.items():
    # Get the image metadata
    metadata = image_metadata.get(image_id, {})

    # Create the output data structure
    output_data = {
        "image_id": image_id,
        "file_name": metadata.get("file_name", f"{image_id}.jpg"),
        "height": metadata.get("height", 0),
        "width": metadata.get("width", 0),
        "annotations": annotations
    }

    # Define the output file path
    output_file_path = os.path.join(output_dir,
                                    output_data["file_name"].replace("/", "_") + ".json")  # Replace slashes for safety

    # Save the JSON file
    with open(output_file_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"Saved annotations for image ID {image_id} to {output_file_path}")
