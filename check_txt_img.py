# import os
#
# # Define the paths to the folders
# text_folder = "/home/fariborz_taherkhani/Joint_Pose_Segment_Yolo-master/DataSet/train/labels"  # Replace with the path to your text files
# image_folder = "/home/fariborz_taherkhani/Joint_Pose_Segment_Yolo-master/DataSet/train/images"  # Replace with the path to your image files
#
# # Get lists of all text and image file names
# text_files = [f for f in os.listdir(text_folder) if f.endswith('.txt')]
# image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
#
# # Convert image names to a set for quick lookup
# image_names = {os.path.splitext(f)[0] for f in image_files}
#
# # Check for corresponding images and print missing ones
# for text_file in text_files:
#     text_name = os.path.splitext(text_file)[0]  # Get the base name without extension
#     if text_name not in image_names:
#         print(f"Missing image for: {text_file}")


import os

# Define the paths to the folders
image_folder = "/home/fariborz_taherkhani/Joint_Pose_Segment_Yolo-master/DataSet/test/images"  # Replace with the path to your image files
text_folder = "/home/fariborz_taherkhani/Joint_Pose_Segment_Yolo-master/DataSet/test/labels"   # Replace with the path to your text files

# Get lists of all image and text file names
image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
text_files = [f for f in os.listdir(text_folder) if f.endswith('.txt')]

# Convert text names to a set for quick lookup
text_names = {os.path.splitext(f)[0] for f in text_files}

# Check for corresponding text files and print missing ones
for image_file in image_files:
    image_name = os.path.splitext(image_file)[0]  # Get the base name without extension
    if image_name not in text_names:
        print(f"Missing text file for: {image_file}")
