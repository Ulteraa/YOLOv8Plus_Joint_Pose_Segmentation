import os
import shutil

# Define folder paths
text_folder = "Pose_Segmetn_YOLO_Dasets/train/labels"  # Replace with your text folder path
image_folder = "Pose_Segmetn_YOLO_Dasets/train/images"  # Replace with your images folder path
output_folder = "path_to_output_images_folder"  # Replace with your destination folder path

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Process each text file
for text_file in os.listdir(text_folder):
    if text_file.endswith(".txt"):
        # Extract the file name without the extension
        file_name = os.path.splitext(text_file)[0]

        # Check for the corresponding image
        image_path = os.path.join(image_folder, f"{file_name}.jpg")
        if os.path.exists(image_path):
            # Copy the image to the output folder
            shutil.copy(image_path, output_folder)
            print(f"Copied: {image_path}")
        else:
            print(f"Image not found for: {text_file}")

print("Process completed!")
