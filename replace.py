import os

# Define the folder containing the text files
folder_path = "/home/fariborz_taherkhani/Joint_Pose_Segment_Yolo-master/DataSet/test/labels"

# Iterate through each file in the folder
for file_name in os.listdir(folder_path):
    # Construct the full file path
    file_path = os.path.join(folder_path, file_name)

    # Check if the file is a text file
    if file_name.endswith(".txt") and os.path.isfile(file_path):
        # Read the file contents
        with open(file_path, "r") as file:
            lines = file.readlines()

        # Replace "id: 1" with "id: 0" in each line
        updated_lines = [line.replace("id: 1", "id: 0") for line in lines]

        # Write the updated lines back to the file
        with open(file_path, "w") as file:
            file.writelines(updated_lines)

print("All occurrences of 'id: 1' have been replaced with 'id: 0'.")
