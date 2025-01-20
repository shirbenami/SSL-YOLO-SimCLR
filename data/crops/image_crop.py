import os
import random
import shutil
import cv2
from PIL import Image
from PIL import ImageOps


def crop_objects_from_dataset(dataset_folder, cropped_folder):
    """
    Step 1: Take a dataset in YOLO format and crop the objects to create a new dataset that contains cropped objects for each class.
    """
    # Iterate over each image and label in the dataset folder
    for image_file in os.listdir(os.path.join(dataset_folder, 'images')):
        if image_file == '.ipynb_checkpoints':
            continue  # Skip the "ipynb_checkpoint" file

        image_path = os.path.join(dataset_folder, 'images', image_file)
        label_file = image_file.replace('jpg', 'txt')
        label_path = os.path.join(dataset_folder, 'labels', label_file)

        # Load the image and read the label
        image = cv2.imread(image_path)
        with open(label_path, 'r') as file:
            label = file.read().strip()  # Remove leading/trailing whitespace and newlines

        # Split the label into separate lines
        label_lines = label.split('\n')

        # Iterate over each line in the label
        for line in label_lines:
            # Skip empty lines
            if not line.strip():
                continue

            # Print the line before unpacking
            print(f"Processing line: {line} in lable: {label_file}")

            # Split the line into the class ID and bounding box coordinates
            class_id, x_center, y_center, width, height = map(float, line.split())

            scale_factor = 1.4
            new_width = width * scale_factor
            new_height = height * scale_factor

            # Convert YOLO format to bounding box coordinates
            x_min = int((x_center - new_width / 2) * image.shape[1])
            y_min = int((y_center - new_height / 2) * image.shape[0])
            x_max = int((x_center + new_width / 2) * image.shape[1])
            y_max = int((y_center + new_height / 2) * image.shape[0])


            # Print out the bounding box coordinates and image dimensions
            print(f"Image: {image_file}")
            print(f"Bounding box coordinates: x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}")
            print(f"Image dimensions: width={image.shape[1]}, height={image.shape[0]}")
            # Crop the image using the bounding box coordinates
            cropped_image = image[y_min:y_max, x_min:x_max]

            # Check if the cropped image is empty
            if cropped_image.size == 0:
                print(f"Error: Cropped image is empty for file {image_file}")
                continue



            # Create a folder for the class ID if it doesn't exist
            class_folder = os.path.join(cropped_folder, str(int(class_id)))
            os.makedirs(class_folder, exist_ok=True)

            # Generate a unique file name for the cropped image
            cropped_file_name = f"{image_file.split('.')[0]}_{x_min}_{y_min}_{x_max}_{y_max}.png"

            # Save the cropped image to the class-specific folder
            cropped_image_path = os.path.join(class_folder, cropped_file_name)
            cropped_image = Image.fromarray(cropped_image)
            cropped_image.save(cropped_image_path)

            # Print a message indicating that the cropped image was saved
            print(f"Saved cropped image: {cropped_image_path}")
