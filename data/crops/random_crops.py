import os
import cv2
from PIL import Image
import random

def crop_random_background(dataset_folder, random_cropped_folder, crop_size=(47, 47), num_crops_per_object=20):
    """
    Creates random background crops of a fixed size from the SSL dataset.

    Args:
        dataset_folder (str): Path to the dataset folder containing 'images' and 'labels'.
        random_cropped_folder (str): Path to save the random background crops.
        crop_size (tuple): Fixed size of the random crops (width, height).
        num_crops_per_object (int): Number of random background crops per labeled object.
    """
    # Create destination directory for random crops
    os.makedirs(random_cropped_folder, exist_ok=True)

    # Iterate over images and labels
    for image_file in os.listdir(os.path.join(dataset_folder, 'images')):
        if image_file == '.ipynb_checkpoints':
            continue  # Skip checkpoint files

        image_path = os.path.join(dataset_folder, 'images', image_file)
        label_file = image_file.replace('jpg', 'txt')
        label_path = os.path.join(dataset_folder, 'labels', label_file)

        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error loading image: {image_file}")
            continue

        height, width, _ = image.shape

        # Parse labels to get bounding boxes
        bounding_boxes = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as file:
                for line in file:
                    if not line.strip():
                        continue
                    try:
                        _, x_center, y_center, box_width, box_height = map(float, line.split())
                        x_min = int((x_center - box_width / 2) * width)
                        y_min = int((y_center - box_height / 2) * height)
                        x_max = int((x_center + box_width / 2) * width)
                        y_max = int((y_center + box_height / 2) * height)
                        bounding_boxes.append((x_min, y_min, x_max, y_max))
                    except ValueError:
                        print(f"Skipping invalid label in {label_file}: {line}")
                        continue

        # Generate random crops
        crop_width, crop_height = crop_size
        for i in range(num_crops_per_object):
            for attempt in range(100):  # Limit to 100 attempts to find a valid crop
                x_min = random.randint(0, width - crop_width)
                y_min = random.randint(0, height - crop_height)
                x_max = x_min + crop_width
                y_max = y_min + crop_height

                # Check if the random crop intersects with any labeled object
                intersects = False
                for box in bounding_boxes:
                    bx_min, by_min, bx_max, by_max = box
                    if x_min < bx_max and x_max > bx_min and y_min < by_max and y_max > by_min:
                        intersects = True
                        break

                # If the crop does not intersect with any labeled object, save it
                if not intersects:
                    cropped_image = image[y_min:y_max, x_min:x_max]
                    cropped_file_name = f"{image_file.split('.')[0]}_random_{i}_{x_min}_{y_min}.png"
                    cropped_path = os.path.join(random_cropped_folder, cropped_file_name)

                    cropped_image = Image.fromarray(cropped_image)
                    cropped_image.save(cropped_path)
                    break  # Exit the attempt loop and move to the next crop

        print(f"Created random crops for image: {image_file}")
