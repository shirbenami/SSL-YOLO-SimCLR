import os
import random
import shutil

def split_ssl_dataset(mix_crops_folder, split_ssl_path):
    """
    Split a folder of mixed images into train, val, and test sets.

    Args:
        mix_crops_folder (str): Path to the folder containing mixed crops.
        split_ssl_path (str): Path to the output folder for the split datasets.

    Returns:
        tuple: Paths to train, val, and test folders.
    """
    # Define the folder paths
    train_folder = os.path.join(split_ssl_path, 'train')
    val_folder = os.path.join(split_ssl_path, 'val')
    test_folder = os.path.join(split_ssl_path, 'test')

    # Create the train, val, and test folders if they don't exist
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    # Get the list of all images in the mix_crops folder
    cropped_images = [image for image in os.listdir(mix_crops_folder) if image.endswith(('.png', '.jpg', '.jpeg'))]

    # Shuffle the images randomly
    random.shuffle(cropped_images)

    # Calculate the split sizes
    num_images = len(cropped_images)
    num_train_images = int(0.75 * num_images)
    num_val_images = int(0.15 * num_images)
    num_test_images = num_images - num_train_images - num_val_images

    # Copy the images to the respective folders
    for i, image in enumerate(cropped_images):
        src_path = os.path.join(mix_crops_folder, image)
        if i < num_train_images:
            dst_path = os.path.join(train_folder, image)
        elif i < num_train_images + num_val_images:
            dst_path = os.path.join(val_folder, image)
        else:
            dst_path = os.path.join(test_folder, image)
        
        shutil.copy(src_path, dst_path)

    print(f"Dataset split completed: {num_train_images} train, {num_val_images} val, {num_test_images} test")
    return train_folder, val_folder, test_folder
