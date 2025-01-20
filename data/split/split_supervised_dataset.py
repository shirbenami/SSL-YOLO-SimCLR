import os
import random
import shutil

def split_supervised_dataset(supervised_path, split_supervised_path):
    """
    Splits the SUPERVISED dataset into three subsets: train (75%), val (15%), and test (10%),
    and saves them into the provided split_supervised_path.

    Args:
        supervised_path (str): Path to the SUPERVISED dataset containing 'images' and 'labels'.
        split_supervised_path (str): Path to save the split dataset with 'train', 'val', and 'test' subfolders.
    """
    # Define paths for images and labels within the SUPERVISED dataset
    images_path = os.path.join(supervised_path, "images")
    labels_path = os.path.join(supervised_path, "labels")

    # Ensure source directories exist
    if not os.path.exists(images_path) or not os.path.exists(labels_path):
        raise FileNotFoundError("Ensure 'images' and 'labels' folders exist in the SUPERVISED dataset.")

    # Create destination directories
    train_path = os.path.join(split_supervised_path, "train")
    val_path = os.path.join(split_supervised_path, "val")
    test_path = os.path.join(split_supervised_path, "test")

    os.makedirs(os.path.join(train_path, "images"), exist_ok=True)
    os.makedirs(os.path.join(train_path, "labels"), exist_ok=True)
    os.makedirs(os.path.join(val_path, "images"), exist_ok=True)
    os.makedirs(os.path.join(val_path, "labels"), exist_ok=True)
    os.makedirs(os.path.join(test_path, "images"), exist_ok=True)
    os.makedirs(os.path.join(test_path, "labels"), exist_ok=True)

    # List all image files and their corresponding labels
    all_images = os.listdir(images_path)
    all_labels = os.listdir(labels_path)

    # Ensure that each image has a corresponding label
    image_label_pairs = [(img, img.replace('.jpg', '.txt')) for img in all_images if img.replace('.jpg', '.txt') in all_labels]

    if not image_label_pairs:
        raise ValueError("No matching image-label pairs found. Check the dataset structure.")

    # Shuffle the list of image-label pairs
    random.shuffle(image_label_pairs)

    # Calculate split indices
    total = len(image_label_pairs)
    train_split = int(total * 0.75)
    val_split = int(total * 0.15)

    train_pairs = image_label_pairs[:train_split]
    val_pairs = image_label_pairs[train_split:train_split + val_split]
    test_pairs = image_label_pairs[train_split + val_split:]

    # Copy image-label pairs to the corresponding folders
    for img, lbl in train_pairs:
        shutil.copy(os.path.join(images_path, img), os.path.join(train_path, "images", img))
        shutil.copy(os.path.join(labels_path, lbl), os.path.join(train_path, "labels", lbl))

    for img, lbl in val_pairs:
        shutil.copy(os.path.join(images_path, img), os.path.join(val_path, "images", img))
        shutil.copy(os.path.join(labels_path, lbl), os.path.join(val_path, "labels", lbl))

    for img, lbl in test_pairs:
        shutil.copy(os.path.join(images_path, img), os.path.join(test_path, "images", img))
        shutil.copy(os.path.join(labels_path, lbl), os.path.join(test_path, "labels", lbl))

    print(f"Dataset split complete. Train: {len(train_pairs)}, Val: {len(val_pairs)}, Test: {len(test_pairs)}")

# Example usage
supervised_path = '/datasets/final_dataset/supervised'
split_supervised_path = '/datasets/final_dataset/split_supervised'

split_supervised_dataset(supervised_path, split_supervised_path)

    