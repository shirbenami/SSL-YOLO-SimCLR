import os
import random
import shutil

def split_to_ssl_and_supervised(dataset_path, ssl_train_path, supervised_path):


    # Create destination directories with subfolders
    os.makedirs(os.path.join(ssl_train_path, "images"), exist_ok=True)
    os.makedirs(os.path.join(ssl_train_path, "labels"), exist_ok=True)
    os.makedirs(os.path.join(supervised_path, "images"), exist_ok=True)
    os.makedirs(os.path.join(supervised_path, "labels"), exist_ok=True)

    # Get a list of all image files and their corresponding labels
    images_path = os.path.join(dataset_path, "images")
    labels_path = os.path.join(dataset_path, "labels")
    
       # Ensure source directories exist
    if not os.path.exists(images_path) or not os.path.exists(labels_path):
        raise FileNotFoundError("Ensure 'images' and 'labels' folders exist in the dataset path.")

    all_images = os.listdir(images_path)
    all_labels = os.listdir(labels_path)

    # Ensure that each image has a corresponding label
    image_label_pairs = [(img, img.replace('.jpg', '.txt')) for img in all_images if img.replace('.jpg', '.txt') in all_labels]

    # Shuffle the list of image-label pairs
    random.shuffle(image_label_pairs)

    # Calculate the split indices
    split_index = int(len(image_label_pairs) * 0.8)

    # Split the data into SSL and SUPERVISED
    ssl_pairs = image_label_pairs[:split_index]
    supervised_pairs = image_label_pairs[split_index:]

    # Copy image-label pairs to the corresponding folders
    for img, lbl in ssl_pairs:
        shutil.copy(os.path.join(images_path, img), os.path.join(ssl_train_path, "images", img))
        shutil.copy(os.path.join(labels_path, lbl), os.path.join(ssl_train_path, "labels", lbl))

    for img, lbl in supervised_pairs:
        shutil.copy(os.path.join(images_path, img), os.path.join(supervised_path, "images", img))
        shutil.copy(os.path.join(labels_path, lbl), os.path.join(supervised_path, "labels", lbl))

    print("Dataset split complete!")