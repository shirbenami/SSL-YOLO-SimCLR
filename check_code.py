import os
import shutil
import random

# Define paths
dataset_dir = r"C:\Users\shir-\PycharmProjects\ssl_yolo\datasets\new_dataset"
output_dir = r"C:\Users\shir-\PycharmProjects\ssl_yolo\datasets\new_dataset\split"

# Set split ratios
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Create destination folders for train, val, and test sets
for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, split, 'labels'), exist_ok=True)

# Get list of image files
images_dir = os.path.join(dataset_dir, 'images')
labels_dir = os.path.join(dataset_dir, 'labels')

image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

# Shuffle the data to ensure randomness
random.shuffle(image_files)

# Calculate the number of files for each set
num_total = len(image_files)
num_train = int(num_total * train_ratio)
num_val = int(num_total * val_ratio)
num_test = num_total - num_train - num_val  # Ensure total sums up correctly

# Split files into train, val, and test sets
train_files = image_files[:num_train]
val_files = image_files[num_train:num_train + num_val]
test_files = image_files[num_train + num_val:]

# Function to copy files to the appropriate split directory
def copy_files(file_list, split):
    for file in file_list:
        # Copy image
        shutil.copy(os.path.join(images_dir, file), os.path.join(output_dir, split, 'images', file))
        
        # Copy label file (if it exists)
        label_file = file.replace('.jpg', '.txt').replace('.png', '.txt')
        if os.path.exists(os.path.join(labels_dir, label_file)):
            shutil.copy(os.path.join(labels_dir, label_file), os.path.join(output_dir, split, 'labels', label_file))

# Copy files to respective folders
copy_files(train_files, 'train')
copy_files(val_files, 'val')
copy_files(test_files, 'test')

print(f"Splitting complete! {num_train} files for training, {num_val} for validation, and {num_test} for testing.")
