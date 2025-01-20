import os
from collections import Counter

def count_objects_in_split(split_supervised_path):
    """
    Counts the number of objects (unique labels) in the train, val, and test directories.

    Args:
        split_supervised_path (str): Path to the split_supervised directory containing train, val, and test.

    Returns:
        dict: A dictionary containing the count of each object type in train, val, and test.
    """
    subsets = ['train', 'val', 'test']
    object_counts = {}

    for subset in subsets:
        labels_path = os.path.join(split_supervised_path, subset, "labels")
        
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"Labels directory not found for {subset}: {labels_path}")

        # Initialize counter for current subset
        subset_counter = Counter()

        # Iterate through label files in the subset
        for label_file in os.listdir(labels_path):
            label_file_path = os.path.join(labels_path, label_file)
            with open(label_file_path, 'r') as f:
                for line in f:
                    # Assuming the label file format is: <class_id> <x_center> <y_center> <width> <height>
                    class_id = line.strip().split()[0]
                    subset_counter[class_id] += 1

        object_counts[subset] = dict(subset_counter)

    return object_counts


