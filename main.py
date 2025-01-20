from data.split.ssl_2_classes_dataset import two_classes_dataset
from data.split.ssl_2_classes_dataset import one_classes_dataset

from data.split.split_ssl_dataset import split_ssl_dataset
from data.split.split_to_ssl_and_supervised import split_to_ssl_and_supervised
from data.split.split_supervised_dataset import split_supervised_dataset
from torchvision.datasets import ImageFolder
from data.crops.image_crop import crop_objects_from_dataset
from data.count_labels.count_objects_in_split import count_objects_in_split
from data.crops.random_crops import crop_random_background
import shutil


if __name__ == '__main__':
    
     # Step 1: Selecting the Dataset. In this project, the AITOD dataset was selected.
     # The data is located in the following directory:
     
    #dataset_folder = r'/datasets/final_dataset'
    dataset_folder = r'C:\Users\shir-\PycharmProjects\ssl_yolo\datasets\final_dataset'
    
    # Step 2: split the data into two groups:
        # 80% of the data will be used for SSL TRAIN.
        # 20% of the data will be used for SUPERVISED
    ssl_train_path = r"C:\Users\shir-\PycharmProjects\ssl_yolo\datasets\final_dataset\ssl_train"
    supervised_path = r"C:\Users\shir-\PycharmProjects\ssl_yolo\datasets\final_dataset\supervised"
    #split_to_ssl_and_supervised(dataset_folder,ssl_train_path,supervised_path)
    
    
   # Step 3: Splitting the supervised set: 70% for TRAIN, 15% for VALIDATION (VAL),15% for TEST:
    split_supervised_path = r"C:\Users\shir-\PycharmProjects\ssl_yolo\datasets\final_dataset\split_supervised"
    #split_supervised_dataset(supervised_path,split_supervised_path)

    #count the number of object in each folder:
    object_counts = count_objects_in_split(split_supervised_path)
   
    # Print results
    #for subset, counts in object_counts.items():
    #    print(f"{subset.upper()} Object Counts:")
     #   for class_id, count in counts.items():
      #      print(f"  Class {class_id}: {count}")
        
    # Step 4: Processing the SSL DATASET:
        # Identifying the objects in each image according to the labels.
        # Cropping the objects from the images.
        # Creating CROPS OBJECT files for all vehicles in the images.
    
    cropped_folder = r"C:\Users\shir-\PycharmProjects\ssl_yolo\datasets\final_dataset\cropped_images"
    #crop_objects_from_dataset(ssl_train_path, cropped_folder)

    #Step 5: Creating random crops of Background for the SSL DATASET:
        # Selecting random areas in the image that do not include labeled objects.
        # Cropping the random areas.
        # Saving the RANDOMCROP files for background to balance the data.
    random_cropped_folder = r"C:\Users\shir-\PycharmProjects\ssl_yolo\datasets\final_dataset\cropped_images\1.0"
    #crop_random_background(ssl_train_path, random_cropped_folder)
    
    #Step 6: merge the cropped images and the random cropped images into one folder.
    
    mix_folder= r"C:\Users\shir-\PycharmProjects\ssl_yolo\datasets\final_dataset\cropped_images\mix_crops"

    # Step 7: Splitting the SSL dataset:
        # - **75%** for TRAIN. 
        # - **15%** for VALIDATION (VAL)
        # - **10%** for TEST.  
    split_ssl_path = r"C:\Users\shir-\PycharmProjects\ssl_yolo\datasets\final_dataset\split_ssl"
    split_ssl_dataset(mix_folder, split_ssl_path)
        
"""
    # define new dataset folder
    #new_dataset_folder = '/MyHomeDir/Dataset2/new_dataset'
    # split cropped images into train, val, and test
    #train_folder, val_folder, test_folder = split_cropped_patches(cropped_folder, new_dataset_folder)


    #dataset_train = ImageFolder(train_folder)
    #dataset_val = ImageFolder(val_folder)
    #dataset_test = ImageFolder(test_folder)
    #print(f'Dataset loaded: {len(dataset_train)} train, {len(dataset_val)} val, {len(dataset_test)} test')

    ssl_dataset = '/MyHomeDir/New_Dataset/ssl_dataset'
    positive_ssl = '/MyHomeDir/New_Dataset/ssl_dataset/positive'
    negative_ssl = '/MyHomeDir/New_Dataset/ssl_dataset/negative'

    # copy 2 classes dataset (5: positive, 2: negative)
    #two_classes_dataset(ssl_dataset, cropped_folder)

    # copy 1 classes dataset (0: positive)
    #one_classes_dataset(ssl_dataset, cropped_folder)

    new_ssl_dataset = '/MyHomeDir/New_Dataset/new_ssl_dataset'
    positive_ssl_new = '/MyHomeDir/New_Dataset/new_ssl_dataset/positive'
    negative_ssl_new = '/MyHomeDir/New_Dataset/new_ssl_dataset/negative'

    # Split the positive and negative folders into three folders: train, val, and test
    #train_folder, val_folder, test_folder = split_ssl_dataset(positive_ssl, positive_ssl_new)
    # Copy the source file to the destination file
    #shutil.copytree(positive_ssl_new, negative_ssl_new)

    #train_folder, val_folder, test_folder = split_ssl_dataset(negative_ssl, negative_ssl_new)

    #train_positive = '/MyHomeDir/Dataset/new_ssl_dataset/positive/train'

    

    
    dataset_folder = '/workspace/Dataset2'
    #crop_objects_from_dataset(dataset_folder)
    cropped_folder = '/workspace/Dataset2/cropped_images2'

    new_dataset_folder = '/workspace/Dataset2/new_dataset'
    train_folder, val_folder, test_folder = split_cropped_patches(cropped_folder, new_dataset_folder)

    dataset_train = ImageFolder(train_folder)
    dataset_val = ImageFolder(val_folder)
    dataset_test = ImageFolder(test_folder)
    print(f'Dataset loaded: {len(dataset_train)} train, {len(dataset_val)} val, {len(dataset_test)} test')


    ssl_dataset = '/workspace/Dataset2/ssl_dataset'
    cropped_folder = '/workspace/Dataset2/cropped_images'
    #two_classes_dataset(ssl_dataset, cropped_folder)

    new_ssl_dataset = '/workspace/Dataset2/new_ssl_dataset'
    #train_folder, val_folder, test_folder = split_cropped_patches2(ssl_dataset, new_ssl_dataset)
    
    """