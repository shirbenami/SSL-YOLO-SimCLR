# Self-Supervised Learning and Fine-Tuning with YOLOv8
This repository contains a project focused on leveraging YOLOv8 for self-supervised learning (SSL) using SimCLR principles and fine-tuning it on a labeled dataset. The project includes SSL training, fine-tuning, supervised training, and detailed evaluation tools to compare the benefits of SSL-pretrained backbones.

## Project Overview
This project explores the integration of YOLOv8 with SimCLR for self-supervised representation learning, followed by fine-tuning on a labeled dataset.

![image4](https://github.com/user-attachments/assets/61beb9a4-b33f-4632-9426-68d9629826fc)

* The model structure:
![model structure](https://github.com/user-attachments/assets/1891baea-cb40-4ce9-a280-1951adb63d02)
For YOLOv8, the backbone is the first 10 layers. You can check the [yaml model definition](https://github.com/ultralytics/ultralytics/blob/1d13575ba16623d711c682118ee118615383ba99/ultralytics/cfg/models/v8/yolov8.yaml) to verify that.

. The main workflow includes:

#### 1. Self-Supervised Pretraining (SimCLR):
* Trains YOLOv8’s backbone using contrastive learning (SimCLR) on unlabeled image crops.
* **Data Augmentations:**  
  During SSL training, we apply several augmentations to the input images to generate similar yet distinct views for contrastive learning. These augmentations include:  

  ```python
  # Define the data transforms
  data_transforms = transforms.Compose([
      transforms.RandomResizedCrop(640),               # Crop and resize
      transforms.RandomHorizontalFlip(),              # Random horizontal flip
      transforms.ColorJitter(                         # Random color changes
          brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1
      ),
      transforms.ToTensor(),                          # Convert image to Tensor
      transforms.Normalize([0.485, 0.456, 0.406],     # Normalize image
                           [0.229, 0.224, 0.225])
  ])
  ```
   * Example Augmented Images:
     
    ![image](https://github.com/user-attachments/assets/2b6c677d-1ee1-483e-aca8-4574af3f40e0)

* Optimizes feature representations using the InfoNCE loss function.
  

#### 2. Fine-Tuning:
* Fine-tunes the pretrained YOLOv8 backbone on labeled datasets.
* Freezes the backbone and trains only the head for classification.

#### 3. Supervised Training Baseline:
* Trains YOLOv8 from scratch in a fully supervised manner for baseline comparison.

#### 4. Evaluation:
* Compares the SSL-pretrained fine-tuned model with the fully supervised baseline.
* Provides visualizations such as loss graphs, confusion matrices, and accuracy metrics.


## Dataset Preparation for Project

### Step 1: Selecting the Dataset

The first step in preparing the dataset for the project is to choose a suitable dataset. In this project, the **AITOD** dataset was selected. 
It contain 1000 images.

### Step 2: Dataset Location

The dataset is divided into two main directories:
- **IMAGES**: Contains the images.
- **LABELS**: Contains the labels.

The data is located in the following directory:
```
C:\Users\shir-\PycharmProjects\ssl_yolo\datasets\train
```

### Step 3: Splitting the Dataset

After selecting the dataset, the next step is to split the data into two groups:
- **80%** of the data will be used for SSL TRAIN. 
- **20%** of the data will be used for SUPERVISED.

### Step 4: Splitting the SUPERVISED dataset

- **75%** for TRAIN. 150 images with 2823 objects.
- **15%** for VALIDATION (VAL). 30 images with 923 objects
- **10%** for TEST. 20 images with 528 objects.

### Step 5: Processing the SSL dataset

In this step, the SSL dataset is processed by cropping all objects in the image using the labels. This includes:
- Identifying the objects in each image according to the labels (LABELS).
- Cropping the objects from the images.
- Creating **crops objects** files for all vehicles in the images.
- There are 16,000 crops images.

### Step 6: Creating random crops of background 

In this step, random background crops are created from the images. This includes:
- Selecting random areas in the image that do not include labeled objects.
- Cropping the random areas.
- Saving the **random crops** files for background to balance the data.
- There are 16,000 crops images.


### Step 7: Merge the crops of the vehicles and background to one mix folder:
This step merges vehicle and background crops into a single folder, allowing for random sampling of mixed, unlabeled images during SSL training to ensure diverse and balanced batches.
- There are **32,000** mix crops unlabeled images.


### Step 8: Splitting the SSL dataset:
- **75%** for TRAIN.  24,033 objects.
- **15%** for VALIDATION (VAL). 4806 objects
- **10%** for TEST.  3205 objects.


  
## Project Structure

```python

project_root/
├── data/
│   ├── count_labels/           # Tools for counting objects in datasets.
│   │   ├── __pycache__/        # Compiled Python files for count_labels.
│   │   └── count_objects_in_split.py # Counts objects in dataset splits.
│   ├── crops/                  # Tools for creating and managing cropped datasets.
│   │   ├── __pycache__/        # Compiled Python files for crops.
│   │   ├── __init__.py         # Initialization for the crops module.
│   │   ├── image_crop.py       # Functionality for cropping images.
│   │   └── random_crops.py     # Generates random cropped images.
│   └── split/                  # Tools for dataset splitting.
│       ├── __pycache__/        # Compiled Python files for split.
│       ├── __init__.py         # Initialization for the split module.
│       ├── split_ssl_dataset.py # Splits datasets for SSL training.
│       ├── split_supervised_dataset.py # Splits datasets for supervised training.
│       └── split_to_ssl_and_supervised.py # Splits datasets into SSL and supervised subsets.
│
├── loss_functions/
│   └── info_nce.py             # Implementation of the InfoNCE loss function for SimCLR.
│
├── model/
│   ├── ssl_model.py            # Defines YOLOv8-based SSL model with projection head.
│   ├── yolov8.py               # Core YOLOv8 backbone and head definitions.
│
├── trainers_ssl/
│   ├── train.py                # Training loop for SSL.
│   ├── validate.py             # Validation loop for SSL.
│   └── test.py                 # Testing loop for SSL.
│
├── trainers_supervised/
│   ├── train.py                # Training loop for supervised models.
│   ├── validate.py             # Validation loop for supervised models.
│   └── test.py                 # Testing loop for supervised models.
│
├── output/
│   ├── ssl_train/              # Stores logs, graphs, and models from SSL training.
│   ├── fine_tuning/            # Stores logs, graphs, and models from fine-tuning.
│   └── supervised_train/       # Stores logs, graphs, and models from supervised training.
│
├── simclr_train.py             # Main script for training YOLOv8 with SSL (SimCLR).
├── fine_tune.py                # Main script for fine-tuning using SSL-pretrained weights.
├── supervised_train.py         # Main script for supervised YOLOv8 training.
├── README.md                   # Project description, instructions, and results.
├── main.py                     # prepare all the datasets.
└── .gitignore                  # Specifies files and folders to exclude from version control.



```


## Installation
### Prerequisites

Python 3.8+
PyTorch with CUDA support (if using a GPU)
YOLOv8 installed via pip install ultralytics

## Usage 
### 1. Self-Supervised Pretraining (SimCLR)

Run SSL- SimCLR training on cropped images using SimCLR principles:

```python
python3 simclr_train.py

```

### 2. Fine-Tuning with SSL Weights
Fine-tune the YOLOv8 model using the SSL-pretrained backbone:

```python
python3 fine_tuning.py
```

### 3. Supervised Baseline
Train YOLOv8 from scratch on labeled datasets:

```python
python3 supervised_train.py
 ```

### 4. Visualizations and Evaluation

* Results will be saved in the output folder.
* Check accuracy graphs, loss plots, and confusion matrices.

## Results

## 1. Loss and Accuracy Over Epochs

### Supervised Baseline:

* Test Accuracy: 

### Fine-Tuned (SSL):

* Test Accuracy: 

## 2. Confusion Matrix

### Supervised Baseline:


### Fine-Tuned (SSL):




## Resources
- [SimCLR Tutorial](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial17/SimCLR.html)
- [Yolov8 Architecture](https://github.com/ultralytics/ultralytics/issues/189)
- [Yolov8 Yaml file](https://github.com/ultralytics/ultralytics/blob/1d13575ba16623d711c682118ee118615383ba99/ultralytics/cfg/models/v8/yolov8.yaml)
- [SimCLR information](https://research.google/blog/advancing-self-supervised-and-semi-supervised-learning-with-simclr)
