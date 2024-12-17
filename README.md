# Self-Supervised Learning and Fine-Tuning with YOLOv8
This repository contains a project focused on leveraging YOLOv8 for self-supervised learning (SSL) using SimCLR principles and fine-tuning it on a labeled dataset. The project includes SSL training, fine-tuning, supervised training, and detailed evaluation tools to compare the benefits of SSL-pretrained backbones.

## Project Overview
This project explores the integration of YOLOv8 with SimCLR for self-supervised representation learning, followed by fine-tuning on a labeled dataset.

![image4](https://github.com/user-attachments/assets/61beb9a4-b33f-4632-9426-68d9629826fc)

* The model structure:
![model structure](https://github.com/user-attachments/assets/1891baea-cb40-4ce9-a280-1951adb63d02)

. The main workflow includes:

#### 1. Self-Supervised Pretraining (SimCLR):
* Trains YOLOv8’s backbone using contrastive learning (SimCLR) on unlabeled image crops.
* * **Data Augmentations:**  
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


## Dataset - cars and backrounds

* The supervised dataset consists of labeled images of cars, each containing multiple annotated objects, split into three subsets:
  * **Train set**: 1,216 images with 22,806 labels
  * **Validation set**: 352 images with  6,796 labels
  * **Test set**: 175 images with 3,208 labels 
* The self-supervised learning stage uses a larger dataset containing **47,043** unlabeled images, consisting of a mix of cars and background objects. This dataset enables the model to learn generalizable features without relying on labels:
  * **Dataset split**: 32930 train (70%), 7056 val(15%), 7057 test(15%) 

  
## Project Structure

```python

project_root/
├── datasets/
│   └── cropped/                # Contains labeled and unlabeled datasets.
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
├── simclr_train.py                # Main script for training YOLOv8 with SSL (SimCLR).
├── fine_tune.py                # Main script for fine-tuning using SSL-pretrained weights.
├── supervised_train.py         # Main script for supervised YOLOv8 training.
├── README.md                   # Project description, instructions, and results.
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
- [SimCLR information](https://research.google/blog/advancing-self-supervised-and-semi-supervised-learning-with-simclr)
