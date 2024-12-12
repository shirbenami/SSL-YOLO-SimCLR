# Self-Supervised Learning and Fine-Tuning with YOLOv8
This repository contains a project focused on leveraging YOLOv8 for self-supervised learning (SSL) using SimCLR principles and fine-tuning it on a labeled dataset. The project includes SSL training, fine-tuning, supervised training, and detailed evaluation tools to compare the benefits of SSL-pretrained backbones.

## Project Overview

This project explores the integration of YOLOv8 with SimCLR for self-supervised representation learning, followed by fine-tuning on a labeled dataset. The main workflow includes:

#### 1. Self-Supervised Pretraining (SimCLR):
* Trains YOLOv8’s backbone using contrastive learning (SimCLR) on unlabeled image crops.
* Optimizes feature representations using the InfoNCE loss function.

#### 2. Fine-Tuning:
* Fine-tunes the pretrained YOLOv8 backbone on labeled datasets.
* Freezes the backbone and trains only the head for classification.

#### 3. Supervised Training Baseline:
* Trains YOLOv8 from scratch in a fully supervised manner for baseline comparison.

#### 4. Evaluation:
* Compares the SSL-pretrained fine-tuned model with the fully supervised baseline.
* Provides visualizations such as loss graphs, confusion matrices, and accuracy metrics.


## Dataset - STL10

* The dataset includes cropped images (with cars and random backround) to train and evaluate the models:
Unlabeled Dataset: For self-supervised learning.
Labeled Dataset: For supervised training and fine-tuning.
The dataset is divided as follows:
70% for Training, 15% for Validation, and 15% for Testing.
  
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
├── ssl_train.py                # Main script for training YOLOv8 with SSL (SimCLR).
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

Run SSL training on cropped images using SimCLR principles:

```python
python3 ssl_train.py

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
![train_val_graphs_supervised_model_60epochs](https://github.com/user-attachments/assets/23b23c3b-d5ac-4422-93cd-3e1ee5c1dd56)

* Test Accuracy: 73.55%

### Fine-Tuned (SSL):
![fine_tuning_classification_graphs_60epochs](https://github.com/user-attachments/assets/f83fb656-e983-41b0-90e0-803fdc654cb0)

* Test Accuracy: 82.49%

## 2. Confusion Matrix

### Supervised Baseline:

![confusion_matrix_60epochs](https://github.com/user-attachments/assets/3a3dd871-be6d-44af-b031-4164b96c6549)

### Fine-Tuned (SSL):

![confusion_matrix_fine_tuning_60epochs](https://github.com/user-attachments/assets/8c530b73-d825-4af8-9b28-dc97de92fd2e)


## Resources
- [SimCLR Tutorial](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial17/SimCLR.html)
- [STL10 Dataset](https://www.kaggle.com/datasets/jessicali9530/stl10?resource=download)
