from ultralytics import YOLO
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn.init as init
from loss_functions.info_nce import InfoNCE
import os
from torchvision.transforms import functional as F
from PIL import Image
from torch.utils.data import random_split


# SimCLRDataset class - creates pairs of images (Anchor and Positive) with transformations
class SimCLRDataset(Dataset):
    def __init__(self, root_dir, transform):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, img) for img in os.listdir(root_dir)]

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        img = Image.open(img_path).convert('RGB')
        anchor = self.transform(img)
        positive = self.transform(img)
        return anchor, positive

    def __len__(self):
        return len(self.image_paths)


def build_classifier(lr):
    """
        Builds the SimCLR model using YOLO's backbone and a projection head.
        Prepares the DataLoaders for the dataset.

        :param lr: Learning rate for the optimizer.
        :return: Train loader, validation loader, model, loss function, optimizer.
    """

    # Load YOLO model and extract the backbone
    model = YOLO("yolov8n.yaml")  # Use YOLOv8 nano model

    # Print the structure of the model
    print(model.model)

    trained_layers = 10  # Number of layers to extract from YOLO backbone
    model_children_list = list(model.model.children())  # Extract all children layers
    for i, child in enumerate(model_children_list):
        print(f"Layer {i}: {child}")
    backbone = model_children_list[0][:trained_layers]  # Get the first n layers
    print("Backbone keys:", backbone.state_dict().keys())

    # Define the SimYOLOv8 class
    class SimYOLOv8(nn.Module):
        def __init__(self):
            super(SimYOLOv8, self).__init__()
            # Feature extraction
            self.backbone = backbone
            # Projection head
            self.mlp = nn.Sequential(
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),  # Output shape: (batch_size, 256, 1, 1)
                nn.Flatten(),  # Output shape: (batch_size, 256)
                nn.Linear(256, 128),  # Map features to a 128-dimensional space
            )

        def forward(self, x):
            # Pass the input through the backbone
            features = self.backbone(x)
            # Pass the features through the projection head
            compact_features = self.mlp(features)
            return compact_features

    # Instantiate the SimYOLOv8 model
    model = SimYOLOv8()

    # Define the loss function and optimizer
    criterion = InfoNCE(temperature=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Define the data transforms
    data_transforms = transforms.Compose([
    transforms.RandomResizedCrop(640),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Replace train_dataset with SimCLRDataset

    full_dataset = SimCLRDataset('datasets/cropped/mix_crops', data_transforms)
    dataset_size = len(full_dataset)
    train_size = int(0.7 * dataset_size)
    val_size = int(0.15 * dataset_size)
    test_size = dataset_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size]
    )

    print(f"Dataset split: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test")

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)
    return train_loader, val_loader,test_loader, model, criterion, optimizer









"""
def visualize_augmentations(dataloader, save_dir, num_images=5):
    
    os.makedirs(save_dir, exist_ok=True)

    # Get a batch from the DataLoader
    anchor_batch, positive_batch = next(iter(dataloader))
    print("Anchor shape:", anchor_batch.shape)
    print("Positive shape:", positive_batch.shape)

    for i in range(min(num_images, len(anchor_batch))):
        anchor = anchor_batch[i]
        positive = positive_batch[i]

        # Convert tensors to images
        anchor_img = F.to_pil_image(anchor)
        positive_img = F.to_pil_image(positive)

        # Save images to the specified directory
        anchor_path = os.path.join(save_dir, f"anchor_{i}.png")
        positive_path = os.path.join(save_dir, f"positive_{i}.png")
        anchor_img.save(anchor_path)
        positive_img.save(positive_path)

        print(f"Saved anchor image to: {anchor_path}")
        print(f"Saved positive image to: {positive_path}")

# Example usage
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(96),  # Resize to 96x96
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# Assuming your dataset is in 'datasets/cropped/mix_crops'
train_dataset = SimCLRDataset(root_dir='datasets/cropped/mix_crops', transform=data_transforms['train'])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

# Directory to save images
save_directory = "./output/ssl_train"

# Visualize and save augmentations
visualize_augmentations(train_loader, save_directory, num_images=5)
"""