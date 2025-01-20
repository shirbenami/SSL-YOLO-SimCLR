from ultralytics import YOLO
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from model.ssl_model import SimCLRDataset
import os
from torchvision.transforms import functional as F
from PIL import Image
from torch.utils.data import random_split


def visualize_augmentations(dataloader, save_dir, num_images):
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
        transforms.RandomResizedCrop(640),  # Resize to 96x96
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# Assuming your dataset is in 'datasets/cropped/mix_crops'
train_dataset = SimCLRDataset(root_dir='datasets/cropped/cropped_images/0.0', transform=data_transforms['train'])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

# Directory to save images
save_directory = "./output/ssl_train"

# Visualize and save augmentations
visualize_augmentations(train_loader, save_directory, num_images=10)