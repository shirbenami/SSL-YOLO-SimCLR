import os
from PIL import Image
import numpy as np
import torch
from torchvision.transforms import transforms
from ultralytics import YOLO
from torch.nn.functional import cosine_similarity


# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((640, 640)),  # Resize to YOLO input size
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize
])


dataset_path = './datasets/check/images'

# Supervised model
supervised_model = YOLO("output/supervised_train/train_results3/weights/best.pt")  # Path to trained weights

#fine_tuning_model = YOLO("yolov8n.pt")  # Load supervised pretrained model
fine_tuning_model = YOLO("output/fine_tuning/train_results15/weights/best.pt")  # Path to trained weights

# SSL model
ssl_model = YOLO("yolov8n.yaml")
ssl_weights = torch.load('./output/ssl_train/simclr_model.pth')
ssl_model.model.load_state_dict(ssl_weights, strict=False)

""""
# 1. check the weights:"
print("supervised")
print(supervised_model.model)
print("fine_tuning")
print(fine_tuning_model.model)

print("ssl")
print(ssl_model.model.backbone)


supervised_weights = supervised_model.model.state_dict()
fine_tuned_weights = fine_tuning_model.model.state_dict()

weights_identical = True
for key in supervised_weights.keys():
    if key in fine_tuned_weights:
        if not torch.equal(supervised_weights[key], fine_tuned_weights[key]):
            print(f"Difference found in layer: {key}")
            weights_identical = False
    else:
        print(f"Layer {key} not found in fine-tuned model.")
        weights_identical = False

if weights_identical:
    print("The weights of both models are identical.")
else:
    print("The weights of the models are different.")

"""
#2. check the backbone and comparing to the ssl model:
model = YOLO("yolov8n.yaml") # Create a YOLOv8 model instance

ssl_weights = torch.load('./output/ssl_train/simclr_model.pth')

backbone = supervised_model.model.model[:10]

print("Before loading weights:", list(backbone.parameters())[0].mean().item())

# בצע מיפוי להסרת ה-prefix 'backbone.'
ssl_weights_mapped = {}
for k, v in ssl_weights.items():
    if k.startswith('backbone.'):
        new_key = k.replace('backbone.', '')  # הסרת ה-prefix
        ssl_weights_mapped[new_key] = v

# טען את המשקולות הממופות ל-backbone
missing, unexpected = backbone.load_state_dict(ssl_weights_mapped, strict=False)
print("After loading weights:", list(backbone.parameters())[0].mean().item())
print("Missing keys:", missing)
print("Unexpected keys:", unexpected)


print("Keys in SSL weights:", ssl_weights_mapped.keys())

backbone_keys = backbone.state_dict().keys()
print("Keys in Backbone:", backbone_keys)

matching_keys = set(ssl_weights_mapped.keys()).intersection(backbone_keys)
print(f"Matching keys between SSL and Backbone: {len(matching_keys)} / {len(backbone_keys)}")
