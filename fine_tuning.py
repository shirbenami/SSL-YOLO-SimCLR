from ultralytics import YOLO
import torch

from check import missing, unexpected

# Step 1: Load the model and pretrained SSL weights

combined_weights = torch.load('./output/ssl_train/combined_model.pth')  # Path to SSL-trained backbone weights

#Load the model
model = YOLO("yolov8n.yaml") # Create a YOLOv8 model instance

print("Before loading weights:", list(model.parameters())[0].mean().item())

# Load weights into the model
model.model.load_state_dict(combined_weights, strict=False)

#check if the keys are suit:
missing,unexpected = model.model.load_state_dict(combined_weights, strict=False)

print("After loading weights:", list(model.parameters())[0].mean().item())
print("Missing keys:", missing)
print("Unexpected keys:", unexpected)
print("Combined model weights loaded successfully!")


# Freeze the backbone layers to retain SSL-learned features
trained_layers = 10
for idx, layer in enumerate(model.model.model[:trained_layers]):  # Iterate through the first trained_layers
    for param in layer.parameters():
        param.requires_grad = False

print("Backbone layers have been frozen.")

#print a check - what are the layers in the train
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)


print("Before fine-tuning:", list(model.model.model[0].parameters())[0].mean().item())


# Step 2: Fine-tune the model
model.train(
    data='datasets/check/data.yaml',  # Path to the YAML file
    epochs=10,  # Number of epochs for fine-tuning
    imgsz=640,  # Image size for training
    batch=4,  # Batch size, adjust based on your hardware
    lr0=1e-5,  # Learning Rate
    pretrained=False,
    project='output/check/fine_tuning',  # Save results in /output/fine_tuning
    name='train_results',  # Sub-folder for this specific run
)

print("After fine-tuning:", list(model.model.model[0].parameters())[0].mean().item())


# Step 3: Evaluate the model on the validation dataset
metrics = model.val(
    project='output/check/fine_tuning',  # Save validation results here
    name='validation_results'  # Sub-folder for validation results
)
print(metrics)  # Prints evaluation metrics such as mAP (Mean Average Precision)

# Step 4: Make predictions on test images
results = model.predict(
    source='datasets/check/test/images',  # Path to test images
    save=True,  # Save predictions with bounding boxes
    project='output/check/fine_tuning',  # Save predictions in this directory
    name='test_predictions',  # Sub-folder for test predictions
    imgsz=640  # Image size for predictions
)

print("Fine-tuning, evaluation, and predictions completed.")

