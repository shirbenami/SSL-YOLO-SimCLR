from ultralytics import YOLO
import torch
import wandb
from wandb.integration.ultralytics import add_wandb_callback
#from check import missing, unexpected

# Step 1: Initialize WANDB
# Initialize W&B run
wandb.init(project="ultralytics", job_type="inference")


model = YOLO("yolov8n.pt") # Create a YOLOv8 model instance

#model_keys = set(model.state_dict().keys())
#print(model.model.state_dict().keys())

# Step 2: Load the model and pretrained SSL weights
ssl_weights = torch.load('./output/ssl_train/simclr_model.pth', map_location=torch.device('cpu'))  # Path to SimCLR-trained backbone
#combined_weights = torch.load('./output/ssl_train/combined_model.pth')  # Path to SSL-trained backbone weights
#for k in ssl_weights.keys():
 #   print(k)
#print(ssl_weights.keys())

trained_layers = 10  # Number of layers to extract from YOLO backbone
model_children_list = list(model.model.children())  # Extract all children layers
head_layers = model_children_list[0][trained_layers:]
backbone = model_children_list[0][:trained_layers]

#print("Backbone keys:", backbone.state_dict().keys())
#print("Head layers keys:", head_layers.state_dict().keys())

#full_state_dict = {**backbone.state_dict(), **head_layers.state_dict()}
#full_state_dict = {f'model.{k}': v for k, v in full_state_dict.items()}

# Print full state_dict keys - all the layers - from 0 to 22
#print("Full model state_dict keys:")
#for key in full_state_dict.keys():
 #   print(key)

print("Before loading the weights:")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)

#******!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# Step 3: Add WANDB callback to automatically log everything
add_wandb_callback(model, enable_model_checkpointing=True)
print("WANDB run URL:", wandb.run.url)

before_weights = list(model.parameters())[0].clone()

adjusted_ssl_weights = {k.replace('backbone.', 'model.model.'): v for k, v in ssl_weights.items() if 'mlp' not in k}
ssl_keys = set(adjusted_ssl_weights.keys())

print(f"adjust ssl:", adjusted_ssl_weights.keys())

adjusted_ssl_weights = {f"model.{k}": v for k, v in adjusted_ssl_weights.items()}
backbone_dict = {k: v for k, v in adjusted_ssl_weights.items() if 'model.0' in k or 'model.1' in k or 'model.2' in k or 'model.3' in k or 'model.4' in k or 'model.5' in k or 'model.6' in k or 'model.7' in k or 'model.8' in k}

#Load SSL weights to the backbone (strict=False to allow partial loading)
backbone.load_state_dict(adjusted_ssl_weights, strict=False)
model.model = torch.nn.Sequential(*backbone, *head_layers)
 
missing, unexpected = model.model.load_state_dict(backbone_dict, strict=False)
print("Missing keys:", missing)
print("Unexpected keys:", unexpected)

model.model.load_state_dict(adjusted_ssl_weights, strict=False)

#for k, v in model.state_dict().items():
 #   if k in adjusted_ssl_weights:
  #      print(f"{k}: {torch.sum(v - adjusted_ssl_weights[k])}")
        
# Freeze the backbone layers to retain SSL-learned features
#trained_layers = 10
#for idx, layer in enumerate(model.model.model[:trained_layers]):  # Iterate through the first trained_layers
 #   for param in layer.parameters():
  #      param.requires_grad = False

#after_weights = list(model.parameters())[0].clone()
#print("Difference in weights:", torch.sum(before_weights - after_weights).item())

#print("after freezing the backbone:")        
#for name, param in model.named_parameters():
 #   if param.requires_grad:
  #      print(name)
        

#print("Backbone layers have been frozen.")
#print("Before fine-tuning:", list(model.model.model[0].parameters())[0].mean().item())


# Step 2: Fine-tune the model
model.train(
    data='datasets/new_dataset/split/data.yaml',  # Path to the YAML file
    epochs=15,  # Number of epochs for fine-tuning
    freeze=0,
    imgsz=640,  # Image size for training
    batch=8  # Batch size, adjust based on your hardware
   # lr0=1e-5,  # Learning Rate
   # project='output/check/fine_tuning',  # Save results in /output/fine_tuning
    #name='train_results',  # Sub-folder for this specific run
)

print("After fine-tuning:", list(model.model.model[0].parameters())[0].mean().item())
print("after fine-tuning LAYERS:")        
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)

# Step 3: Evaluate the model on the validation dataset
#metrics = model.val(
    #project='output/check/fine_tuning',  # Save validation results here
    #name='validation_results'  # Sub-folder for validation results
#)
#print(metrics)  # Prints evaluation metrics such as mAP (Mean Average Precision)

# Step 4: Make predictions on test images
results = model.predict(
    source='datasets/new_dataset/split/test/images'  # Path to test images
   # save=True,  # Save predictions with bounding boxes
   #project='output/check/fine_tuning',  # Save predictions in this directory
    #name='test_predictions',  # Sub-folder for test predictions
    #imgsz=640  # Image size for predictions
)

print("Fine-tuning, evaluation, and predictions completed.")

# Step 6: Finish WANDB run
wandb.finish()
