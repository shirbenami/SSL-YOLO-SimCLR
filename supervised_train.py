from ultralytics import YOLO
import wandb
from wandb.integration.ultralytics import add_wandb_callback

# Step 1: Initialize WANDB
# Initialize W&B run
wandb.init(project="ultralytics", job_type="inference")

# Step 2: Create a new YOLO model or load a pre-trained model
model = YOLO("yolov8n.pt")  # Create a YOLOv8 model instance

# Step 3: Add WANDB callback to automatically log everything
add_wandb_callback(model, enable_model_checkpointing=True)
print("WANDB run URL:", wandb.run.url)

# Step 4: Train the model and log results to WANDB
results = model.train(
    data='datasets/new_dataset/split/data.yaml',  # Path to the YAML file
    epochs=5,  # Number of epochs
    imgsz=640,  # Image size
    batch=8  # Batch size
    #project='output/supervised_train',
    #name='train_results'
)

# Step 5: Evaluate the model on the validation dataset
#metrics = model.val(
    #project='output/supervised_train',
    #name='validation_results'
#)


"""
# Step 4: Log the training results to WANDB
wandb.log({
    "train/box_loss": results.results_dict["metrics/box_loss"],
    "train/cls_loss": results.results_dict["metrics/cls_loss"],
    "val/mAP50": results.results_dict.get("metrics/mAP_50(B)", 0),
    "val/mAP50-95": results.results_dict.get("metrics/mAP_50-95(B)", 0),
    "precision": results.results_dict.get("metrics/precision(B)", 0),
    "recall": results.results_dict.get("metrics/recall(B)", 0)
})

conf_matrix = metrics.confusion_matrix  # מטריצת ה-Confusion ישירות מהמודל

wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(
    y_true=np.argmax(conf_matrix, axis=1), 
    probs=conf_matrix,                    
    class_names=['car', 'background']     
   )})
    
"""

# Step 5: Make predictions on test images
predictions = model.predict(
    source='datasets/new_dataset/split/test/images',
    #project='output/supervised_train',
    #name='test_predictions'
)


# Step 6: Finish WANDB run
wandb.finish()
