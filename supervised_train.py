from ultralytics import YOLO

# Step 1: Create a new model or use a pre-trained model
model = YOLO("yolov8n.yaml") # Create a YOLOv8 model instance

# Step 2: Train the model
model.train(
    data='datasets/check/data.yaml',  # Path to the YAML file
    epochs=10,  # Number of epochs for training
    imgsz=640,  # Image size for training
    batch=4,  # Batch size, adjust based on your hardware
    project='output/check/supervised_train',  # Save results in /data/Projects/fed_learn_fasterRcnn/ssl_project/ssl_yolo/output/supervised_train
    name='train_results'  # Sub-folder for this specific run
)

# Step 3: Evaluate the model on the validation dataset
# Step 3: Evaluate the model on the validation dataset
metrics = model.val(
    project='output/check/supervised_train',  # Save validation results here
    name='validation_results'  # Sub-folder for validation results
)
print(metrics)  # Prints evaluation metrics such as mAP (Mean Average Precision)

# Step 4: Make predictions on test images
results = model.predict(
    source='datasets/check/test/images',  # Path to test images
    save=True,  # Save predictions with bounding boxes
    project='output/check/supervised_train',  # Save predictions in this directory
    name='test_predictions',  # Sub-folder for test predictions
    imgsz=640  # Image size for predictions
)

print("Supervised training, evaluation, and predictions completed.")
