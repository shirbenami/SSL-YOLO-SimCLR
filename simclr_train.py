import torch
from model.ssl_model import build_classifier
from trainers_ssl.train import train_model
from trainers_ssl.validate import validate_model
from trainers_ssl.test import test_model
import os
from ultralytics import YOLO

if __name__ == "__main__":


    # Define number of epochs
    num_epochs = 30

    # Define the output directory
    output_dir = './output/ssl_train'
    os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist

    # Load the model, dataloaders, loss function, and optimizer
    train_loader, val_loader,test_loader, model, criterion, optimizer = build_classifier(lr=0.001)

    # Check if CUDA (GPU) is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)


    train_losses = []
    val_losses = []


    # Training loop
    for epoch in range(num_epochs):
        # Training step
        train_loss = train_model(model, train_loader, criterion, optimizer, device)

        # Validation step
        val_loss = validate_model(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Print progress
        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        model.train()
        total_loss = 0.0

    #print the model names - only backbone layers 0-9
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)


    # Save the trained model
    model_path = os.path.join(output_dir, "simclr_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved successfully at: {model_path}")


    #save the combined model

    # Extracting Backbone
    backbone = model.backbone

    yolo = YOLO("yolov8n.yaml")  # Use YOLOv8 nano model
    trained_layers = 10  # Number of layers to extract from YOLO backbone
    model_children_list = list(yolo.model.children())  # Extract all children layers
    head_layers = model_children_list[0][trained_layers:]

    print("Backbone keys:", backbone.state_dict().keys())
    print("Head layers keys:", head_layers.state_dict().keys())

    full_state_dict = {**backbone.state_dict(), **head_layers.state_dict()}
    full_state_dict = {f'model.{k}': v for k, v in full_state_dict.items()}

    # Print full state_dict keys - all the layers - from 0 to 22
    print("Full model state_dict keys:")
    for key in full_state_dict.keys():
        print(key)




    # Save results as graphs
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 5))

    # Loss graph
    plt.plot(range(1, num_epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, num_epochs + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss over Epochs")
    plt.legend()

    # Save the graph to the output directory
    graph_path = os.path.join(output_dir, "ssl_losses_graphs.png")
    plt.tight_layout()
    plt.savefig(graph_path)
    print(f"Loss graph saved at: {graph_path}")


    # Save the combined train model
    combined_model_path = os.path.join(output_dir, "combined_model.pth")
    torch.save(full_state_dict, combined_model_path)
    print(f"Model saved successfully at: {combined_model_path}")

    log_path = os.path.join(output_dir, "training_log.txt")
    with open(log_path, "w") as f:
        for epoch, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses), 1):
            f.write(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Validation Loss={val_loss:.4f}\n")
    print(f"Training log saved at: {log_path}")

    # Test step
    test_loss = test_model(model, test_loader, criterion, device)
    test_log_path = os.path.join(output_dir, "test_log.txt")
    with open(test_log_path, "w") as f:
        f.write(f"Test Loss: {test_loss:.4f}\n")
    print(f"Test log saved at: {test_log_path}")

