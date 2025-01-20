import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np


def test_model(model, test_loader, device, class_names, output_path="./output/logs/"):
    """
    Evaluates the model on the test set, calculates accuracy, and generates a Confusion Matrix.

    Args:
        model: Trained PyTorch model.
        test_loader: DataLoader for the test dataset.
        device: Device to run the computations (e.g., 'cuda' or 'cpu').
        class_names: List of class names for the Confusion Matrix.
        output_path: Path to save the confusion matrix plot.

    Returns:
        accuracy: Accuracy of the model on the test set.
    """
    model.eval()  # Set model to evaluation mode
    all_preds = []
    all_labels = []

    # Perform inference
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate accuracy
    accuracy = 100 * np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
    print(f"Test Accuracy: {accuracy:.2f}%")

    # Generate Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig(f"{output_path}confusion_matrix.png")
    plt.show()

    return accuracy
