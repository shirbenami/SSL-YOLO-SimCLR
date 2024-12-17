import torch

def validate_model(model, val_loader, criterion, device):
    """
    Performs validation and calculates average loss and accuracy.
    """
    model.eval()  # Switch to evaluation mode
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient computation
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            val_loss += loss.item()

    accuracy = 100 * correct / total
    return val_loss / len(val_loader), accuracy
