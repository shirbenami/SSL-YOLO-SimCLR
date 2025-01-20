import torch

def train_model(model, train_loader, criterion, optimizer, device):
    """
    Performs one epoch of supervised training and calculates accuracy.
    """
    model.train()  # Switch to training mode
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        total_loss += loss.item()

    accuracy = 100 * correct / total
    return total_loss / len(train_loader), accuracy
