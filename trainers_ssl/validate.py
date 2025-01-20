import torch

def validate_model(model, val_loader, criterion, device):
    """
    Perform validation and return average loss.
    """
    model.eval()  # Switch to evaluation mode
    val_loss = 0.0

    with torch.no_grad():  # Disable gradient computation
        for anchors, positives in val_loader:
            anchors, positives = anchors.to(device), positives.to(device)

            # Combine Anchor and Positive pairs into a single tensor
            inputs = torch.cat([anchors, positives], dim=0)

            # Extract features using the model
            features = model(inputs)

            # Calculate InfoNCE Loss
            loss = criterion(features)

            # Accumulate loss
            val_loss += loss.item()

    # Return average loss
    return val_loss / len(val_loader)


