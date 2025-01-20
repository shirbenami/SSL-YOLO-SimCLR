import torch

def train_model(model, train_loader, criterion, optimizer, device):
    """
    Perform one epoch of training.
    """
    model.train()
    total_loss = 0.0

    for anchors, positives in train_loader:
        # Move data to the specified device
        anchors, positives = anchors.to(device), positives.to(device)

        # Combine Anchor and Positive pairs into a single tensor
        inputs = torch.cat([anchors, positives], dim=0)

        # Extract features using the model
        features = model(inputs)

        # Debugging sizes
        #print("Input batch size:", inputs.shape)
        #print("Features batch size:", features.shape)

        # Calculate InfoNCE Loss
        loss = criterion(features)

        # Update the model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate loss
        total_loss += loss.item()

    # Return average loss
    return total_loss / len(train_loader)


