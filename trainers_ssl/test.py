import torch


def test_model(model, test_loader,criterion, device):
    model.eval()
    test_loss = 0

    with torch.no_grad():
        for anchors, positives in test_loader:
            # העברת נתונים למכשיר (CPU/GPU)
            anchors, positives = anchors.to(device), positives.to(device)
            # Combine Anchor and Positive pairs into a single tensor
            inputs = torch.cat([anchors, positives], dim=0)

            # Extract features using the model
            features = model(inputs)

            # Calculate InfoNCE Loss
            loss = criterion(features)

            # Accumulate loss
            test_loss += loss.item()

            # Return average loss
        return test_loss / len(test_loader)




