from torch import nn
import torch
import os

class InfoNCE(nn.Module):
    def __init__(self, temperature=0.1):
        super(InfoNCE, self).__init__()
        self.temperature = temperature
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, features):
        """
        :param features: Tensor of shape [2 * batch_size, feature_dim],
                         where 2 * batch_size includes anchor and positive pairs.
        :return: InfoNCE loss
        """
        batch_size = features.shape[0] // 2
        labels = torch.cat([torch.arange(batch_size) for _ in range(2)]).to(features.device)
        features = nn.functional.normalize(features, dim=-1)

        # Cosine similarity matrix
        similarity_matrix = torch.matmul(features, features.T)

        # Mask out self-similarity
        self_mask  = torch.eye(2 * batch_size, device=features.device).bool()
        similarity_matrix = similarity_matrix.masked_fill(self_mask , -float('inf'))

        # Positive mask: roll the eye matrix to find positive pairs
        pos_mask = self_mask.roll(shifts=batch_size, dims=0)

        # Scale similarity scores by temperature
        similarity_matrix = similarity_matrix / self.temperature

        # Compute InfoNCE loss
        nll = -similarity_matrix[pos_mask] + torch.logsumexp(similarity_matrix, dim=-1)
        loss = nll.mean()

        return loss