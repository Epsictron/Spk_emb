import torch
import torch.nn as nn
import torch.nn.functional as F


class SpeakerEncoder(nn.Module):
    def __init__(self, n_mels=80, embedding_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_mels, 512, 5, padding=2), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Conv1d(512, 512, 3, padding=1), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Conv1d(512, 512, 3, padding=1), nn.BatchNorm1d(512), nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, embedding_dim)

    def forward(self, x):
        # x: (B, n_mels, T)
        x = self.conv(x)
        x = self.pool(x).squeeze(-1)  # (B, 512)
        x = self.fc(x)  # (B, embedding_dim)
        return x


class AAMSoftmaxLoss(nn.Module):
    def __init__(self, embedding_dim, num_speakers, margin=0.2, scale=30):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_speakers, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        self.margin = margin
        self.scale = scale
        self.ce = nn.CrossEntropyLoss()

    def forward(self, embeddings, labels):
        # Normalize
        embeddings = F.normalize(embeddings, dim=1)
        weight = F.normalize(self.weight, dim=1)
        cosine = F.linear(embeddings, weight)

        # Add angular margin to target
        one_hot = F.one_hot(labels, cosine.size(1)).float()
        cosine = cosine - one_hot * self.margin

        logits = cosine * self.scale
        return self.ce(logits, labels)


class PrototypicalLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, embeddings, labels):
        unique_labels = labels.unique()
        prototypes = torch.stack([embeddings[labels == l].mean(0) for l in unique_labels])

        # Remap labels to 0..N-1
        label_map = {l.item(): i for i, l in enumerate(unique_labels)}
        mapped = torch.tensor([label_map[l.item()] for l in labels], device=labels.device)

        dists = torch.cdist(embeddings, prototypes)  # (B, N_classes)
        return F.cross_entropy(-dists, mapped)
