import torch.nn as nn


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
