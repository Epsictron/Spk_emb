import torch
import torch.nn as nn


class SpeakerEncoder(nn.Module):
    def __init__(self, n_mels=80, embedding_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_mels, 512, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(512), nn.ReLU(),
            nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(512), nn.ReLU(),
            nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(512), nn.ReLU(),
        )
        self.lstm = nn.LSTM(512, 512, num_layers=2, batch_first=True, dropout=0.1)
        self.attention = nn.Sequential(
            nn.Linear(512, 256), nn.Tanh(), nn.Linear(256, 1),
        )
        self.fc = nn.Linear(512, embedding_dim, bias=False)

    def forward(self, x):
        x = self.conv(x)             # (B, 512, T')
        x = x.transpose(1, 2)        # (B, T', 512)
        x, _ = self.lstm(x)          # (B, T', 512)
        w = torch.softmax(self.attention(x), dim=1)  # (B, T', 1)
        x = (x * w).sum(dim=1)       # (B, 512)
        return self.fc(x)            # (B, embedding_dim)
