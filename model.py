import torch
import torch.nn as nn


class SpeakerEncoder(nn.Module):
    """Causal speaker encoder: Conv1d front-end → LSTM → Attention pooling → FC.
    ~6M parameters with default settings.
    """

    def __init__(self, n_mels=80, embedding_dim=256):
        super().__init__()

        # Conv1d front-end: causal convolutions (left padding only)
        self.conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(n_mels, 512, kernel_size=5, padding=0),
                nn.BatchNorm1d(512), nn.ReLU(),
            ),
            nn.Sequential(
                nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=0),
                nn.BatchNorm1d(512), nn.ReLU(),
            ),
            nn.Sequential(
                nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=0),
                nn.BatchNorm1d(512), nn.ReLU(),
            ),
        ])
        # Causal padding sizes (kernel_size - 1 for each layer)
        self.causal_pad = [4, 2, 2]

        # LSTM (causal by nature)
        self.lstm = nn.LSTM(
            input_size=512, hidden_size=512,
            num_layers=2, batch_first=True, dropout=0.1,
        )

        # Attention pooling
        self.attention = nn.Sequential(
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
        )

        # Final projection
        self.fc = nn.Linear(512, embedding_dim, bias=False)

    def forward(self, x):
        # x: (B, n_mels, T)

        # Causal conv: pad left side only
        for conv, pad in zip(self.conv, self.causal_pad):
            x = nn.functional.pad(x, (pad, 0))  # left pad only
            x = conv(x)
        # x: (B, 512, T')

        # LSTM expects (B, T, C)
        x = x.transpose(1, 2)  # (B, T', 512)
        x, _ = self.lstm(x)    # (B, T', 512)

        # Attention pooling
        w = self.attention(x)          # (B, T', 1)
        w = torch.softmax(w, dim=1)    # (B, T', 1)
        x = (x * w).sum(dim=1)         # (B, 512)

        # Project to embedding
        x = self.fc(x)    # (B, embedding_dim)
        return x
