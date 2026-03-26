import torch
import torch.nn as nn


class SpeakerEncoder(nn.Module):
    """Simple CNN-based speaker encoder."""

    def __init__(self, in_channels, emb_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 512, 5, padding=2),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 512, 3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 512, 3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, emb_dim)

    def forward(self, x):
        # x: (batch, freq, time)
        x = self.conv(x)
        x = self.pool(x).squeeze(-1)
        x = self.fc(x)
        return x


class AAMSoftmax(nn.Module):
    """Additive Angular Margin Softmax loss."""

    def __init__(self, emb_dim, num_classes, margin=0.2, scale=30.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_classes, emb_dim))
        nn.init.xavier_uniform_(self.weight)
        self.margin = margin
        self.scale = scale
        self.ce = nn.CrossEntropyLoss()

    def forward(self, emb, labels):
        # normalize
        emb_norm = nn.functional.normalize(emb, dim=1)
        w_norm = nn.functional.normalize(self.weight, dim=1)
        cosine = torch.mm(emb_norm, w_norm.t())

        # add margin to target class
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
        cosine = cosine - one_hot * self.margin

        logits = cosine * self.scale
        return self.ce(logits, labels)


class ContrastiveLoss(nn.Module):
    """Simple contrastive loss using in-batch negatives."""

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, emb, labels):
        emb_norm = nn.functional.normalize(emb, dim=1)
        sim = torch.mm(emb_norm, emb_norm.t()) / self.temperature

        # mask: same speaker = positive
        mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        mask.fill_diagonal_(0)

        # for each anchor, compute loss over positives vs all
        exp_sim = torch.exp(sim)
        exp_sim.fill_diagonal_(0)
        denom = exp_sim.sum(dim=1, keepdim=True)

        log_prob = sim - torch.log(denom + 1e-9)
        # mean over positive pairs
        pos_count = mask.sum(dim=1)
        loss = -(mask * log_prob).sum(dim=1) / (pos_count + 1e-9)
        # only count anchors that have at least one positive
        valid = pos_count > 0
        if valid.sum() == 0:
            return torch.tensor(0.0, device=emb.device, requires_grad=True)
        return loss[valid].mean()
