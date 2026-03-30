import torch
import torch.nn as nn
import torch.nn.functional as F


class AAMSoftmaxLoss(nn.Module):
    def __init__(self, embedding_dim, num_speakers, margin=0.2, scale=30,
                 label_smoothing=0.0):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(num_speakers, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        self.margin = margin
        self.scale = scale
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, embeddings, labels):
        emb = F.normalize(embeddings, dim=1, eps=1e-8)
        w = F.normalize(self.weight, dim=1, eps=1e-8)
        logits = F.linear(emb, w)  # (B, num_speakers)
        one_hot = F.one_hot(labels, self.weight.size(0)).float()
        logits = logits - one_hot * self.margin
        return self.ce(logits * self.scale, labels)
