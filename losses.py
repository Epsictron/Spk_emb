import torch
import torch.nn as nn
import torch.nn.functional as F


class AAMSoftmaxLoss(nn.Module):
    def __init__(self, embedding_dim, num_speakers, margin=0.2, scale=30):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_speakers, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        self.margin = margin
        self.scale = scale
        self.ce = nn.CrossEntropyLoss()

    def forward(self, embeddings, labels):
        embeddings = F.normalize(embeddings, dim=1, eps=1e-8)
        weight = F.normalize(self.weight, dim=1, eps=1e-8)
        cosine = F.linear(embeddings, weight)

        # Clamp cosine to avoid numerical issues with arccos-like operations
        cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        one_hot = F.one_hot(labels, cosine.size(1)).float()
        cosine = cosine - one_hot * self.margin

        logits = cosine * self.scale
        return self.ce(logits, labels)


class PrototypicalLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, embeddings, labels):
        embeddings = F.normalize(embeddings, dim=1, eps=1e-8)
        unique_labels = labels.unique()
        prototypes = torch.stack([embeddings[labels == l].mean(0) for l in unique_labels])
        prototypes = F.normalize(prototypes, dim=1, eps=1e-8)

        label_map = {l.item(): i for i, l in enumerate(unique_labels)}
        mapped = torch.tensor([label_map[l.item()] for l in labels], device=labels.device)

        dists = torch.cdist(embeddings, prototypes)
        return F.cross_entropy(-dists, mapped)


class ContrastiveLoss(nn.Module):
    """Simple contrastive loss using in-batch negatives."""

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, emb, labels):
        emb_norm = F.normalize(emb, dim=1, eps=1e-8)
        sim = torch.mm(emb_norm, emb_norm.t()) / self.temperature

        mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        mask.fill_diagonal_(0)

        # Use logsumexp for numerical stability (temperature=0.07 makes exp overflow)
        # Set diagonal to -inf so self-similarity is excluded from denominator
        sim_for_denom = sim.clone()
        sim_for_denom.fill_diagonal_(float("-inf"))
        log_denom = torch.logsumexp(sim_for_denom, dim=1, keepdim=True)

        log_prob = sim - log_denom
        pos_count = mask.sum(dim=1)
        loss = -(mask * log_prob).sum(dim=1) / (pos_count + 1e-9)
        valid = pos_count > 0
        if valid.sum() == 0:
            return torch.tensor(0.0, device=emb.device, requires_grad=True)
        return loss[valid].mean()


class CombinedLoss(nn.Module):
    """Weighted combination of AAMSoftmax and Prototypical loss."""

    def __init__(self, embedding_dim, num_speakers, aam_weight=0.7, proto_weight=0.3,
                 margin=0.2, scale=30):
        super().__init__()
        self.aam = AAMSoftmaxLoss(embedding_dim, num_speakers, margin, scale)
        self.proto = PrototypicalLoss()
        self.aam_weight = aam_weight
        self.proto_weight = proto_weight

    def forward(self, embeddings, labels):
        loss_aam = self.aam(embeddings, labels)
        loss_proto = self.proto(embeddings, labels)
        return self.aam_weight * loss_aam + self.proto_weight * loss_proto
