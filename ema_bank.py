"""EMA Memory Bank for speaker embeddings with diagnostic loss computation."""

import random
import torch
import torch.nn.functional as F


class EMAMemoryBank:
    """Maintains EMA-updated speaker embeddings and computes diagnostic losses.

    Diagnostic losses:
    - M-self: avg distance of male batch embeddings to their EMA bank embeddings
    - F-self: avg distance of female batch embeddings to their EMA bank embeddings
    - M-mix: avg pairwise distance among male speakers in the bank (sampled)
    - F-mix: avg pairwise distance among female speakers in the bank (sampled)

    Dynamic ratio monitoring:
    - Computes recommended male/female ratio based on diagnostic scores
    - Logs what the ratio WOULD be (does not apply changes)
    """

    def __init__(self, num_speakers, embedding_dim, spk2label, spk2gender,
                 ema_alpha=0.01, cold_speaker_limit=500, mix_sample_size=500,
                 diag_score_alpha=0.05, ratio_thresholds=None, ratio_options=None):
        """
        Args:
            num_speakers: total number of speakers
            embedding_dim: size of speaker embeddings
            spk2label: dict mapping speaker_id -> label index
            spk2gender: dict mapping speaker_id -> 'male'/'female'
            ema_alpha: EMA update rate (new = (1-alpha)*old + alpha*current)
            cold_speaker_limit: skip speakers not seen in last N steps
            mix_sample_size: number of random speakers to sample for mix computation
            diag_score_alpha: EMA smoothing rate for diagnostic scores
            ratio_thresholds: [low, high] thresholds for ratio changes
            ratio_options: list of male% options e.g. [30, 40, 50, 60, 70]
        """
        self.num_speakers = num_speakers
        self.embedding_dim = embedding_dim
        self.ema_alpha = ema_alpha
        self.cold_speaker_limit = cold_speaker_limit
        self.mix_sample_size = mix_sample_size
        self.diag_score_alpha = diag_score_alpha

        self.ratio_thresholds = ratio_thresholds or [0.55, 0.65]
        self.ratio_options = ratio_options or [30, 40, 50, 60, 70]
        self.current_ratio_idx = len(self.ratio_options) // 2  # start at 50/50

        # Bank: (num_speakers, embedding_dim) - initialized to zeros
        self.bank = torch.zeros(num_speakers, embedding_dim)
        self.last_seen = torch.zeros(num_speakers, dtype=torch.long)  # step when last updated
        self.initialized = torch.zeros(num_speakers, dtype=torch.bool)

        # Speaker gender lookup by label index
        self.label2gender = {}  # label_idx -> 'male'/'female'
        for spk_id, label in spk2label.items():
            gender = spk2gender.get(spk_id, "").lower()
            if gender in ("male", "female"):
                self.label2gender[label] = gender

        self.male_labels = [l for l, g in self.label2gender.items() if g == "male"]
        self.female_labels = [l for l, g in self.label2gender.items() if g == "female"]

        # EMA-smoothed diagnostic scores
        self.ema_m_self = 0.0
        self.ema_f_self = 0.0
        self.ema_m_mix = 0.0
        self.ema_f_mix = 0.0

    def to(self, device):
        """Move bank tensors to device."""
        self.bank = self.bank.to(device)
        self.last_seen = self.last_seen.to(device)
        self.initialized = self.initialized.to(device)
        return self

    @torch.no_grad()
    def update(self, embeddings, labels, step):
        """Update bank with EMA for speakers in the batch.

        Args:
            embeddings: (B, D) normalized embeddings from encoder
            labels: (B,) speaker label indices
            step: current training step
        """
        emb_norm = F.normalize(embeddings.float(), dim=1)

        unique_labels = labels.unique()
        for label in unique_labels:
            label_idx = label.item()
            mask = labels == label
            mean_emb = emb_norm[mask].mean(dim=0)

            if not self.initialized[label_idx]:
                self.bank[label_idx] = mean_emb
                self.initialized[label_idx] = True
            else:
                self.bank[label_idx] = (
                    (1 - self.ema_alpha) * self.bank[label_idx] +
                    self.ema_alpha * mean_emb
                )
                # Re-normalize after EMA update
                self.bank[label_idx] = F.normalize(self.bank[label_idx].unsqueeze(0), dim=1).squeeze(0)

            self.last_seen[label_idx] = step

    @torch.no_grad()
    def compute_diagnostics(self, embeddings, labels, gender_indices, step):
        """Compute diagnostic losses from batch + bank.

        Args:
            embeddings: (B, D) embeddings from current batch
            labels: (B,) speaker labels
            gender_indices: (B,) 0=male, 1=female
            step: current step

        Returns:
            dict with m_self, f_self, m_mix, f_mix scores and recommended_ratio
        """
        emb_norm = F.normalize(embeddings.float(), dim=1)

        # --- Self losses: batch embedding vs bank embedding ---
        m_self_dists = []
        f_self_dists = []

        unique_labels = labels.unique()
        for label in unique_labels:
            label_idx = label.item()
            if not self.initialized[label_idx]:
                continue

            mask = labels == label
            batch_mean = emb_norm[mask].mean(dim=0)
            bank_emb = self.bank[label_idx]

            # Cosine distance = 1 - cosine_similarity
            cos_sim = F.cosine_similarity(batch_mean.unsqueeze(0), bank_emb.unsqueeze(0)).item()
            dist = 1.0 - cos_sim

            gender = self.label2gender.get(label_idx, "")
            if gender == "male":
                m_self_dists.append(dist)
            elif gender == "female":
                f_self_dists.append(dist)

        m_self = sum(m_self_dists) / max(len(m_self_dists), 1)
        f_self = sum(f_self_dists) / max(len(f_self_dists), 1)

        # --- Mix losses: pairwise distances among bank speakers (sampled) ---
        m_mix = self._compute_mix("male", step)
        f_mix = self._compute_mix("female", step)

        # EMA smooth the scores
        self.ema_m_self = (1 - self.diag_score_alpha) * self.ema_m_self + self.diag_score_alpha * m_self
        self.ema_f_self = (1 - self.diag_score_alpha) * self.ema_f_self + self.diag_score_alpha * f_self
        self.ema_m_mix = (1 - self.diag_score_alpha) * self.ema_m_mix + self.diag_score_alpha * m_mix
        self.ema_f_mix = (1 - self.diag_score_alpha) * self.ema_f_mix + self.diag_score_alpha * f_mix

        # Compute recommended ratio (but don't apply)
        recommended_ratio = self._compute_recommended_ratio()

        return {
            "m_self": m_self,
            "f_self": f_self,
            "m_mix": m_mix,
            "f_mix": f_mix,
            "ema_m_self": self.ema_m_self,
            "ema_f_self": self.ema_f_self,
            "ema_m_mix": self.ema_m_mix,
            "ema_f_mix": self.ema_f_mix,
            "recommended_ratio": recommended_ratio,
            "current_ratio_idx": self.current_ratio_idx,
        }

    def _compute_mix(self, gender, step):
        """Compute avg pairwise cosine distance among bank speakers of given gender.

        Only uses warm (non-cold) speakers. Samples mix_sample_size speakers.
        """
        if gender == "male":
            all_labels = self.male_labels
        else:
            all_labels = self.female_labels

        # Filter cold speakers (not seen in last cold_speaker_limit steps)
        warm_labels = [
            l for l in all_labels
            if self.initialized[l] and (step - self.last_seen[l].item()) <= self.cold_speaker_limit
        ]

        if len(warm_labels) < 2:
            return 0.0

        # Sample if too many
        if len(warm_labels) > self.mix_sample_size:
            sampled = random.sample(warm_labels, self.mix_sample_size)
        else:
            sampled = warm_labels

        # Get bank embeddings for sampled speakers
        indices = torch.tensor(sampled, device=self.bank.device)
        embs = self.bank[indices]  # (N, D)
        embs = F.normalize(embs, dim=1)

        # Pairwise cosine similarity matrix
        sim_matrix = torch.mm(embs, embs.t())  # (N, N)

        # Extract upper triangle (exclude diagonal)
        n = sim_matrix.size(0)
        mask = torch.triu(torch.ones(n, n, device=sim_matrix.device), diagonal=1).bool()
        pairwise_sims = sim_matrix[mask]

        if pairwise_sims.numel() == 0:
            return 0.0

        # Average cosine distance
        avg_dist = (1.0 - pairwise_sims.mean()).item()
        return avg_dist

    def _compute_recommended_ratio(self):
        """Compute what the ratio WOULD be based on diagnostic scores.

        Logic: if male loss is relatively higher than female, recommend more male speakers.
        Uses one-step-at-a-time changes.

        Returns the recommended male% from ratio_options.
        """
        # Combine self + mix for each gender
        m_score = self.ema_m_self + self.ema_m_mix
        f_score = self.ema_f_self + self.ema_f_mix

        total = m_score + f_score
        if total < 1e-9:
            return self.ratio_options[self.current_ratio_idx]

        # Male proportion of total diagnostic score
        m_proportion = m_score / total

        low_thresh, high_thresh = self.ratio_thresholds

        # One-step-at-a-time: if male score is high, move ratio toward more male
        if m_proportion > high_thresh and self.current_ratio_idx < len(self.ratio_options) - 1:
            self.current_ratio_idx += 1
        elif m_proportion < low_thresh and self.current_ratio_idx > 0:
            self.current_ratio_idx -= 1

        return self.ratio_options[self.current_ratio_idx]

    def get_bank_stats(self):
        """Return summary stats about the bank."""
        n_init = self.initialized.sum().item()
        n_male_init = sum(1 for l in self.male_labels if self.initialized[l])
        n_female_init = sum(1 for l in self.female_labels if self.initialized[l])
        return {
            "total_initialized": n_init,
            "male_initialized": n_male_init,
            "female_initialized": n_female_init,
            "total_speakers": self.num_speakers,
        }

    def state_dict(self):
        """For checkpoint saving."""
        return {
            "bank": self.bank.cpu(),
            "last_seen": self.last_seen.cpu(),
            "initialized": self.initialized.cpu(),
            "ema_m_self": self.ema_m_self,
            "ema_f_self": self.ema_f_self,
            "ema_m_mix": self.ema_m_mix,
            "ema_f_mix": self.ema_f_mix,
            "current_ratio_idx": self.current_ratio_idx,
        }

    def load_state_dict(self, state):
        """For checkpoint loading."""
        self.bank = state["bank"].to(self.bank.device)
        self.last_seen = state["last_seen"].to(self.last_seen.device)
        self.initialized = state["initialized"].to(self.initialized.device)
        self.ema_m_self = state["ema_m_self"]
        self.ema_f_self = state["ema_f_self"]
        self.ema_m_mix = state["ema_m_mix"]
        self.ema_f_mix = state["ema_f_mix"]
        self.current_ratio_idx = state["current_ratio_idx"]
