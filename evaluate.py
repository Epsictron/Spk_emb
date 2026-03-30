"""Speaker embedding evaluation: similarity distributions + KDE + TensorBoard logging.

Computes 5 cosine similarity distributions from a held-out manifest:
  M-self, F-self   — within-speaker pairwise similarities
  M-mix, F-mix, MF — between-speaker centroid similarities

Logs mean, variance, 95% coverage bounds, and separation metric.
"""

import json
import random

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
from scipy.stats import gaussian_kde


class _EvalDataset(Dataset):
    """Lightweight dataset for evaluation: loads audio, extracts features."""

    def __init__(self, entries, cfg):
        self.entries = entries
        self.sample_rate = cfg["sample_rate"]
        self.segment_len = int(self.sample_rate * cfg["segment_duration"])

        feature_type = cfg.get("feature_type", "melspectrogram")
        if feature_type == "mfcc":
            self.transform = torchaudio.transforms.MFCC(
                sample_rate=self.sample_rate, n_mfcc=40,
                melkwargs={"n_fft": cfg["n_fft"], "hop_length": cfg["hop_length"],
                           "win_length": cfg["win_length"], "n_mels": cfg["n_mels"]},
            )
        else:
            self.transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sample_rate, n_fft=cfg["n_fft"],
                hop_length=cfg["hop_length"], win_length=cfg["win_length"],
                n_mels=cfg["n_mels"],
            )

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        wav, sr = torchaudio.load(entry["audio_file_path"])

        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        wav = wav.squeeze(0)

        # Fixed segment: crop or loop-repeat
        if wav.size(0) > self.segment_len:
            start = random.randint(0, wav.size(0) - self.segment_len)
            wav = wav[start:start + self.segment_len]
        elif wav.size(0) < self.segment_len:
            repeats = self.segment_len // wav.size(0) + 1
            wav = wav.repeat(repeats)[:self.segment_len]

        feat = self.transform(wav)
        feat = torch.log(feat + 1e-9)  # (n_mels, T)
        return feat, idx  # return idx to map back to speaker


def _extract_embeddings(model, manifest, cfg, device, max_samples_per_speaker=10):
    """Extract L2-normalized embeddings grouped by speaker using DataLoader.

    Returns:
        spk_embeddings: dict {speaker_id: tensor (N, emb_dim)}
        spk_genders:    dict {speaker_id: "male" or "female"}
    """
    # Group manifest entries by speaker, cap per speaker
    spk_entries = {}
    spk_genders = {}
    for entry in manifest:
        sid = entry["speaker_id"]
        spk_entries.setdefault(sid, []).append(entry)
        if sid not in spk_genders:
            spk_genders[sid] = entry.get("gender", "").lower()

    for sid in spk_entries:
        entries = spk_entries[sid]
        if len(entries) > max_samples_per_speaker:
            spk_entries[sid] = random.sample(entries, max_samples_per_speaker)

    # Flatten into a single list with speaker tracking
    flat_entries = []
    flat_spk_ids = []
    for sid, entries in spk_entries.items():
        for entry in entries:
            flat_entries.append(entry)
            flat_spk_ids.append(sid)

    # DataLoader for parallel audio loading
    eval_ds = _EvalDataset(flat_entries, cfg)
    num_workers = min(cfg.get("num_workers", 4), 4)
    eval_loader = DataLoader(
        eval_ds, batch_size=64, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    # Batch inference
    all_embeddings = []
    model.eval()
    with torch.no_grad():
        for feats, indices in eval_loader:
            feats = feats.to(device)
            embs = model(feats)
            embs = F.normalize(embs.float(), dim=1, eps=1e-8)
            all_embeddings.append(embs.cpu())

    all_embeddings = torch.cat(all_embeddings, dim=0)  # (total_samples, emb_dim)

    # Group embeddings back by speaker
    spk_embeddings = {}
    for i, sid in enumerate(flat_spk_ids):
        if sid not in spk_embeddings:
            spk_embeddings[sid] = []
        spk_embeddings[sid].append(all_embeddings[i])

    for sid in spk_embeddings:
        spk_embeddings[sid] = torch.stack(spk_embeddings[sid])

    return spk_embeddings, spk_genders


def _pairwise_cosine(a, b):
    """Cosine similarity between all pairs in a and b. Returns 1D tensor."""
    # a: (N, D), b: (M, D) -> (N, M)
    a = F.normalize(a, dim=1, eps=1e-8)
    b = F.normalize(b, dim=1, eps=1e-8)
    sim = torch.mm(a, b.t())
    return sim.flatten()


def _within_speaker_sims(embeddings):
    """All pairwise cosine similarities within a set of embeddings (upper triangle only)."""
    n = embeddings.size(0)
    if n < 2:
        return torch.tensor([])
    sims = torch.mm(
        F.normalize(embeddings, dim=1, eps=1e-8),
        F.normalize(embeddings, dim=1, eps=1e-8).t()
    )
    # Upper triangle (exclude diagonal)
    idx = torch.triu_indices(n, n, offset=1)
    return sims[idx[0], idx[1]]


def _kde_stats(values):
    """Fit Gaussian KDE and return mean, var, 2.5th percentile, 97.5th percentile."""
    if len(values) < 5:
        return {"mean": 0.0, "var": 0.0, "p2_5": 0.0, "p97_5": 0.0}

    arr = np.array(values, dtype=np.float64)
    mean = float(arr.mean())
    var = float(arr.var())

    try:
        kde = gaussian_kde(arr)
        # Sample from KDE to get percentiles
        samples = kde.resample(10000, seed=42).flatten()
        p2_5 = float(np.percentile(samples, 2.5))
        p97_5 = float(np.percentile(samples, 97.5))
    except Exception:
        # Fallback if KDE fails (e.g., all identical values)
        p2_5 = float(np.percentile(arr, 2.5))
        p97_5 = float(np.percentile(arr, 97.5))

    return {"mean": mean, "var": var, "p2_5": p2_5, "p97_5": p97_5}


def evaluate(model, val_manifest, cfg, device, writer=None, step=0):
    """Run full evaluation and log to TensorBoard.

    Args:
        model: SpeakerEncoder (will be set to eval mode)
        val_manifest: list of manifest entries (already filtered)
        cfg: config dict
        device: torch device
        writer: TensorBoard SummaryWriter (optional)
        step: current training step

    Returns:
        dict with all metrics
    """
    max_samples = cfg.get("eval_samples_per_speaker", 10)

    print(f"\n[Eval @ step {step}] Extracting embeddings...")
    spk_embs, spk_genders = _extract_embeddings(
        model, val_manifest, cfg, device, max_samples_per_speaker=max_samples
    )

    # Split speakers by gender
    male_spks = [s for s, g in spk_genders.items() if g == "male" and s in spk_embs]
    female_spks = [s for s, g in spk_genders.items() if g == "female" and s in spk_embs]

    print(f"  Speakers with embeddings: {len(male_spks)}M + {len(female_spks)}F = {len(spk_embs)}")

    # --- Self distributions (within-speaker pairwise) ---
    m_self_sims = []
    for sid in male_spks:
        sims = _within_speaker_sims(spk_embs[sid])
        if len(sims) > 0:
            m_self_sims.append(sims)
    m_self_sims = torch.cat(m_self_sims).numpy() if m_self_sims else np.array([])

    f_self_sims = []
    for sid in female_spks:
        sims = _within_speaker_sims(spk_embs[sid])
        if len(sims) > 0:
            f_self_sims.append(sims)
    f_self_sims = torch.cat(f_self_sims).numpy() if f_self_sims else np.array([])

    # --- Mix distributions (between-speaker centroid pairwise) ---
    # Compute centroids
    male_centroids = []
    for sid in male_spks:
        centroid = F.normalize(spk_embs[sid].mean(dim=0, keepdim=True), dim=1, eps=1e-8)
        male_centroids.append(centroid)
    male_centroids = torch.cat(male_centroids) if male_centroids else torch.zeros(0, cfg["embedding_dim"])

    female_centroids = []
    for sid in female_spks:
        centroid = F.normalize(spk_embs[sid].mean(dim=0, keepdim=True), dim=1, eps=1e-8)
        female_centroids.append(centroid)
    female_centroids = torch.cat(female_centroids) if female_centroids else torch.zeros(0, cfg["embedding_dim"])

    # M-mix: pairwise between different male centroids
    if len(male_centroids) >= 2:
        m_mix_sim = torch.mm(male_centroids, male_centroids.t())
        n = male_centroids.size(0)
        idx = torch.triu_indices(n, n, offset=1)
        m_mix_sims = m_mix_sim[idx[0], idx[1]].numpy()
    else:
        m_mix_sims = np.array([])

    # F-mix: pairwise between different female centroids
    if len(female_centroids) >= 2:
        f_mix_sim = torch.mm(female_centroids, female_centroids.t())
        n = female_centroids.size(0)
        idx = torch.triu_indices(n, n, offset=1)
        f_mix_sims = f_mix_sim[idx[0], idx[1]].numpy()
    else:
        f_mix_sims = np.array([])

    # MF: pairwise between male and female centroids
    if len(male_centroids) > 0 and len(female_centroids) > 0:
        mf_sims = torch.mm(male_centroids, female_centroids.t()).flatten().numpy()
    else:
        mf_sims = np.array([])

    # --- KDE stats ---
    stats = {
        "m_self": _kde_stats(m_self_sims),
        "f_self": _kde_stats(f_self_sims),
        "m_mix": _kde_stats(m_mix_sims),
        "f_mix": _kde_stats(f_mix_sims),
        "mf": _kde_stats(mf_sims),
    }

    # --- Separation metric ---
    # 95% lower bound for self (2.5th percentile)
    # 95% upper bound for mix (97.5th percentile)
    self_min = min(stats["m_self"]["p2_5"], stats["f_self"]["p2_5"])
    mix_max = max(stats["m_mix"]["p97_5"], stats["f_mix"]["p97_5"], stats["mf"]["p97_5"])
    separation = self_min - mix_max

    # --- Print ---
    print(f"  M-self:  mean={stats['m_self']['mean']:.4f}  var={stats['m_self']['var']:.6f}  "
          f"95%=[{stats['m_self']['p2_5']:.4f}, {stats['m_self']['p97_5']:.4f}]")
    print(f"  F-self:  mean={stats['f_self']['mean']:.4f}  var={stats['f_self']['var']:.6f}  "
          f"95%=[{stats['f_self']['p2_5']:.4f}, {stats['f_self']['p97_5']:.4f}]")
    print(f"  M-mix:   mean={stats['m_mix']['mean']:.4f}  var={stats['m_mix']['var']:.6f}  "
          f"95%=[{stats['m_mix']['p2_5']:.4f}, {stats['m_mix']['p97_5']:.4f}]")
    print(f"  F-mix:   mean={stats['f_mix']['mean']:.4f}  var={stats['f_mix']['var']:.6f}  "
          f"95%=[{stats['f_mix']['p2_5']:.4f}, {stats['f_mix']['p97_5']:.4f}]")
    print(f"  MF:      mean={stats['mf']['mean']:.4f}  var={stats['mf']['var']:.6f}  "
          f"95%=[{stats['mf']['p2_5']:.4f}, {stats['mf']['p97_5']:.4f}]")
    print(f"  Separation (min_self_2.5% - max_mix_97.5%): {separation:.4f}")

    # --- TensorBoard logging ---
    if writer is not None:
        # Mean and variance
        for name in ["m_self", "f_self", "m_mix", "f_mix", "mf"]:
            writer.add_scalar(f"eval/{name}_mean", stats[name]["mean"], step)
            writer.add_scalar(f"eval/{name}_var", stats[name]["var"], step)
            writer.add_scalar(f"eval/{name}_p2.5", stats[name]["p2_5"], step)
            writer.add_scalar(f"eval/{name}_p97.5", stats[name]["p97_5"], step)

        # Separation metric
        writer.add_scalar("eval/separation", separation, step)
        writer.add_scalar("eval/self_min_p2.5", self_min, step)
        writer.add_scalar("eval/mix_max_p97.5", mix_max, step)

    results = {
        "stats": stats,
        "separation": separation,
        "self_min_p2_5": self_min,
        "mix_max_p97_5": mix_max,
        "n_male_speakers": len(male_spks),
        "n_female_speakers": len(female_spks),
        "n_m_self_pairs": len(m_self_sims),
        "n_f_self_pairs": len(f_self_sims),
        "n_m_mix_pairs": len(m_mix_sims),
        "n_f_mix_pairs": len(f_mix_sims),
        "n_mf_pairs": len(mf_sims),
    }

    return results
