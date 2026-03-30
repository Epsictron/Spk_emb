"""Evaluation: 5 similarity distributions + KDE + separation metric."""

import random
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
from scipy.stats import gaussian_kde


class _EvalDataset(Dataset):
    def __init__(self, entries, cfg):
        self.entries = entries
        self.sample_rate = cfg["sample_rate"]
        self.segment_len = int(self.sample_rate * cfg["segment_duration"])
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

        if wav.size(0) > self.segment_len:
            start = random.randint(0, wav.size(0) - self.segment_len)
            wav = wav[start:start + self.segment_len]
        elif wav.size(0) < self.segment_len:
            repeats = self.segment_len // wav.size(0) + 1
            wav = wav.repeat(repeats)[:self.segment_len]

        feat = torch.log(self.transform(wav) + 1e-9)
        return feat, idx


def _kde_stats(values):
    if len(values) < 5:
        return {"mean": 0.0, "var": 0.0, "p2_5": 0.0, "p97_5": 0.0}
    arr = np.array(values, dtype=np.float64)
    mean, var = float(arr.mean()), float(arr.var())
    try:
        kde = gaussian_kde(arr)
        samples = kde.resample(10000, seed=42).flatten()
        p2_5 = float(np.percentile(samples, 2.5))
        p97_5 = float(np.percentile(samples, 97.5))
    except Exception:
        p2_5 = float(np.percentile(arr, 2.5))
        p97_5 = float(np.percentile(arr, 97.5))
    return {"mean": mean, "var": var, "p2_5": p2_5, "p97_5": p97_5}


def _within_speaker_sims(embeddings):
    n = embeddings.size(0)
    if n < 2:
        return torch.tensor([])
    normed = F.normalize(embeddings, dim=1, eps=1e-8)
    sims = torch.mm(normed, normed.t())
    idx = torch.triu_indices(n, n, offset=1)
    return sims[idx[0], idx[1]]


def evaluate(model, test_manifest, cfg, device, writer=None, step=0):
    max_samples = cfg.get("eval_samples_per_speaker", 10)

    # Group by speaker, cap samples
    spk_entries, spk_genders = {}, {}
    for entry in test_manifest:
        sid = entry["speaker_id"]
        spk_entries.setdefault(sid, []).append(entry)
        if sid not in spk_genders:
            spk_genders[sid] = entry.get("gender", "").lower()
    for sid in spk_entries:
        if len(spk_entries[sid]) > max_samples:
            spk_entries[sid] = random.sample(spk_entries[sid], max_samples)

    # Flatten for DataLoader
    flat_entries, flat_spk_ids = [], []
    for sid, entries in spk_entries.items():
        for e in entries:
            flat_entries.append(e)
            flat_spk_ids.append(sid)

    eval_ds = _EvalDataset(flat_entries, cfg)
    loader = DataLoader(eval_ds, batch_size=64, shuffle=False,
                        num_workers=min(cfg.get("num_workers", 4), 4), pin_memory=True)

    # Extract embeddings
    all_embs = []
    model.eval()
    with torch.no_grad():
        for feats, _ in loader:
            embs = model(feats.to(device))
            all_embs.append(F.normalize(embs.float(), dim=1, eps=1e-8).cpu())
    all_embs = torch.cat(all_embs)

    # Group by speaker
    spk_embs = {}
    for i, sid in enumerate(flat_spk_ids):
        spk_embs.setdefault(sid, []).append(all_embs[i])
    for sid in spk_embs:
        spk_embs[sid] = torch.stack(spk_embs[sid])

    male_spks = [s for s, g in spk_genders.items() if g == "male" and s in spk_embs]
    female_spks = [s for s, g in spk_genders.items() if g == "female" and s in spk_embs]
    print(f"  Eval speakers: {len(male_spks)}M + {len(female_spks)}F = {len(spk_embs)}")

    # Self distributions (within-speaker pairwise)
    m_self = [_within_speaker_sims(spk_embs[s]) for s in male_spks]
    m_self = torch.cat(m_self).numpy() if any(len(s) > 0 for s in m_self) else np.array([])
    f_self = [_within_speaker_sims(spk_embs[s]) for s in female_spks]
    f_self = torch.cat(f_self).numpy() if any(len(s) > 0 for s in f_self) else np.array([])

    # Centroids
    emb_dim = cfg["embedding_dim"]
    m_cent = torch.cat([F.normalize(spk_embs[s].mean(0, keepdim=True), dim=1) for s in male_spks]) if male_spks else torch.zeros(0, emb_dim)
    f_cent = torch.cat([F.normalize(spk_embs[s].mean(0, keepdim=True), dim=1) for s in female_spks]) if female_spks else torch.zeros(0, emb_dim)

    # Mix distributions (between-speaker centroids)
    def upper_tri_sims(centroids):
        if centroids.size(0) < 2:
            return np.array([])
        sims = torch.mm(centroids, centroids.t())
        n = centroids.size(0)
        idx = torch.triu_indices(n, n, offset=1)
        return sims[idx[0], idx[1]].numpy()

    m_mix = upper_tri_sims(m_cent)
    f_mix = upper_tri_sims(f_cent)
    mf_mix = torch.mm(m_cent, f_cent.t()).flatten().numpy() if m_cent.size(0) > 0 and f_cent.size(0) > 0 else np.array([])

    # KDE stats
    stats = {
        "m_self": _kde_stats(m_self), "f_self": _kde_stats(f_self),
        "m_mix": _kde_stats(m_mix), "f_mix": _kde_stats(f_mix), "mf": _kde_stats(mf_mix),
    }

    # Separation
    self_min = min(stats["m_self"]["p2_5"], stats["f_self"]["p2_5"])
    mix_max = max(stats["m_mix"]["p97_5"], stats["f_mix"]["p97_5"], stats["mf"]["p97_5"])
    separation = self_min - mix_max

    # Print
    for name in ["m_self", "f_self", "m_mix", "f_mix", "mf"]:
        s = stats[name]
        print(f"  {name:8s}: mean={s['mean']:.4f}  var={s['var']:.6f}  95%=[{s['p2_5']:.4f}, {s['p97_5']:.4f}]")
    print(f"  Separation: {separation:.4f}")

    # TensorBoard
    if writer:
        for name in ["m_self", "f_self", "m_mix", "f_mix", "mf"]:
            writer.add_scalar(f"eval/{name}_mean", stats[name]["mean"], step)
            writer.add_scalar(f"eval/{name}_var", stats[name]["var"], step)
            writer.add_scalar(f"eval/{name}_p2.5", stats[name]["p2_5"], step)
            writer.add_scalar(f"eval/{name}_p97.5", stats[name]["p97_5"], step)
        writer.add_scalar("eval/separation", separation, step)

    return {"stats": stats, "separation": separation}
