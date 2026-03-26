import json
import random

import torch
import torchaudio
from torch.utils.data import Dataset, Sampler


class SpeakerDataset(Dataset):
    def __init__(self, entries, sample_rate=16000, max_duration=3.0, feature="melspec", n_mels=80, n_mfcc=40):
        self.entries = entries
        self.sample_rate = sample_rate
        self.max_samples = int(max_duration * sample_rate)
        self.feature = feature
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc

        # build speaker to index mapping
        speakers = sorted(set(e["speaker_id"] for e in entries))
        self.spk2idx = {s: i for i, s in enumerate(speakers)}
        self.num_speakers = len(speakers)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        wav, sr = torchaudio.load(entry["audio_file_path"])

        # resample if needed
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)

        # mono
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        # trim or pad to fixed length
        if wav.shape[1] > self.max_samples:
            start = random.randint(0, wav.shape[1] - self.max_samples)
            wav = wav[:, start:start + self.max_samples]
        else:
            wav = torch.nn.functional.pad(wav, (0, self.max_samples - wav.shape[1]))

        # extract feature
        feat = self._extract_feature(wav)
        label = self.spk2idx[entry["speaker_id"]]
        return feat, label

    def _extract_feature(self, wav):
        if self.feature == "melspec":
            transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sample_rate, n_mels=self.n_mels
            )
            feat = transform(wav)
            feat = torch.log(feat + 1e-9)
        elif self.feature == "mfcc":
            transform = torchaudio.transforms.MFCC(
                sample_rate=self.sample_rate, n_mfcc=self.n_mfcc
            )
            feat = transform(wav)
        return feat.squeeze(0)  # (freq, time)


class GenderBalancedSampler(Sampler):
    """Ensures roughly equal male/female samples per epoch."""

    def __init__(self, entries, batch_size):
        self.batch_size = batch_size
        self.male_indices = [i for i, e in enumerate(entries) if e.get("gender", "").lower() == "male"]
        self.female_indices = [i for i, e in enumerate(entries) if e.get("gender", "").lower() == "female"]
        self.other_indices = [i for i, e in enumerate(entries)
                              if e.get("gender", "").lower() not in ("male", "female")]
        n = max(len(self.male_indices), len(self.female_indices))
        self._len = n * 2 + len(self.other_indices)

    def __iter__(self):
        n = max(len(self.male_indices), len(self.female_indices))
        males = self._resample(self.male_indices, n)
        females = self._resample(self.female_indices, n)
        combined = males + females + self.other_indices
        random.shuffle(combined)
        return iter(combined)

    def __len__(self):
        return self._len

    @staticmethod
    def _resample(indices, target_len):
        if len(indices) == 0:
            return []
        result = []
        while len(result) < target_len:
            random.shuffle(indices)
            result.extend(indices)
        return result[:target_len]


def load_manifest(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data


def split_data(entries, val_split=0.1, seed=42):
    random.seed(seed)
    entries = list(entries)
    random.shuffle(entries)
    n_val = int(len(entries) * val_split)
    return entries[n_val:], entries[:n_val]
