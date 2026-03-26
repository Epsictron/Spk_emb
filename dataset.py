import json
import random

import torch
import torchaudio
from torch.utils.data import Dataset, Sampler


class SpeakerDataset(Dataset):
    def __init__(self, entries, sample_rate=16000, segment_duration=3.0,
                 feature_type="melspectrogram", n_mels=80, n_fft=512,
                 hop_length=160, win_length=400):
        self.entries = entries
        self.sample_rate = sample_rate
        self.segment_len = int(sample_rate * segment_duration)
        self.feature_type = feature_type

        # Build speaker-to-label mapping
        speakers = sorted(set(e["speaker_id"] for e in self.entries))
        self.spk2label = {s: i for i, s in enumerate(speakers)}
        self.num_speakers = len(speakers)

        # Feature extractor
        if feature_type == "mfcc":
            self.feature_fn = torchaudio.transforms.MFCC(
                sample_rate=sample_rate, n_mfcc=40,
                melkwargs={"n_fft": n_fft, "hop_length": hop_length,
                           "win_length": win_length, "n_mels": n_mels}
            )
        else:
            self.feature_fn = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate, n_fft=n_fft,
                hop_length=hop_length, win_length=win_length, n_mels=n_mels
            )

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        wav, sr = torchaudio.load(entry["audio_file_path"])
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)

        # mono
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        wav = wav[0]

        # Crop or pad to fixed length
        if wav.size(0) > self.segment_len:
            start = random.randint(0, wav.size(0) - self.segment_len)
            wav = wav[start:start + self.segment_len]
        else:
            wav = torch.nn.functional.pad(wav, (0, self.segment_len - wav.size(0)))

        features = self.feature_fn(wav)  # (n_mels, T)
        label = self.spk2label[entry["speaker_id"]]
        return features, label


class GenderBalancedSampler(Sampler):
    """Ensures equal representation of male and female samples per epoch."""

    def __init__(self, manifest):
        self.male_indices = [i for i, e in enumerate(manifest) if e.get("gender", "").lower() == "male"]
        self.female_indices = [i for i, e in enumerate(manifest) if e.get("gender", "").lower() == "female"]
        self.other_indices = [i for i, e in enumerate(manifest)
                              if e.get("gender", "").lower() not in ("male", "female")]

        if self.male_indices or self.female_indices:
            n = max(len(self.male_indices), len(self.female_indices))
            self.epoch_size = n * 2 + len(self.other_indices)
            self.balanced = True
        else:
            self.epoch_size = len(manifest)
            self.all_indices = list(range(len(manifest)))
            self.balanced = False

    def _resample(self, indices, target_len):
        if len(indices) == 0:
            return []
        result = []
        while len(result) < target_len:
            random.shuffle(indices)
            result.extend(indices)
        return result[:target_len]

    def __iter__(self):
        if not self.balanced:
            random.shuffle(self.all_indices)
            return iter(self.all_indices)
        n = max(len(self.male_indices), len(self.female_indices))
        males = self._resample(self.male_indices, n)
        females = self._resample(self.female_indices, n)
        combined = males + females + self.other_indices
        random.shuffle(combined)
        return iter(combined)

    def __len__(self):
        return self.epoch_size


def load_manifest(path):
    with open(path) as f:
        return json.load(f)


def split_manifest(manifest, val_split=0.1):
    """Split manifest by speaker so no speaker leaks between train/val."""
    spk_to_entries = {}
    for e in manifest:
        spk_to_entries.setdefault(e["speaker_id"], []).append(e)

    speakers = list(spk_to_entries.keys())
    random.shuffle(speakers)
    val_count = max(1, int(len(speakers) * val_split))

    val_spks = set(speakers[:val_count])
    train, val = [], []
    for spk, entries in spk_to_entries.items():
        if spk in val_spks:
            val.extend(entries)
        else:
            train.extend(entries)
    return train, val
