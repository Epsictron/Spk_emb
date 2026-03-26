import json
import random

import torch
import torchaudio
from torch.utils.data import Dataset, Sampler


class SpeakerDataset(Dataset):
    def __init__(self, entries, sample_rate=16000, segment_duration=3.0,
                 feature_type="melspectrogram", n_mels=80, n_fft=512,
                 hop_length=160, win_length=400, spk2label=None):
        self.entries = entries
        self.sample_rate = sample_rate
        self.segment_len = int(sample_rate * segment_duration)
        self.feature_type = feature_type

        # Use provided mapping or build from entries
        if spk2label is not None:
            self.spk2label = spk2label
        else:
            speakers = sorted(set(e["speaker_id"] for e in self.entries))
            self.spk2label = {s: i for i, s in enumerate(speakers)}
        self.num_speakers = len(self.spk2label)

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
        spk = entry["speaker_id"]
        if spk not in self.spk2label:
            raise KeyError(f"Speaker '{spk}' not in spk2label. Available: {len(self.spk2label)} speakers. "
                           f"File: {entry['audio_file_path']}")
        label = self.spk2label[spk]
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


class SpeakerBatchSampler(Sampler):
    """Samples fixed number of speakers per batch, each with fixed number of utterances.

    Gender balanced: half male speakers, half female speakers per batch.
    No speaker is repeated until all speakers of that gender are used.

    batch_size = speakers_per_batch * samples_per_speaker
    e.g. speakers_per_batch=16, samples_per_speaker=4 -> 8 male * 4 + 8 female * 4 = 64
    """

    def __init__(self, manifest, speakers_per_batch=8, samples_per_speaker=4):
        self.speakers_per_batch = speakers_per_batch
        self.samples_per_speaker = samples_per_speaker
        self.batch_size = speakers_per_batch * samples_per_speaker
        self.half_spk = speakers_per_batch // 2

        # Group indices by speaker
        self.spk_to_indices = {}
        spk_to_gender = {}
        for i, e in enumerate(manifest):
            self.spk_to_indices.setdefault(e["speaker_id"], []).append(i)
            if e["speaker_id"] not in spk_to_gender:
                spk_to_gender[e["speaker_id"]] = e.get("gender", "").lower()

        # Split speakers by gender
        self.male_speakers = [s for s, g in spk_to_gender.items() if g == "male"]
        self.female_speakers = [s for s, g in spk_to_gender.items() if g == "female"]

        if not self.male_speakers or not self.female_speakers:
            raise ValueError(
                f"Gender balanced batching requires both male and female speakers. "
                f"Found {len(self.male_speakers)} male, {len(self.female_speakers)} female."
            )

        # Number of batches = limited by the gender with fewer speakers
        usable_per_gender = min(len(self.male_speakers), len(self.female_speakers))
        self.num_batches = max(1, usable_per_gender // self.half_spk)
        print(f"SpeakerBatchSampler: {len(self.male_speakers)} male, {len(self.female_speakers)} female speakers")
        print(f"  {self.half_spk} male + {self.half_spk} female per batch x {samples_per_speaker} samples = {self.batch_size}/batch")
        print(f"  {self.num_batches} batches per epoch")

    def _pick_speakers(self, speaker_list, count):
        """Pick `count` speakers without replacement. Reshuffles when exhausted."""
        picked = []
        pool = list(speaker_list)
        random.shuffle(pool)
        idx = 0
        while len(picked) < count:
            if idx >= len(pool):
                # All speakers used, reshuffle
                random.shuffle(pool)
                idx = 0
            picked.append(pool[idx])
            idx += 1
        return picked, pool[idx:]  # return remaining for next batch

    def __iter__(self):
        male_pool = list(self.male_speakers)
        female_pool = list(self.female_speakers)
        random.shuffle(male_pool)
        random.shuffle(female_pool)

        male_ptr = 0
        female_ptr = 0

        for _ in range(self.num_batches):
            # Pick male speakers
            if male_ptr + self.half_spk > len(male_pool):
                random.shuffle(male_pool)
                male_ptr = 0
            batch_males = male_pool[male_ptr:male_ptr + self.half_spk]
            male_ptr += self.half_spk

            # Pick female speakers
            if female_ptr + self.half_spk > len(female_pool):
                random.shuffle(female_pool)
                female_ptr = 0
            batch_females = female_pool[female_ptr:female_ptr + self.half_spk]
            female_ptr += self.half_spk

            # Gather sample indices
            indices = []
            for spk in batch_males + batch_females:
                spk_idxs = self.spk_to_indices[spk]
                if len(spk_idxs) >= self.samples_per_speaker:
                    chosen = random.sample(spk_idxs, self.samples_per_speaker)
                else:
                    chosen = random.choices(spk_idxs, k=self.samples_per_speaker)
                indices.extend(chosen)
            yield indices

    def __len__(self):
        return self.num_batches


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
