import json
import random
import torch
import torchaudio
from torch.utils.data import Dataset, Sampler


def load_manifest(path):
    with open(path) as f:
        return json.load(f)


def filter_manifest(manifest, min_duration=0.0, min_samples_per_speaker=0):
    before = len(manifest)
    before_spk = len(set(e["speaker_id"] for e in manifest))

    if min_duration > 0:
        manifest = [e for e in manifest if e.get("duration", 0) >= min_duration]

    if min_samples_per_speaker > 0:
        counts = {}
        for e in manifest:
            counts[e["speaker_id"]] = counts.get(e["speaker_id"], 0) + 1
        keep = {s for s, c in counts.items() if c >= min_samples_per_speaker}
        manifest = [e for e in manifest if e["speaker_id"] in keep]

    after = len(manifest)
    after_spk = len(set(e["speaker_id"] for e in manifest))
    print(f"  Filter: {before} -> {after} utterances, {before_spk} -> {after_spk} speakers")
    return manifest


class SpeakerDataset(Dataset):
    def __init__(self, entries, sample_rate, segment_duration, n_mels, n_fft,
                 hop_length, win_length, spk2label):
        self.entries = entries
        self.sample_rate = sample_rate
        self.segment_len = int(sample_rate * segment_duration)
        self.spk2label = spk2label
        self.num_speakers = len(spk2label)
        self.feature_fn = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft,
            hop_length=hop_length, win_length=win_length, n_mels=n_mels,
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

        features = torch.log(self.feature_fn(wav) + 1e-9)
        label = self.spk2label[entry["speaker_id"]]
        gender_idx = 0 if entry.get("gender", "").lower() == "male" else 1
        return features, label, gender_idx


class SpeakerBatchSampler(Sampler):
    """Random speaker batch sampler: N speakers x K samples per batch."""

    def __init__(self, manifest, speakers_per_batch, samples_per_speaker):
        self.speakers_per_batch = speakers_per_batch
        self.samples_per_speaker = samples_per_speaker

        self.spk_to_indices = {}
        for i, e in enumerate(manifest):
            self.spk_to_indices.setdefault(e["speaker_id"], []).append(i)
        self.all_speakers = list(self.spk_to_indices.keys())
        self.num_batches = len(self.all_speakers) // speakers_per_batch
        print(f"  SpeakerBatchSampler: {len(self.all_speakers)} speakers, "
              f"{self.num_batches} batches/epoch, "
              f"{speakers_per_batch}spk x {samples_per_speaker}samp = "
              f"{speakers_per_batch * samples_per_speaker}/batch")

    def __iter__(self):
        speakers = self.all_speakers[:]
        random.shuffle(speakers)
        for b in range(self.num_batches):
            batch = []
            batch_spks = speakers[b * self.speakers_per_batch:(b + 1) * self.speakers_per_batch]
            for spk in batch_spks:
                indices = self.spk_to_indices[spk]
                if len(indices) >= self.samples_per_speaker:
                    chosen = random.sample(indices, self.samples_per_speaker)
                else:
                    chosen = random.choices(indices, k=self.samples_per_speaker)
                batch.extend(chosen)
            yield batch

    def __len__(self):
        return self.num_batches
