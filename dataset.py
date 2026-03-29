import json
import random

import torch
import torchaudio
from torch.utils.data import Dataset, Sampler


class SpeakerDataset(Dataset):
    def __init__(self, entries, sample_rate=16000, segment_duration=3.0,
                 feature_type="melspectrogram", n_mels=80, n_fft=512,
                 hop_length=160, win_length=400, spk2label=None,
                 wav_augmentor=None, spec_augmentor=None):
        self.entries = entries
        self.sample_rate = sample_rate
        self.segment_len = int(sample_rate * segment_duration)
        self.feature_type = feature_type
        self.wav_augmentor = wav_augmentor
        self.spec_augmentor = spec_augmentor

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

    def __getitem__(self, idx, _retry=0):
        entry = self.entries[idx]
        try:
            wav, sr = torchaudio.load(entry["audio_file_path"])
        except Exception as e:
            if _retry >= 3:
                raise RuntimeError(f"Failed to load audio after 3 retries: {entry['audio_file_path']}") from e
            print(f"[WARN] Failed to load {entry['audio_file_path']}: {e}, picking another sample")
            return self.__getitem__(random.randint(0, len(self) - 1), _retry + 1)
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)

        # mono
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        wav = wav[0]

        # Waveform augmentation (before cropping so speed perturb can change length)
        if self.wav_augmentor is not None:
            wav = self.wav_augmentor(wav)

        # Crop or loop-pad to fixed length
        if wav.size(0) > self.segment_len:
            start = random.randint(0, wav.size(0) - self.segment_len)
            wav = wav[start:start + self.segment_len]
        elif wav.size(0) < self.segment_len:
            # Loop/repeat instead of zero-pad to avoid silence artifacts
            repeats = self.segment_len // wav.size(0) + 1
            wav = wav.repeat(repeats)[:self.segment_len]

        features = self.feature_fn(wav)  # (n_mels, T)

        # Log mel + epsilon to avoid -inf/NaN
        features = torch.log(features + 1e-9)

        # Feature-level augmentation (SpecAugment)
        if self.spec_augmentor is not None:
            features = self.spec_augmentor(features)
        spk = entry["speaker_id"]
        if spk not in self.spk2label:
            raise KeyError(f"Speaker '{spk}' not in spk2label. Available: {len(self.spk2label)} speakers. "
                           f"File: {entry['audio_file_path']}")
        label = self.spk2label[spk]
        gender_idx = 0 if entry.get("gender", "").lower() == "male" else 1
        return features, label, gender_idx



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


def filter_manifest(manifest, min_duration=0.0, min_samples_per_speaker=0):
    """Filter manifest: remove short utterances and speakers with too few samples.

    Args:
        manifest: list of entries
        min_duration: minimum utterance duration in seconds (0 = no filter)
        min_samples_per_speaker: minimum utterances per speaker (0 = no filter)

    Returns:
        filtered manifest (list of entries)
    """
    before_utts = len(manifest)
    before_spks = set(e["speaker_id"] for e in manifest)

    # Gender counts before
    spk_gender_before = {}
    for e in manifest:
        sid = e["speaker_id"]
        if sid not in spk_gender_before:
            spk_gender_before[sid] = e.get("gender", "").lower()
    male_before = sum(1 for g in spk_gender_before.values() if g == "male")
    female_before = sum(1 for g in spk_gender_before.values() if g == "female")

    print(f"\n{'─' * 60}")
    print(f"  MANIFEST FILTERING")
    print(f"{'─' * 60}")
    print(f"  Before: {len(before_spks)} speakers ({male_before}M + {female_before}F), "
          f"{before_utts} utterances")

    # Step 1: filter short utterances
    if min_duration > 0:
        removed_short = [e for e in manifest if e.get("duration", float("inf")) < min_duration]
        manifest = [e for e in manifest if e.get("duration", float("inf")) >= min_duration]
        print(f"\n  Duration filter (>= {min_duration}s):")
        print(f"    Removed {len(removed_short)} short utterances")
        print(f"    Remaining: {len(manifest)} utterances")

    # Step 2: filter speakers with too few samples
    if min_samples_per_speaker > 0:
        spk_counts = {}
        spk_gender = {}
        for e in manifest:
            sid = e["speaker_id"]
            spk_counts[sid] = spk_counts.get(sid, 0) + 1
            if sid not in spk_gender:
                spk_gender[sid] = e.get("gender", "").lower()

        removed_spks = {s for s, c in spk_counts.items() if c < min_samples_per_speaker}
        valid_spks = {s for s, c in spk_counts.items() if c >= min_samples_per_speaker}
        removed_male = sum(1 for s in removed_spks if spk_gender.get(s) == "male")
        removed_female = sum(1 for s in removed_spks if spk_gender.get(s) == "female")
        removed_utts = sum(spk_counts[s] for s in removed_spks)

        manifest = [e for e in manifest if e["speaker_id"] in valid_spks]

        print(f"\n  Speaker filter (>= {min_samples_per_speaker} samples):")
        print(f"    Dropped {len(removed_spks)} speakers ({removed_male}M + {removed_female}F)")
        print(f"    Dropped {removed_utts} utterances from those speakers")
        print(f"    Remaining: {len(valid_spks)} speakers, {len(manifest)} utterances")

    # Final summary
    after_spks = set(e["speaker_id"] for e in manifest)
    spk_gender_after = {}
    for e in manifest:
        sid = e["speaker_id"]
        if sid not in spk_gender_after:
            spk_gender_after[sid] = e.get("gender", "").lower()
    male_after = sum(1 for g in spk_gender_after.values() if g == "male")
    female_after = sum(1 for g in spk_gender_after.values() if g == "female")

    dropped_spks = before_spks - after_spks
    dropped_utts = before_utts - len(manifest)

    print(f"\n  After:  {len(after_spks)} speakers ({male_after}M + {female_after}F), "
          f"{len(manifest)} utterances")
    print(f"  Total dropped: {len(dropped_spks)} speakers, {dropped_utts} utterances")
    print(f"{'─' * 60}\n")

    return manifest


def split_manifest(manifest, val_split=0.1):
    """Split manifest by speaker, gender-balanced (equal male/female in val)."""
    spk_to_entries = {}
    spk_to_gender = {}
    for e in manifest:
        spk_to_entries.setdefault(e["speaker_id"], []).append(e)
        if e["speaker_id"] not in spk_to_gender:
            spk_to_gender[e["speaker_id"]] = e.get("gender", "").lower()

    male_spks = [s for s, g in spk_to_gender.items() if g == "male"]
    female_spks = [s for s, g in spk_to_gender.items() if g == "female"]

    random.shuffle(male_spks)
    random.shuffle(female_spks)

    total_spks = len(male_spks) + len(female_spks)
    val_count = max(2, int(total_spks * val_split))
    val_per_gender = val_count // 2

    val_males = male_spks[:val_per_gender]
    val_females = female_spks[:val_per_gender]
    val_spks = set(val_males + val_females)

    train, val = [], []
    for spk, entries in spk_to_entries.items():
        if spk in val_spks:
            val.extend(entries)
        else:
            train.extend(entries)

    print(f"Split: {len(val_males)} male + {len(val_females)} female speakers in val, "
          f"{len(male_spks) - len(val_males)} male + {len(female_spks) - len(val_females)} female in train")
    return train, val
