"""Audio augmentation for speaker embedding training.

Supports: noise addition, reverberation, speed perturbation, SpecAugment.
All augmentation probabilities default to 0.0 (disabled).
Enable by setting probability > 0 in config.

Reference: https://github.com/modelscope/3D-Speaker/blob/main/speakerlab/process/augmentation.py
"""

import os
import random

import numpy as np
import torch
import torchaudio
from scipy import signal


def add_reverb(wav, rir_wav):
    """Convolve waveform with room impulse response.

    Args:
        wav: (T,) torch tensor
        rir_wav: (T,) torch tensor
    Returns:
        (T,) torch tensor
    """
    wav_np = wav.numpy()
    rir_np = rir_wav.numpy()
    wav_len = wav_np.shape[0]

    # Normalize RIR
    rir_np = rir_np / (np.sqrt(np.sum(rir_np ** 2)) + 1e-6)

    # Convolve and trim to original length
    out = signal.convolve(wav_np, rir_np, mode="full")[:wav_len]

    # Normalize to prevent clipping
    out = out / (np.max(np.abs(out)) + 1e-6)
    return torch.from_numpy(out).float()


def add_noise(wav, noise=None, snr_low=0, snr_high=15):
    """Add noise to waveform at random SNR.

    Args:
        wav: (T,) torch tensor
        noise: (T,) torch tensor or None (uses gaussian noise)
        snr_low: minimum SNR in dB
        snr_high: maximum SNR in dB
    Returns:
        (T,) torch tensor
    """
    if noise is None:
        noise = torch.randn_like(wav)

    wav_np = wav.numpy()
    noise_np = noise.numpy()
    wav_len = wav_np.shape[0]
    noise_len = noise_np.shape[0]

    # Match noise length to wav length
    if noise_len >= wav_len:
        start = random.randint(0, noise_len - wav_len)
        noise_np = noise_np[start:start + wav_len]
    else:
        repeats = wav_len // noise_len + 1
        noise_np = np.tile(noise_np, repeats)[:wav_len]

    # Scale noise to target SNR
    wav_db = 10 * np.log10(np.mean(wav_np ** 2) + 1e-6)
    noise_db = 10 * np.log10(np.mean(noise_np ** 2) + 1e-6)
    target_snr = random.uniform(snr_low, snr_high)
    noise_np = np.sqrt(10 ** ((wav_db - noise_db - target_snr) / 10)) * noise_np

    out = wav_np + noise_np
    out = out / (np.max(np.abs(out)) + 1e-6)
    return torch.from_numpy(out).float()


def speed_perturb(wav, sample_rate, speed_factor=None):
    """Apply speed perturbation.

    Args:
        wav: (T,) torch tensor
        sample_rate: audio sample rate
        speed_factor: specific factor or None for random choice from [0.9, 1.0, 1.1]
    Returns:
        (T,) torch tensor, possibly different length
    """
    if speed_factor is None:
        speed_factor = random.choice([0.9, 1.0, 1.1])

    if speed_factor == 1.0:
        return wav

    # Resample to simulate speed change
    effects = [["speed", str(speed_factor)], ["rate", str(sample_rate)]]
    wav_2d = wav.unsqueeze(0)  # (1, T)
    out, _ = torchaudio.sox_effects.apply_effects_tensor(wav_2d, sample_rate, effects)
    return out.squeeze(0)


def spec_augment(features, freq_mask_width=10, time_mask_width=20,
                 num_freq_masks=1, num_time_masks=1):
    """Apply SpecAugment (frequency + time masking) on mel features.

    Args:
        features: (n_mels, T) tensor
        freq_mask_width: max frequency channels to mask
        time_mask_width: max time frames to mask
        num_freq_masks: number of frequency masks
        num_time_masks: number of time masks
    Returns:
        (n_mels, T) tensor with masks applied
    """
    features = features.clone()
    n_mels, T = features.shape

    # Frequency masking
    for _ in range(num_freq_masks):
        f = random.randint(0, min(freq_mask_width, n_mels - 1))
        f0 = random.randint(0, n_mels - f)
        features[f0:f0 + f, :] = 0.0

    # Time masking
    for _ in range(num_time_masks):
        t = random.randint(0, min(time_mask_width, T - 1))
        t0 = random.randint(0, T - t)
        features[:, t0:t0 + t] = 0.0

    return features


def load_file_list(file_path):
    """Load a file list (one path per line) or scp file (key path per line).

    Returns list of file paths.
    """
    paths = []
    with open(file_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            # scp format: key path  OR  just path
            path = parts[-1]
            if os.path.isfile(path):
                paths.append(path)
    return paths


class WavAugmentor:
    """Waveform-level augmentation: noise, reverb, speed perturbation.

    All probabilities default to 0.0 (disabled).
    """

    def __init__(self, sample_rate=16000,
                 noise_prob=0.0, noise_file=None, noise_snr_low=0, noise_snr_high=15,
                 reverb_prob=0.0, reverb_file=None,
                 speed_prob=0.0, speed_factors=None):
        self.sample_rate = sample_rate
        self.noise_prob = noise_prob
        self.noise_snr_low = noise_snr_low
        self.noise_snr_high = noise_snr_high
        self.reverb_prob = reverb_prob
        self.speed_prob = speed_prob
        self.speed_factors = speed_factors or [0.9, 1.0, 1.1]

        # Load noise files
        self.noise_paths = []
        if noise_prob > 0.0:
            if noise_file is None or not os.path.isfile(noise_file):
                raise ValueError(f"noise_prob={noise_prob} but noise_file is missing: {noise_file}")
            self.noise_paths = load_file_list(noise_file)
            print(f"[AUG] Loaded {len(self.noise_paths)} noise files from {noise_file}")

        # Load RIR files
        self.rir_paths = []
        if reverb_prob > 0.0:
            if reverb_file is None or not os.path.isfile(reverb_file):
                raise ValueError(f"reverb_prob={reverb_prob} but reverb_file is missing: {reverb_file}")
            self.rir_paths = load_file_list(reverb_file)
            print(f"[AUG] Loaded {len(self.rir_paths)} RIR files from {reverb_file}")

        self._print_status()

    def _print_status(self):
        augs = []
        if self.noise_prob > 0:
            augs.append(f"noise(p={self.noise_prob}, snr={self.noise_snr_low}-{self.noise_snr_high}dB)")
        if self.reverb_prob > 0:
            augs.append(f"reverb(p={self.reverb_prob})")
        if self.speed_prob > 0:
            augs.append(f"speed(p={self.speed_prob}, factors={self.speed_factors})")
        if augs:
            print(f"[AUG] Waveform augmentations: {', '.join(augs)}")
        else:
            print("[AUG] Waveform augmentations: disabled (all probabilities=0)")

    def __call__(self, wav):
        """Apply augmentations to waveform.

        Args:
            wav: (T,) torch tensor
        Returns:
            (T,) torch tensor (augmented)
        """
        # Speed perturbation (changes length, so apply first)
        if self.speed_prob > random.random():
            factor = random.choice(self.speed_factors)
            wav = speed_perturb(wav, self.sample_rate, factor)

        # Reverb
        if self.reverb_prob > random.random() and self.rir_paths:
            rir_path = random.choice(self.rir_paths)
            rir_wav, rir_sr = torchaudio.load(rir_path)
            if rir_sr != self.sample_rate:
                rir_wav = torchaudio.functional.resample(rir_wav, rir_sr, self.sample_rate)
            wav = add_reverb(wav, rir_wav[0])

        # Noise
        if self.noise_prob > random.random() and self.noise_paths:
            noise_path = random.choice(self.noise_paths)
            noise_wav, noise_sr = torchaudio.load(noise_path)
            if noise_sr != self.sample_rate:
                noise_wav = torchaudio.functional.resample(noise_wav, noise_sr, self.sample_rate)
            if noise_wav.shape[0] > 1:
                noise_wav = noise_wav.mean(dim=0, keepdim=True)
            wav = add_noise(wav, noise_wav[0],
                            snr_low=self.noise_snr_low, snr_high=self.noise_snr_high)

        return wav


class SpecAugmentor:
    """Feature-level augmentation: SpecAugment (time + frequency masking).

    Probability defaults to 0.0 (disabled).
    """

    def __init__(self, prob=0.0, freq_mask_width=10, time_mask_width=20,
                 num_freq_masks=1, num_time_masks=1):
        self.prob = prob
        self.freq_mask_width = freq_mask_width
        self.time_mask_width = time_mask_width
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks

        if prob > 0:
            print(f"[AUG] SpecAugment: p={prob}, freq_mask={freq_mask_width}x{num_freq_masks}, "
                  f"time_mask={time_mask_width}x{num_time_masks}")
        else:
            print("[AUG] SpecAugment: disabled (probability=0)")

    def __call__(self, features):
        """Apply SpecAugment to features.

        Args:
            features: (n_mels, T) tensor
        Returns:
            (n_mels, T) tensor
        """
        if self.prob > random.random():
            features = spec_augment(
                features,
                freq_mask_width=self.freq_mask_width,
                time_mask_width=self.time_mask_width,
                num_freq_masks=self.num_freq_masks,
                num_time_masks=self.num_time_masks,
            )
        return features
