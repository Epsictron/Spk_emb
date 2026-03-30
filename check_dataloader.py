"""Sanity check: load one batch from the dataloader and print shapes, paths, labels."""
import json
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import SpeakerDataset, SpeakerBatchSampler, load_manifest, filter_manifest


def main():
    with open("config.json") as f:
        cfg = json.load(f)

    seed = cfg.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load and filter manifest
    manifest = load_manifest(cfg["manifest_path"])
    print(f"Raw manifest: {len(manifest)} entries")

    manifest = filter_manifest(
        manifest,
        min_duration=cfg.get("min_duration", 0),
        min_samples_per_speaker=cfg.get("min_samples_per_speaker", 0),
    )
    print(f"After filter: {len(manifest)} entries")

    if len(manifest) == 0:
        print("No entries after filtering. Check min_duration / min_samples_per_speaker.")
        return

    # Speaker mapping
    all_speakers = sorted(set(e["speaker_id"] for e in manifest))
    spk2label = {s: i for i, s in enumerate(all_speakers)}
    label2spk = {i: s for s, i in spk2label.items()}
    print(f"Total speakers: {len(all_speakers)}")

    # Dataset
    ds = SpeakerDataset(
        manifest, cfg["sample_rate"], cfg["segment_duration"],
        cfg["feature_type"], cfg["n_mels"], cfg["n_fft"],
        cfg["hop_length"], cfg["win_length"], spk2label=spk2label,
    )

    # Batch sampler
    sampler = SpeakerBatchSampler(
        manifest, cfg["speakers_per_batch"], cfg["samples_per_speaker"]
    )

    loader = DataLoader(ds, batch_sampler=sampler, num_workers=0)

    print(f"\nExpected batch size: {cfg['speakers_per_batch']} spk x {cfg['samples_per_speaker']} samp "
          f"= {cfg['speakers_per_batch'] * cfg['samples_per_speaker']}")
    print(f"Segment duration: {cfg['segment_duration']}s")

    expected_frames = int(cfg["sample_rate"] * cfg["segment_duration"] / cfg["hop_length"]) + 1
    print(f"Expected feature shape: ({cfg['n_mels']}, ~{expected_frames})")

    # Fetch one batch
    print("\n--- Batch 1 ---")
    for features, labels, genders in loader:
        print(f"Features shape: {features.shape}")
        print(f"Labels shape:   {labels.shape}")
        print(f"Genders shape:  {genders.shape}")
        print(f"Features dtype: {features.dtype}")
        print(f"Labels dtype:   {labels.dtype}")

        # Stats
        print(f"\nFeature stats:")
        print(f"  min={features.min().item():.4f}  max={features.max().item():.4f}  "
              f"mean={features.mean().item():.4f}  std={features.std().item():.4f}")

        has_nan = torch.isnan(features).any().item()
        has_inf = torch.isinf(features).any().item()
        print(f"  NaN: {has_nan}  Inf: {has_inf}")

        # Unique speakers and gender breakdown
        unique_labels = labels.unique()
        n_male = (genders == 0).sum().item()
        n_female = (genders == 1).sum().item()
        print(f"\nUnique speakers in batch: {len(unique_labels)}")
        print(f"Gender: {n_male} male samples, {n_female} female samples")

        # Per-speaker breakdown (first 10)
        print(f"\nPer-speaker sample counts (showing up to 10):")
        for i, lbl in enumerate(unique_labels[:10]):
            spk_id = label2spk.get(lbl.item(), "?")
            count = (labels == lbl).sum().item()
            gender = "M" if genders[labels == lbl][0].item() == 0 else "F"
            print(f"  {spk_id} (label={lbl.item()}, {gender}): {count} samples")

        # Sample audio paths from manifest for the batch indices
        print(f"\nSample entries from manifest (first 5 in batch):")
        batch_indices = list(sampler)[0][:5]
        for idx in batch_indices:
            e = manifest[idx]
            print(f"  idx={idx}: {e['audio_file_path']} | spk={e['speaker_id']} | "
                  f"dur={e.get('duration', '?')}s | gender={e.get('gender', '?')}")

        break  # Only one batch

    print("\n--- Dataloader check complete ---")


if __name__ == "__main__":
    main()
