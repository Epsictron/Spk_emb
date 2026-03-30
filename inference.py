import argparse
import glob
import json
import os

import torch
import torch.nn.functional as F
import torchaudio
from torch.amp import autocast

from model import SpeakerEncoder


def load_model(checkpoint_path, config_path="config.json", device=None):
    """Load trained model from checkpoint.

    Returns (model, cfg, device).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(config_path) as f:
        cfg = json.load(f)

    # Compute context in frames from ms
    frame_ms = cfg["hop_length"] / cfg["sample_rate"] * 1000
    left_context_frames = int(cfg.get("left_context_ms", 100) / frame_ms)
    right_context_frames = int(cfg.get("right_context_ms", 0) / frame_ms)

    model = SpeakerEncoder(
        n_mels=cfg["n_mels"],
        embedding_dim=cfg["embedding_dim"],
        left_context_frames=left_context_frames,
        right_context_frames=right_context_frames,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["encoder"])
    model.eval()

    step = ckpt.get("step", "?")
    separation = ckpt.get("separation", "?")
    print(f"Loaded checkpoint: step={step}, separation={separation}")
    print(f"Device: {device}")

    return model, cfg, device


def load_audio(audio_path, cfg, target_duration=10.0):
    """Load audio file and return log-mel features with fixed duration.

    Short audio is loop-repeated. Long audio is cropped.
    Returns tensor of shape (n_mels, T).
    """
    wav, sr = torchaudio.load(audio_path)

    # Resample if needed
    if sr != cfg["sample_rate"]:
        wav = torchaudio.functional.resample(wav, sr, cfg["sample_rate"])

    # Mono
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav.squeeze(0)

    # Normalize to target_duration
    target_len = int(cfg["sample_rate"] * target_duration)
    if wav.size(0) < target_len:
        repeats = target_len // wav.size(0) + 1
        wav = wav.repeat(repeats)[:target_len]
    elif wav.size(0) > target_len:
        wav = wav[:target_len]

    # Feature extraction (must match training)
    feature_type = cfg.get("feature_type", "melspectrogram")
    if feature_type == "mfcc":
        transform = torchaudio.transforms.MFCC(
            sample_rate=cfg["sample_rate"], n_mfcc=40,
            melkwargs={"n_fft": cfg["n_fft"], "hop_length": cfg["hop_length"],
                       "win_length": cfg["win_length"], "n_mels": cfg["n_mels"]},
        )
    else:
        transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=cfg["sample_rate"],
            n_fft=cfg["n_fft"],
            hop_length=cfg["hop_length"],
            win_length=cfg["win_length"],
            n_mels=cfg["n_mels"],
        )
    features = transform(wav)
    features = torch.log(features + 1e-9)

    return features


def extract_embeddings_batch(model, audio_paths, cfg, device,
                              target_duration=10.0, batch_size=32):
    """Extract L2-normalized embeddings for a list of audio files.

    All audio is padded/cropped to target_duration for uniform batching.
    Returns dict: {audio_path: embedding_tensor (embedding_dim,)}.
    """
    use_amp = cfg.get("amp", False) and device.type == "cuda"
    results = {}
    failed = []

    # Load all features
    all_features = []
    valid_paths = []
    for path in audio_paths:
        try:
            feat = load_audio(path, cfg, target_duration)
            all_features.append(feat)
            valid_paths.append(path)
        except Exception as e:
            print(f"[WARN] Failed to load {path}: {e}")
            failed.append(path)

    if not all_features:
        print("No audio files loaded successfully.")
        return results

    # Batch inference
    model.eval()
    with torch.no_grad():
        for i in range(0, len(all_features), batch_size):
            batch_feats = torch.stack(all_features[i:i + batch_size]).to(device)
            batch_paths = valid_paths[i:i + batch_size]

            with autocast(device_type=device.type, enabled=use_amp):
                embeddings = model(batch_feats)

            embeddings = F.normalize(embeddings.float(), dim=1, eps=1e-8)

            for path, emb in zip(batch_paths, embeddings):
                results[path] = emb.cpu()

    if failed:
        print(f"Failed to load {len(failed)} files: {failed}")

    return results


def compute_similarity(emb1, emb2):
    """Cosine similarity between two embeddings."""
    emb1 = F.normalize(emb1.unsqueeze(0), dim=1, eps=1e-8)
    emb2 = F.normalize(emb2.unsqueeze(0), dim=1, eps=1e-8)
    return torch.mm(emb1, emb2.t()).item()


def main():
    parser = argparse.ArgumentParser(description="Speaker embedding inference")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--config", default="config.json", help="Path to config.json")
    parser.add_argument("--mode", choices=["extract", "verify"], required=True)

    # Extract mode
    parser.add_argument("--audio_dir", help="Directory of audio files for batch extraction")
    parser.add_argument("--audio_list", help="Text file with one audio path per line")
    parser.add_argument("--output", default="embeddings.pt", help="Output path for embeddings (.pt)")

    # Verify mode
    parser.add_argument("--audio1", help="First audio file for verification")
    parser.add_argument("--audio2", help="Second audio file for verification")
    parser.add_argument("--threshold", type=float, default=0.5, help="Similarity threshold for verify decision")

    # Common
    parser.add_argument("--duration", type=float, default=10.0, help="Target audio duration in seconds")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for extraction")
    parser.add_argument("--device", default=None, help="Device (cuda/cpu), auto-detected if omitted")

    args = parser.parse_args()

    device = torch.device(args.device) if args.device else None
    model, cfg, device = load_model(args.checkpoint, args.config, device)

    if args.mode == "extract":
        # Collect audio paths
        audio_paths = []
        if args.audio_dir:
            for ext in ("*.wav", "*.flac", "*.mp3", "*.ogg"):
                audio_paths.extend(glob.glob(os.path.join(args.audio_dir, "**", ext), recursive=True))
            audio_paths.sort()
        elif args.audio_list:
            with open(args.audio_list) as f:
                audio_paths = [line.strip() for line in f if line.strip()]
        else:
            parser.error("--audio_dir or --audio_list required for extract mode")

        print(f"Found {len(audio_paths)} audio files")
        print(f"Target duration: {args.duration}s, batch size: {args.batch_size}")

        embeddings = extract_embeddings_batch(
            model, audio_paths, cfg, device,
            target_duration=args.duration, batch_size=args.batch_size,
        )

        torch.save(embeddings, args.output)
        print(f"Saved {len(embeddings)} embeddings to {args.output}")

    elif args.mode == "verify":
        if not args.audio1 or not args.audio2:
            parser.error("--audio1 and --audio2 required for verify mode")

        embeddings = extract_embeddings_batch(
            model, [args.audio1, args.audio2], cfg, device,
            target_duration=args.duration, batch_size=2,
        )

        emb1 = embeddings[args.audio1]
        emb2 = embeddings[args.audio2]
        score = compute_similarity(emb1, emb2)
        decision = "SAME" if score >= args.threshold else "DIFFERENT"

        print(f"\nSimilarity: {score:.4f}")
        print(f"Threshold:  {args.threshold}")
        print(f"Decision:   {decision} speaker")


if __name__ == "__main__":
    main()
