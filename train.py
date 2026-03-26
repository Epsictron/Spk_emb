import argparse
import json
import math
import os
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.amp import GradScaler, autocast

from dataset import SpeakerDataset, SpeakerBatchSampler, load_manifest, split_manifest
from model import SpeakerEncoder
from losses import AAMSoftmaxLoss, PrototypicalLoss, ContrastiveLoss, CombinedLoss


def validate_config(cfg):
    """Validate config values before training starts."""
    errors = []

    # Required fields
    required = ["manifest_path", "output_dir", "sample_rate", "n_mels", "n_fft",
                "hop_length", "win_length", "segment_duration", "embedding_dim",
                "max_steps", "lr", "speakers_per_batch", "samples_per_speaker"]
    for key in required:
        if key not in cfg:
            errors.append(f"Missing required field: '{key}'")

    # Type and range checks
    if cfg.get("sample_rate", 16000) not in (8000, 16000, 22050, 44100, 48000):
        errors.append(f"Unusual sample_rate: {cfg['sample_rate']}. Expected 8000/16000/22050/44100/48000")

    if cfg.get("n_mels", 80) <= 0:
        errors.append("n_mels must be > 0")

    if cfg.get("embedding_dim", 256) <= 0:
        errors.append("embedding_dim must be > 0")

    if cfg.get("max_steps", 0) <= 0:
        errors.append("max_steps must be > 0")

    lr = cfg.get("lr", 0.001)
    if lr <= 0 or lr > 1:
        errors.append(f"lr={lr} looks wrong. Expected 0 < lr <= 1")

    if cfg.get("val_split", 0.1) <= 0 or cfg.get("val_split", 0.1) >= 1:
        errors.append("val_split must be between 0 and 1 (exclusive)")

    # Loss type
    valid_losses = ("aam", "prototypical", "contrastive", "combined")
    loss_type = cfg.get("loss_type", "aam")
    if loss_type not in valid_losses:
        errors.append(f"loss_type='{loss_type}' not in {valid_losses}")

    if loss_type == "combined":
        aw = cfg.get("aam_weight", 0.7)
        pw = cfg.get("proto_weight", 0.3)
        if aw < 0 or pw < 0:
            errors.append("aam_weight and proto_weight must be >= 0")
        if abs(aw + pw) < 1e-9:
            errors.append("aam_weight + proto_weight must be > 0")

    # Feature type
    valid_features = ("melspectrogram", "mfcc")
    if cfg.get("feature_type", "melspectrogram") not in valid_features:
        errors.append(f"feature_type not in {valid_features}")

    # Speaker batch sampler
    spb = cfg.get("speakers_per_batch", 0)
    sps = cfg.get("samples_per_speaker", 0)
    if spb <= 0 or sps <= 0:
        errors.append("speakers_per_batch and samples_per_speaker must be > 0")
    if spb % 2 != 0:
        errors.append("speakers_per_batch must be even (half male, half female)")
    if loss_type in ("prototypical", "combined") and sps < 2:
        errors.append(f"samples_per_speaker should be >= 2 for {loss_type} loss (need multiple samples per speaker)")

    # Warmup
    warmup = cfg.get("warmup_steps", 0)
    if warmup < 0:
        errors.append("warmup_steps must be >= 0")
    if warmup >= cfg.get("max_steps", 1):
        errors.append("warmup_steps must be < max_steps")

    # Val interval
    val_interval = cfg.get("val_interval", 2000)
    if val_interval <= 0:
        errors.append("val_interval must be > 0")

    # Grad clip
    gc = cfg.get("grad_clip", 0)
    if gc < 0:
        errors.append("grad_clip must be >= 0 (0 = disabled)")

    # Manifest file exists
    manifest_path = cfg.get("manifest_path", "")
    if manifest_path and not os.path.isfile(manifest_path):
        errors.append(f"manifest_path '{manifest_path}' does not exist")

    if errors:
        print("Config validation errors:")
        for e in errors:
            print(f"  [ERROR] {e}")
        raise ValueError(f"Config has {len(errors)} error(s). Fix them and retry.")

    print("[OK] Config validated")


def infinite_loader(data_loader):
    """Yields batches forever, restarting the loader when exhausted."""
    while True:
        for batch in data_loader:
            yield batch


def run_validation(encoder, criterion, val_loader, device, use_amp, loss_type):
    """Run validation and return loss, accuracy, time."""
    val_start = time.time()
    encoder.eval()
    criterion.eval()
    val_loss_sum, val_total, val_correct = 0.0, 0, 0

    with torch.no_grad():
        for feats, labels in val_loader:
            feats, labels = feats.to(device), labels.to(device)
            with autocast(device_type="cuda", enabled=use_amp):
                embeddings = encoder(feats)
                loss = criterion(embeddings, labels)
            val_loss_sum += loss.item() * labels.size(0)
            val_total += labels.size(0)

            if loss_type in ("aam", "combined"):
                w = criterion.weight if loss_type == "aam" else criterion.aam.weight
                emb_norm = F.normalize(embeddings.float(), dim=1)
                w_norm = F.normalize(w, dim=1)
                preds = F.linear(emb_norm, w_norm).argmax(dim=1)
                val_correct += (preds == labels).sum().item()

    encoder.train()
    criterion.train()

    val_loss = val_loss_sum / max(val_total, 1)
    val_acc = val_correct / max(val_total, 1) if loss_type in ("aam", "combined") else None
    val_time = time.time() - val_start
    return val_loss, val_acc, val_time


def train(config_path, checkpoint=None):
    with open(config_path) as f:
        cfg = json.load(f)

    validate_config(cfg)

    os.makedirs(cfg["output_dir"], exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = cfg.get("amp", False) and device.type == "cuda"
    if use_amp:
        print("Using mixed precision (AMP)")

    # Data
    manifest = load_manifest(cfg["manifest_path"])
    train_manifest, val_manifest = split_manifest(manifest, cfg.get("val_split", 0.1))
    print(f"Train: {len(train_manifest)}, Val: {len(val_manifest)}")

    # Build unified speaker-to-label mapping from full manifest
    all_speakers = sorted(set(e["speaker_id"] for e in manifest))
    spk2label = {s: i for i, s in enumerate(all_speakers)}
    print(f"Total speakers: {len(spk2label)}")

    train_ds = SpeakerDataset(
        train_manifest, cfg["sample_rate"], cfg["segment_duration"],
        cfg["feature_type"], cfg["n_mels"], cfg["n_fft"],
        cfg["hop_length"], cfg["win_length"], spk2label=spk2label
    )
    val_ds = SpeakerDataset(
        val_manifest, cfg["sample_rate"], cfg["segment_duration"],
        cfg["feature_type"], cfg["n_mels"], cfg["n_fft"],
        cfg["hop_length"], cfg["win_length"], spk2label=spk2label
    )

    # Sampler: gender-balanced speaker batch sampler
    batch_sampler = SpeakerBatchSampler(
        train_manifest, cfg["speakers_per_batch"], cfg["samples_per_speaker"]
    )
    train_loader = DataLoader(
        train_ds, batch_sampler=batch_sampler,
        num_workers=cfg["num_workers"], pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["speakers_per_batch"] * cfg["samples_per_speaker"],
        shuffle=False, num_workers=cfg["num_workers"], pin_memory=True
    )

    # Model
    encoder = SpeakerEncoder(cfg["n_mels"], cfg["embedding_dim"]).to(device)

    # Loss
    loss_type = cfg.get("loss_type", "aam")
    if loss_type == "aam":
        criterion = AAMSoftmaxLoss(
            cfg["embedding_dim"], train_ds.num_speakers,
            cfg.get("aam_margin", 0.2), cfg.get("aam_scale", 30)
        ).to(device)
    elif loss_type == "prototypical":
        criterion = PrototypicalLoss().to(device)
    elif loss_type == "contrastive":
        criterion = ContrastiveLoss().to(device)
    elif loss_type == "combined":
        criterion = CombinedLoss(
            cfg["embedding_dim"], train_ds.num_speakers,
            aam_weight=cfg.get("aam_weight", 0.7),
            proto_weight=cfg.get("proto_weight", 0.3),
            margin=cfg.get("aam_margin", 0.2),
            scale=cfg.get("aam_scale", 30)
        ).to(device)
        print(f"Combined loss: aam_weight={cfg.get('aam_weight', 0.7)}, proto_weight={cfg.get('proto_weight', 0.3)}")
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    max_steps = cfg["max_steps"]
    warmup_steps = cfg.get("warmup_steps", 0)
    val_interval = cfg.get("val_interval", 2000)
    log_interval = cfg.get("log_interval", 50)
    save_interval = cfg.get("save_interval", val_interval)
    grad_clip = cfg.get("grad_clip", 0)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(criterion.parameters()), lr=cfg["lr"]
    )

    # LR schedule: linear warmup + cosine decay (step-based)
    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / (warmup_steps + 1)
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler("cuda", enabled=use_amp)

    start_step = 1

    # Resume from checkpoint
    if checkpoint and os.path.isfile(checkpoint):
        print(f"Loading checkpoint: {checkpoint}")
        ckpt = torch.load(checkpoint, map_location=device)
        encoder.load_state_dict(ckpt["encoder"])
        criterion.load_state_dict(ckpt["criterion"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt and use_amp:
            scaler.load_state_dict(ckpt["scaler"])
        start_step = ckpt.get("step", 0) + 1
        print(f"Resuming from step {start_step}")
    else:
        print("Starting from scratch")

    writer = SummaryWriter(log_dir=os.path.join(cfg["output_dir"], "tb_logs"))
    best_val_loss = float("inf")

    # Print training summary
    batch_size = cfg["speakers_per_batch"] * cfg["samples_per_speaker"]
    batches_per_cycle = len(batch_sampler)
    total_params = sum(p.numel() for p in encoder.parameters()) + sum(p.numel() for p in criterion.parameters())

    print("\n" + "=" * 60)
    print("  TRAINING CONFIGURATION")
    print("=" * 60)
    print(f"  Device:              {device}")
    print(f"  AMP:                 {use_amp}")
    print(f"  Max steps:           {max_steps} (start: {start_step})")
    print(f"  Loss:                {loss_type}")
    print(f"  LR:                  {cfg['lr']} (warmup: {warmup_steps} steps)")
    print(f"  Grad clip:           {grad_clip if grad_clip > 0 else 'disabled'}")
    print(f"  Batch size:          {batch_size} ({cfg['speakers_per_batch']} spk x {cfg['samples_per_speaker']} samp)")
    print(f"  Batches per cycle:   {batches_per_cycle} (before reshuffling speakers)")
    print(f"  Val every:           {val_interval} steps")
    print(f"  Save every:          {save_interval} steps")
    print(f"  Log every:           {log_interval} steps")
    print(f"  Train samples:       {len(train_manifest)}")
    print(f"  Val samples:         {len(val_manifest)}")
    print(f"  Total speakers:      {len(spk2label)}")
    print(f"  Model params:        {total_params:,}")
    print(f"  Feature:             {cfg['feature_type']}")
    print("=" * 60 + "\n")

    # Training loop (step-based)
    encoder.train()
    criterion.train()
    train_iter = infinite_loader(train_loader)

    running_loss = 0.0
    running_correct = 0
    running_total = 0
    training_start = time.time()
    step_start = time.time()

    for step in range(start_step, max_steps + 1):
        feats, labels = next(train_iter)
        feats, labels = feats.to(device), labels.to(device)

        optimizer.zero_grad()
        with autocast(device_type="cuda", enabled=use_amp):
            embeddings = encoder(feats)
            loss = criterion(embeddings, labels)

        scaler.scale(loss).backward()

        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(criterion.parameters()), grad_clip
            )

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        running_loss += loss.item() * labels.size(0)
        running_total += labels.size(0)

        if loss_type in ("aam", "combined"):
            with torch.no_grad():
                w = criterion.weight if loss_type == "aam" else criterion.aam.weight
                emb_norm = F.normalize(embeddings.float(), dim=1)
                w_norm = F.normalize(w, dim=1)
                preds = F.linear(emb_norm, w_norm).argmax(dim=1)
                running_correct += (preds == labels).sum().item()

        # Log every N steps
        if step % log_interval == 0:
            avg_loss = running_loss / running_total
            elapsed = time.time() - step_start
            ms_per_step = (elapsed / log_interval) * 1000
            current_lr = optimizer.param_groups[0]["lr"]

            msg = f"  Step {step}/{max_steps} | loss: {avg_loss:.4f}"
            if loss_type in ("aam", "combined") and running_total > 0:
                msg += f" | acc: {running_correct / running_total:.4f}"
            msg += f" | lr: {current_lr:.6f} | {ms_per_step:.0f}ms/step"

            print(msg)
            writer.add_scalar("train/loss", avg_loss, step)
            writer.add_scalar("train/lr", current_lr, step)
            writer.add_scalar("train/ms_per_step", ms_per_step, step)
            if loss_type in ("aam", "combined") and running_total > 0:
                writer.add_scalar("train/accuracy", running_correct / running_total, step)

            running_loss = 0.0
            running_correct = 0
            running_total = 0
            step_start = time.time()

        # Validation every N steps
        if step % val_interval == 0:
            val_loss, val_acc, val_time = run_validation(
                encoder, criterion, val_loader, device, use_amp, loss_type
            )
            elapsed = time.time() - training_start
            remaining = (elapsed / (step - start_step + 1)) * (max_steps - step)

            print(f"\n{'─' * 60}")
            print(f"  VALIDATION @ Step {step}/{max_steps}")
            print(f"{'─' * 60}")
            print(f"  Val loss:     {val_loss:.4f}")
            if val_acc is not None:
                print(f"  Val accuracy: {val_acc:.4f}")
            print(f"  Val time:     {val_time:.1f}s")
            print(f"  Elapsed:      {elapsed/60:.1f}min | ETA: {remaining/60:.1f}min")

            writer.add_scalar("val/loss", val_loss, step)
            if val_acc is not None:
                writer.add_scalar("val/accuracy", val_acc, step)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                ckpt_dict = {
                    "step": step,
                    "encoder": encoder.state_dict(),
                    "criterion": criterion.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "val_loss": val_loss,
                }
                if use_amp:
                    ckpt_dict["scaler"] = scaler.state_dict()
                torch.save(ckpt_dict, os.path.join(cfg["output_dir"], "best_model.pt"))
                print(f"  ** New best model (val_loss={val_loss:.4f}) **")

            print(f"{'─' * 60}\n")

        # Save checkpoint every N steps
        if step % save_interval == 0:
            ckpt_dict = {
                "step": step,
                "encoder": encoder.state_dict(),
                "criterion": criterion.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_loss": best_val_loss,
            }
            if use_amp:
                ckpt_dict["scaler"] = scaler.state_dict()
            torch.save(ckpt_dict, os.path.join(cfg["output_dir"], "latest.pt"))

    total_time = time.time() - training_start
    print("=" * 60)
    print(f"  Training complete!")
    print(f"  Total steps: {max_steps}")
    print(f"  Total time:  {total_time/60:.1f}min")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print("=" * 60)
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--checkpoint", default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()
    train(args.config, args.checkpoint)
