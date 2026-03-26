import argparse
import json
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast

from dataset import SpeakerDataset, GenderBalancedSampler, SpeakerBatchSampler, load_manifest, split_manifest
from model import SpeakerEncoder, AAMSoftmaxLoss, PrototypicalLoss, ContrastiveLoss, CombinedLoss


def validate_config(cfg):
    """Validate config values before training starts."""
    errors = []

    # Required fields
    required = ["manifest_path", "output_dir", "sample_rate", "n_mels", "n_fft",
                "hop_length", "win_length", "segment_duration", "embedding_dim",
                "epochs", "lr"]
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

    if cfg.get("epochs", 50) <= 0:
        errors.append("epochs must be > 0")

    lr = cfg.get("lr", 0.001)
    if lr <= 0 or lr > 1:
        errors.append(f"lr={lr} looks wrong. Expected 0 < lr <= 1")

    if cfg.get("val_split", 0.1) <= 0 or cfg.get("val_split", 0.1) >= 1:
        errors.append("val_split must be between 0 and 1 (exclusive)")

    if cfg.get("batch_size", 32) <= 0:
        errors.append("batch_size must be > 0")

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
    spb = cfg.get("speakers_per_batch")
    sps = cfg.get("samples_per_speaker")
    if spb is not None and sps is not None:
        if spb <= 0 or sps <= 0:
            errors.append("speakers_per_batch and samples_per_speaker must be > 0")
        if loss_type in ("prototypical", "combined") and sps < 2:
            errors.append(f"samples_per_speaker should be >= 2 for {loss_type} loss (need multiple samples per speaker)")

    # Warmup
    warmup = cfg.get("warmup_epochs", 0)
    if warmup < 0:
        errors.append("warmup_epochs must be >= 0")
    if warmup >= cfg.get("epochs", 50):
        errors.append("warmup_epochs must be < epochs")

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

    train_ds = SpeakerDataset(
        train_manifest, cfg["sample_rate"], cfg["segment_duration"],
        cfg["feature_type"], cfg["n_mels"], cfg["n_fft"],
        cfg["hop_length"], cfg["win_length"]
    )
    val_ds = SpeakerDataset(
        val_manifest, cfg["sample_rate"], cfg["segment_duration"],
        cfg["feature_type"], cfg["n_mels"], cfg["n_fft"],
        cfg["hop_length"], cfg["win_length"]
    )
    val_ds.spk2label = train_ds.spk2label
    val_ds.num_speakers = train_ds.num_speakers

    # Sampler: speaker-batch or gender-balanced or default shuffle
    spk_per_batch = cfg.get("speakers_per_batch")
    samp_per_spk = cfg.get("samples_per_speaker")
    use_spk_sampler = spk_per_batch and samp_per_spk

    if use_spk_sampler:
        batch_sampler = SpeakerBatchSampler(train_manifest, spk_per_batch, samp_per_spk)
        train_loader = DataLoader(
            train_ds, batch_sampler=batch_sampler,
            num_workers=cfg["num_workers"], pin_memory=True
        )
        print(f"Using SpeakerBatchSampler: {spk_per_batch} spk/batch x {samp_per_spk} samp/spk = {spk_per_batch * samp_per_spk} batch_size")
    else:
        sampler = GenderBalancedSampler(train_manifest) if cfg.get("gender_balanced") else None
        train_loader = DataLoader(
            train_ds, batch_size=cfg["batch_size"], sampler=sampler,
            shuffle=(sampler is None), num_workers=cfg["num_workers"], pin_memory=True
        )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=cfg["num_workers"], pin_memory=True
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

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(criterion.parameters()), lr=cfg["lr"]
    )

    # Schedulers: warmup + cosine
    warmup_epochs = cfg.get("warmup_epochs", 0)
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / (warmup_epochs + 1)
        progress = (epoch - warmup_epochs) / max(1, cfg["epochs"] - warmup_epochs)
        return 0.5 * (1 + __import__("math").cos(__import__("math").pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    grad_clip = cfg.get("grad_clip", 0)
    scaler = GradScaler(enabled=use_amp)

    start_epoch = 1

    # Resume from checkpoint
    if checkpoint and os.path.isfile(checkpoint):
        print(f"Loading checkpoint: {checkpoint}")
        ckpt = torch.load(checkpoint, map_location=device)
        encoder.load_state_dict(ckpt["encoder"])
        criterion.load_state_dict(ckpt["criterion"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt and use_amp:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"Resuming from epoch {start_epoch}")
    else:
        print("Starting from scratch")

    writer = SummaryWriter(log_dir=os.path.join(cfg["output_dir"], "tb_logs"))
    log_interval = cfg.get("log_interval", 50)
    best_val_loss = float("inf")

    if warmup_epochs > 0:
        print(f"LR warmup: {warmup_epochs} epochs")
    if grad_clip > 0:
        print(f"Gradient clipping: {grad_clip}")

    for epoch in range(start_epoch, cfg["epochs"] + 1):
        # Train
        encoder.train()
        criterion.train()
        total_loss, correct, total = 0.0, 0, 0

        for step, (feats, labels) in enumerate(train_loader, 1):
            feats, labels = feats.to(device), labels.to(device)

            optimizer.zero_grad()
            with autocast(enabled=use_amp):
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

            total_loss += loss.item() * labels.size(0)
            total += labels.size(0)

            # Accuracy for AAM / combined loss
            if loss_type in ("aam", "combined"):
                with torch.no_grad():
                    w = criterion.weight if loss_type == "aam" else criterion.aam.weight
                    emb_norm = F.normalize(embeddings.float(), dim=1)
                    w_norm = F.normalize(w, dim=1)
                    preds = F.linear(emb_norm, w_norm).argmax(dim=1)
                    correct += (preds == labels).sum().item()

            if step % log_interval == 0:
                print(f"Epoch {epoch} Step {step} Loss: {loss.item():.4f}")

        train_loss = total_loss / total
        current_lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("train/lr", current_lr, epoch)
        msg = f"Epoch {epoch}: train_loss={train_loss:.4f} lr={current_lr:.6f}"

        if loss_type in ("aam", "combined") and total > 0:
            train_acc = correct / total
            writer.add_scalar("train/accuracy", train_acc, epoch)
            msg += f" train_acc={train_acc:.4f}"

        # Validate
        encoder.eval()
        criterion.eval()
        val_loss_sum, val_total, val_correct = 0.0, 0, 0

        with torch.no_grad():
            for feats, labels in val_loader:
                feats, labels = feats.to(device), labels.to(device)
                with autocast(enabled=use_amp):
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

        val_loss = val_loss_sum / max(val_total, 1)
        writer.add_scalar("val/loss", val_loss, epoch)
        msg += f" val_loss={val_loss:.4f}"

        if loss_type in ("aam", "combined") and val_total > 0:
            val_acc = val_correct / val_total
            writer.add_scalar("val/accuracy", val_acc, epoch)
            msg += f" val_acc={val_acc:.4f}"

        print(msg)

        # Save checkpoint (always save latest)
        ckpt_dict = {
            "epoch": epoch,
            "encoder": encoder.state_dict(),
            "criterion": criterion.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_loss": val_loss,
        }
        if use_amp:
            ckpt_dict["scaler"] = scaler.state_dict()
        torch.save(ckpt_dict, os.path.join(cfg["output_dir"], "latest.pt"))

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(ckpt_dict, os.path.join(cfg["output_dir"], "best_model.pt"))
            print(f"  -> Saved best model (val_loss={val_loss:.4f})")

        scheduler.step()

    writer.close()
    print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--checkpoint", default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()
    train(args.config, args.checkpoint)
