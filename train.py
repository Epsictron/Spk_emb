import argparse
import json
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import SpeakerDataset, GenderBalancedSampler, SpeakerBatchSampler, load_manifest, split_manifest
from model import SpeakerEncoder, AAMSoftmaxLoss, PrototypicalLoss, ContrastiveLoss, CombinedLoss


def train(config_path, checkpoint=None):
    with open(config_path) as f:
        cfg = json.load(f)

    os.makedirs(cfg["output_dir"], exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg["epochs"])

    start_epoch = 1

    # Resume from checkpoint
    if checkpoint and os.path.isfile(checkpoint):
        print(f"Loading checkpoint: {checkpoint}")
        ckpt = torch.load(checkpoint, map_location=device)
        encoder.load_state_dict(ckpt["encoder"])
        criterion.load_state_dict(ckpt["criterion"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"Resuming from epoch {start_epoch}")
    else:
        print("Starting from scratch")

    writer = SummaryWriter(log_dir=os.path.join(cfg["output_dir"], "tb_logs"))
    log_interval = cfg.get("log_interval", 50)
    best_val_loss = float("inf")

    for epoch in range(start_epoch, cfg["epochs"] + 1):
        # Train
        encoder.train()
        criterion.train()
        total_loss, correct, total = 0.0, 0, 0

        for step, (feats, labels) in enumerate(train_loader, 1):
            feats, labels = feats.to(device), labels.to(device)
            embeddings = encoder(feats)
            loss = criterion(embeddings, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            total += labels.size(0)

            # Accuracy for AAM / combined loss
            if loss_type in ("aam", "combined"):
                with torch.no_grad():
                    w = criterion.weight if loss_type == "aam" else criterion.aam.weight
                    emb_norm = F.normalize(embeddings, dim=1)
                    w_norm = F.normalize(w, dim=1)
                    preds = F.linear(emb_norm, w_norm).argmax(dim=1)
                    correct += (preds == labels).sum().item()

            if step % log_interval == 0:
                print(f"Epoch {epoch} Step {step} Loss: {loss.item():.4f}")

        train_loss = total_loss / total
        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("train/lr", scheduler.get_last_lr()[0], epoch)
        msg = f"Epoch {epoch}: train_loss={train_loss:.4f}"

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
                embeddings = encoder(feats)
                loss = criterion(embeddings, labels)
                val_loss_sum += loss.item() * labels.size(0)
                val_total += labels.size(0)

                if loss_type in ("aam", "combined"):
                    w = criterion.weight if loss_type == "aam" else criterion.aam.weight
                    emb_norm = F.normalize(embeddings, dim=1)
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
        torch.save({
            "epoch": epoch,
            "encoder": encoder.state_dict(),
            "criterion": criterion.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_loss": val_loss,
        }, os.path.join(cfg["output_dir"], "latest.pt"))

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "encoder": encoder.state_dict(),
                "criterion": criterion.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_loss": val_loss,
            }, os.path.join(cfg["output_dir"], "best_model.pt"))
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
