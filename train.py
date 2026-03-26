import argparse
import json
import os
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import SpeakerDataset, GenderBalancedSampler, load_manifest, split_manifest
from model import SpeakerEncoder, AAMSoftmaxLoss, PrototypicalLoss


def train(config_path):
    with open(config_path) as f:
        cfg = json.load(f)

    os.makedirs(cfg["output_dir"], exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    manifest = load_manifest(cfg["manifest_path"])
    train_manifest, val_manifest = split_manifest(manifest, cfg.get("val_split", 0.1))

    train_ds = SpeakerDataset(
        train_manifest, cfg["sample_rate"], cfg["segment_duration"],
        cfg["feature_type"], cfg["n_mels"], cfg["n_fft"],
        cfg["hop_length"], cfg["win_length"]
    )
    val_ds = SpeakerDataset(
        val_manifest, cfg["sample_rate"], cfg["segment_duration"],
        cfg["feature_type"], cfg["n_mels"], cfg["n_mels"],
        cfg["hop_length"], cfg["win_length"]
    )
    # Use label mapping from training set for val
    val_ds.spk2label = train_ds.spk2label
    val_ds.num_speakers = train_ds.num_speakers

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
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(criterion.parameters()), lr=cfg["lr"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg["epochs"])
    writer = SummaryWriter(log_dir=os.path.join(cfg["output_dir"], "tb_logs"))

    best_val_loss = float("inf")
    log_interval = cfg.get("log_interval", 50)

    for epoch in range(1, cfg["epochs"] + 1):
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

            if step % log_interval == 0:
                print(f"Epoch {epoch} Step {step} Loss: {loss.item():.4f}")

        train_loss = total_loss / total
        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("train/lr", scheduler.get_last_lr()[0], epoch)

        # Validate
        encoder.eval()
        criterion.eval()
        val_loss_sum, val_total = 0.0, 0

        with torch.no_grad():
            for feats, labels in val_loader:
                feats, labels = feats.to(device), labels.to(device)
                embeddings = encoder(feats)
                loss = criterion(embeddings, labels)
                val_loss_sum += loss.item() * labels.size(0)
                val_total += labels.size(0)

        val_loss = val_loss_sum / max(val_total, 1)
        writer.add_scalar("val/loss", val_loss, epoch)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "encoder": encoder.state_dict(),
                "criterion": criterion.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
            }, os.path.join(cfg["output_dir"], "best_model.pt"))
            print(f"  -> Saved best model (val_loss={val_loss:.4f})")

        scheduler.step()

    writer.close()
    print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()
    train(args.config)
