import os

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config import get_config
from dataset import SpeakerDataset, GenderBalancedSampler, load_manifest, split_data
from model import SpeakerEncoder, AAMSoftmax, ContrastiveLoss


def train():
    cfg = get_config()
    os.makedirs(cfg.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # data
    entries = load_manifest(cfg.manifest)
    train_entries, val_entries = split_data(entries, cfg.val_split)
    print(f"Train: {len(train_entries)}, Val: {len(val_entries)}")

    train_ds = SpeakerDataset(train_entries, cfg.sample_rate, cfg.max_duration, cfg.feature, cfg.n_mels, cfg.n_mfcc)
    val_ds = SpeakerDataset(val_entries, cfg.sample_rate, cfg.max_duration, cfg.feature, cfg.n_mels, cfg.n_mfcc)

    num_speakers = cfg.num_speakers or train_ds.num_speakers
    in_channels = cfg.n_mels if cfg.feature == "melspec" else cfg.n_mfcc

    sampler = GenderBalancedSampler(train_entries, cfg.batch_size) if cfg.balanced_sampling else None
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                              shuffle=(sampler is None), sampler=sampler, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=2)

    # model
    encoder = SpeakerEncoder(in_channels, cfg.emb_dim).to(device)

    if cfg.loss == "aam":
        criterion = AAMSoftmax(cfg.emb_dim, num_speakers, cfg.aam_margin, cfg.aam_scale).to(device)
    else:
        criterion = ContrastiveLoss().to(device)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(criterion.parameters()), lr=cfg.lr
    )

    start_epoch = 0

    # load checkpoint if provided
    if cfg.checkpoint and os.path.isfile(cfg.checkpoint):
        print(f"Loading checkpoint: {cfg.checkpoint}")
        ckpt = torch.load(cfg.checkpoint, map_location=device)
        encoder.load_state_dict(ckpt["encoder"])
        criterion.load_state_dict(ckpt["criterion"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"Resuming from epoch {start_epoch}")
    else:
        print("Starting from scratch")

    writer = SummaryWriter(log_dir=os.path.join(cfg.save_dir, "logs"))

    # train loop
    for epoch in range(start_epoch, cfg.epochs):
        encoder.train()
        criterion.train()
        total_loss = 0
        correct = 0
        total = 0

        for feats, labels in train_loader:
            feats, labels = feats.to(device), labels.to(device)
            emb = encoder(feats)
            loss = criterion(emb, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * feats.size(0)

            # accuracy (for AAM)
            if cfg.loss == "aam":
                with torch.no_grad():
                    emb_norm = torch.nn.functional.normalize(emb, dim=1)
                    w_norm = torch.nn.functional.normalize(criterion.weight, dim=1)
                    preds = torch.mm(emb_norm, w_norm.t()).argmax(dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

        avg_loss = total_loss / len(train_ds)
        writer.add_scalar("train/loss", avg_loss, epoch)
        msg = f"Epoch {epoch}/{cfg.epochs} - Train Loss: {avg_loss:.4f}"

        if total > 0:
            acc = correct / total
            writer.add_scalar("train/accuracy", acc, epoch)
            msg += f" - Acc: {acc:.4f}"

        # validation
        encoder.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for feats, labels in val_loader:
                feats, labels = feats.to(device), labels.to(device)
                emb = encoder(feats)
                loss = criterion(emb, labels)
                val_loss += loss.item() * feats.size(0)

                if cfg.loss == "aam":
                    emb_norm = torch.nn.functional.normalize(emb, dim=1)
                    w_norm = torch.nn.functional.normalize(criterion.weight, dim=1)
                    preds = torch.mm(emb_norm, w_norm.t()).argmax(dim=1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)

        avg_val_loss = val_loss / max(len(val_ds), 1)
        writer.add_scalar("val/loss", avg_val_loss, epoch)
        msg += f" - Val Loss: {avg_val_loss:.4f}"

        if val_total > 0:
            val_acc = val_correct / val_total
            writer.add_scalar("val/accuracy", val_acc, epoch)
            msg += f" - Val Acc: {val_acc:.4f}"

        print(msg)

        # save checkpoint
        torch.save({
            "epoch": epoch,
            "encoder": encoder.state_dict(),
            "criterion": criterion.state_dict(),
            "optimizer": optimizer.state_dict(),
        }, os.path.join(cfg.save_dir, "latest.pt"))

    writer.close()
    print("Done.")


if __name__ == "__main__":
    train()
