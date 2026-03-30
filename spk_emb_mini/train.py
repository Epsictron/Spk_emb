import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import SpeakerDataset, SpeakerBatchSampler, load_manifest, filter_manifest
from model import SpeakerEncoder
from loss import AAMSoftmaxLoss
from evaluate import evaluate


def infinite_loader(loader):
    while True:
        yield from loader


def find_lr(encoder, criterion, train_loader, optimizer, device,
            start_lr=1e-7, end_lr=1.0, num_steps=100, writer=None):
    """LR range test: sweep LR from start to end, find steepest loss drop."""
    print(f"\n  LR Finder: {start_lr:.1e} -> {end_lr:.1e} over {num_steps} steps")
    encoder.train()
    criterion.train()

    # Save state
    enc_state = {k: v.clone() for k, v in encoder.state_dict().items()}
    crit_state = {k: v.clone() for k, v in criterion.state_dict().items()}
    opt_state = optimizer.state_dict()

    mult = (end_lr / start_lr) ** (1.0 / num_steps)
    lr = start_lr
    best_loss = float("inf")
    lrs, losses = [], []
    data_iter = infinite_loader(train_loader)

    for i in range(num_steps):
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        feats, labels, _ = next(data_iter)
        feats, labels = feats.to(device), labels.to(device)

        optimizer.zero_grad()
        emb = encoder(feats)
        loss = criterion(emb, labels)

        if torch.isnan(loss) or torch.isinf(loss):
            break
        if loss.item() > best_loss * 4:
            break

        loss.backward()
        optimizer.step()

        lrs.append(lr)
        losses.append(loss.item())
        if loss.item() < best_loss:
            best_loss = loss.item()

        if writer:
            writer.add_scalar("lr_finder/loss", loss.item(), i)
            writer.add_scalar("lr_finder/lr", lr, i)

        lr *= mult

    # Restore state
    encoder.load_state_dict(enc_state)
    criterion.load_state_dict(crit_state)
    optimizer.load_state_dict(opt_state)

    if len(losses) < 10:
        print("  LR Finder: not enough data points")
        return None

    # Find LR with steepest loss decrease (smoothed)
    window = max(len(losses) // 10, 3)
    smoothed = np.convolve(losses, np.ones(window) / window, mode="valid")
    gradients = np.gradient(smoothed)
    idx = int(np.argmin(gradients))
    suggested = lrs[idx + window // 2] if idx + window // 2 < len(lrs) else lrs[idx]

    print(f"  LR Finder: suggested LR = {suggested:.6f}")
    return suggested


def train(config_path, checkpoint=None):
    with open(config_path) as f:
        cfg = json.load(f)

    seed = cfg.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    os.makedirs(cfg["output_dir"], exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    manifest = load_manifest(cfg["manifest_path"])
    print(f"Raw manifest: {len(manifest)} entries")
    manifest = filter_manifest(manifest, cfg.get("min_duration", 0),
                               cfg.get("min_samples_per_speaker", 0))

    all_speakers = sorted(set(e["speaker_id"] for e in manifest))
    spk2label = {s: i for i, s in enumerate(all_speakers)}
    print(f"Speakers: {len(spk2label)}")

    ds = SpeakerDataset(manifest, cfg["sample_rate"], cfg["segment_duration"],
                        cfg["n_mels"], cfg["n_fft"], cfg["hop_length"],
                        cfg["win_length"], spk2label,
                        speed_perturb=cfg.get("speed_perturb", False),
                        spec_augment=cfg.get("spec_augment", False))
    sampler = SpeakerBatchSampler(manifest, cfg["speakers_per_batch"],
                                  cfg["samples_per_speaker"])
    nw = cfg.get("num_workers", 4)
    loader = DataLoader(ds, batch_sampler=sampler, num_workers=nw,
                        pin_memory=True, persistent_workers=nw > 0)

    # Test manifest for evaluation
    test_manifest_path = cfg.get("test_manifest_path", "")
    if test_manifest_path and os.path.isfile(test_manifest_path):
        test_manifest = load_manifest(test_manifest_path)
        test_manifest = filter_manifest(test_manifest, cfg.get("min_duration", 0),
                                        cfg.get("min_samples_per_speaker", 0))
        print(f"Test manifest: {len(test_manifest)} entries")
    else:
        test_manifest = []
        print("No test manifest, evaluation disabled")

    # Model + Loss
    encoder = SpeakerEncoder(cfg["n_mels"], cfg["embedding_dim"]).to(device)
    criterion = AAMSoftmaxLoss(cfg["embedding_dim"], ds.num_speakers,
                               cfg.get("aam_margin", 0.2),
                               cfg.get("aam_scale", 30),
                               cfg.get("label_smoothing", 0.0)).to(device)

    params = list(encoder.parameters()) + list(criterion.parameters())
    optimizer = torch.optim.Adam(params, lr=cfg["lr"])

    max_steps = cfg["max_steps"]
    warmup_steps = cfg.get("warmup_steps", 0)
    restart_period = cfg.get("restart_period", max_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / (warmup_steps + 1)
        t = step - warmup_steps
        period = restart_period
        progress = (t % period) / max(1, period)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    grad_clip = cfg.get("grad_clip", 0)
    accum_steps = cfg.get("accumulation_steps", 1)
    val_interval = cfg.get("val_interval", 2000)
    log_interval = cfg.get("log_interval", 50)
    start_step = 1

    # Resume
    if checkpoint and os.path.isfile(checkpoint):
        print(f"Loading checkpoint: {checkpoint}")
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        encoder.load_state_dict(ckpt["encoder"], strict=False)
        criterion.load_state_dict(ckpt["criterion"], strict=False)
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except (ValueError, KeyError):
            pass
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_step = ckpt.get("step", 0) + 1
        print(f"Resuming from step {start_step}")

    writer = SummaryWriter(os.path.join(cfg["output_dir"], "tb_logs"))
    best_separation = float("-inf")

    total_params = sum(p.numel() for p in encoder.parameters()) + sum(p.numel() for p in criterion.parameters())
    batch_size = cfg["speakers_per_batch"] * cfg["samples_per_speaker"]
    print(f"\n{'=' * 50}")
    print(f"  Device: {device} | Params: {total_params:,}")
    print(f"  Batch: {batch_size} ({cfg['speakers_per_batch']}spk x {cfg['samples_per_speaker']}samp)")
    print(f"  Accum: {accum_steps} | Effective batch: {batch_size * accum_steps}")
    print(f"  Steps: {max_steps} | LR: {cfg['lr']} | Warmup: {warmup_steps}")
    print(f"  Restart period: {restart_period} | Label smoothing: {cfg.get('label_smoothing', 0.0)}")
    print(f"  Speed perturb: {cfg.get('speed_perturb', False)} | SpecAugment: {cfg.get('spec_augment', False)}")
    print(f"  Eval every: {val_interval} | Log every: {log_interval}")
    print(f"{'=' * 50}\n")

    # LR finder (optional, fresh start only)
    if cfg.get("lr_finder", False) and start_step == 1:
        suggested_lr = find_lr(encoder, criterion, loader, optimizer, device,
                               cfg.get("lr_finder_start", 1e-7),
                               cfg.get("lr_finder_end", 1.0),
                               cfg.get("lr_finder_steps", 100), writer)
        if suggested_lr:
            print(f"  Suggested LR: {suggested_lr:.6f} (not auto-applied)")

    # Training loop
    encoder.train()
    criterion.train()
    data_iter = infinite_loader(loader)
    running_loss, running_total = 0.0, 0
    t_start = time.time()
    t_log = time.time()

    for step in range(start_step, max_steps + 1):
        # Gradient accumulation loop
        optimizer.zero_grad()
        step_loss = 0.0
        step_total = 0
        for _ in range(accum_steps):
            feats, labels, _ = next(data_iter)
            feats, labels = feats.to(device), labels.to(device)
            emb = encoder(feats)
            loss = criterion(emb, labels) / accum_steps
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            loss.backward()
            step_loss += loss.item() * accum_steps * labels.size(0)
            step_total += labels.size(0)

        if step_total == 0:
            print(f"  [WARN] Step {step}: all sub-batches NaN, skipping")
            scheduler.step()
            continue

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
        optimizer.step()
        scheduler.step()

        running_loss += step_loss
        running_total += step_total

        if step % log_interval == 0:
            avg = running_loss / running_total
            ms = (time.time() - t_log) / log_interval * 1000
            lr = optimizer.param_groups[0]["lr"]
            print(f"  Step {step}/{max_steps} | loss: {avg:.4f} | lr: {lr:.6f} | {ms:.0f}ms/step")
            writer.add_scalar("train/loss", avg, step)
            writer.add_scalar("train/lr", lr, step)
            running_loss, running_total = 0.0, 0
            t_log = time.time()

        if step % val_interval == 0 and test_manifest:
            print(f"\n{'─' * 50}")
            print(f"  EVAL @ Step {step}/{max_steps}")
            print(f"{'─' * 50}")

            result = evaluate(encoder, test_manifest, cfg, device, writer, step)
            separation = result["separation"]

            encoder.train()
            criterion.train()

            ckpt_dict = {
                "step": step,
                "encoder": encoder.state_dict(),
                "criterion": criterion.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "separation": separation,
            }
            torch.save(ckpt_dict, os.path.join(cfg["output_dir"], f"checkpoint_{step}.pt"))
            torch.save(ckpt_dict, os.path.join(cfg["output_dir"], "latest.pt"))

            if separation > best_separation:
                best_separation = separation
                torch.save(ckpt_dict, os.path.join(cfg["output_dir"], "best_model.pt"))
                print(f"  ** New best (separation={separation:.4f}) **")

            print(f"{'─' * 50}\n")

    elapsed = time.time() - t_start
    print(f"\n{'=' * 50}")
    print(f"  Done! {max_steps} steps in {elapsed/60:.1f}min")
    print(f"  Best separation: {best_separation:.4f}")
    print(f"{'=' * 50}")
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()
    train(args.config, args.checkpoint)
