"""Quick verification that all components work."""
import json
import torch
from model import SpeakerEncoder, AAMSoftmaxLoss, PrototypicalLoss, ContrastiveLoss, CombinedLoss
from dataset import SpeakerDataset, SpeakerBatchSampler, split_manifest


def main():
    print("=== Verifying Speaker Embedding Project ===\n")

    # 1. Config
    with open("config.json") as f:
        cfg = json.load(f)
    print("[OK] Config loaded")

    # 2. Manifest format
    with open("data/manifest.json") as f:
        manifest = json.load(f)
    required = {"audio_file_path", "speaker_id", "duration", "dataset_name", "language"}
    for entry in manifest:
        assert required.issubset(entry.keys()), f"Missing keys: {required - entry.keys()}"
    print(f"[OK] Manifest has {len(manifest)} entries with all required fields")

    # 3. Model forward pass
    encoder = SpeakerEncoder(n_mels=cfg["n_mels"], embedding_dim=cfg["embedding_dim"])
    dummy = torch.randn(4, cfg["n_mels"], 300)  # (B, n_mels, T)
    emb = encoder(dummy)
    assert emb.shape == (4, cfg["embedding_dim"])
    print(f"[OK] Encoder output shape: {emb.shape}")

    # 4. AAM loss
    aam = AAMSoftmaxLoss(cfg["embedding_dim"], num_speakers=10)
    labels = torch.tensor([0, 1, 2, 3])
    loss = aam(emb, labels)
    assert loss.item() > 0
    print(f"[OK] AAMSoftmax loss: {loss.item():.4f}")

    # 5. Prototypical loss
    proto = PrototypicalLoss()
    labels2 = torch.tensor([0, 0, 1, 1])
    loss2 = proto(emb, labels2)
    assert loss2.item() > 0
    print(f"[OK] Prototypical loss: {loss2.item():.4f}")

    # 6. Contrastive loss
    contrastive = ContrastiveLoss()
    labels3 = torch.tensor([0, 0, 1, 1])
    loss3 = contrastive(emb, labels3)
    print(f"[OK] Contrastive loss: {loss3.item():.4f}")

    # 7. Combined loss
    combined = CombinedLoss(cfg["embedding_dim"], num_speakers=10, aam_weight=0.7, proto_weight=0.3)
    labels4 = torch.tensor([0, 0, 1, 1])
    loss4 = combined(emb, labels4)
    assert loss4.item() > 0
    print(f"[OK] Combined loss (0.7*aam + 0.3*proto): {loss4.item():.4f}")

    # 8. Speaker batch sampler (gender balanced)
    fake_spk_manifest = []
    for i in range(10):
        gender = "male" if i < 5 else "female"
        for j in range(8):
            fake_spk_manifest.append({
                "speaker_id": f"spk{i:03d}", "audio_file_path": f"f{i}_{j}.wav", "gender": gender
            })
    # 10 speakers (5M, 5F), 8 samples each, batch = 4 spk (2M+2F) x 4 samp = 16
    spk_sampler = SpeakerBatchSampler(fake_spk_manifest, speakers_per_batch=4, samples_per_speaker=4)
    batches = list(spk_sampler)
    assert all(len(b) == 16 for b in batches), "Batch size mismatch"
    # Verify no speaker repeated within a batch
    for batch in batches:
        spk_ids = [fake_spk_manifest[i]["speaker_id"] for i in batch]
        assert len(set(spk_ids)) == 4, f"Expected 4 unique speakers, got {len(set(spk_ids))}"
        genders = [fake_spk_manifest[i]["gender"] for i in batch]
        assert genders.count("male") == 8, f"Expected 8 male samples, got {genders.count('male')}"
        assert genders.count("female") == 8, f"Expected 8 female samples, got {genders.count('female')}"
    print(f"[OK] SpeakerBatchSampler: {len(batches)} batches, 2M+2F spk x 4 samp = 16/batch, gender balanced")

    # 9. Split
    fake_manifest = [
        {"speaker_id": f"spk{i:03d}", "gender": "male" if i % 2 == 0 else "female"}
        for i in range(20)
    ]
    train_m, val_m = split_manifest(fake_manifest, val_split=0.2)
    train_spks = set(e["speaker_id"] for e in train_m)
    val_spks = set(e["speaker_id"] for e in val_m)
    assert train_spks.isdisjoint(val_spks), "Speaker leak between train/val!"
    print(f"[OK] Split: {len(train_m)} train, {len(val_m)} val (no speaker overlap)")

    print("\n=== All checks passed ===")


if __name__ == "__main__":
    main()
