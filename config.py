import argparse


def get_config():
    parser = argparse.ArgumentParser(description="Speaker Embedding Training")

    # data
    parser.add_argument("--manifest", type=str, required=True, help="Path to manifest JSON file")
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation split ratio")

    # features
    parser.add_argument("--feature", type=str, default="melspec", choices=["melspec", "mfcc"], help="Feature type")
    parser.add_argument("--n_mels", type=int, default=80)
    parser.add_argument("--n_mfcc", type=int, default=40)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--max_duration", type=float, default=3.0, help="Max audio duration in seconds")

    # model
    parser.add_argument("--emb_dim", type=int, default=256, help="Embedding dimension")
    parser.add_argument("--num_speakers", type=int, default=None, help="Number of speakers (auto from manifest)")

    # loss
    parser.add_argument("--loss", type=str, default="aam", choices=["aam", "contrastive"])
    parser.add_argument("--aam_margin", type=float, default=0.2)
    parser.add_argument("--aam_scale", type=float, default=30.0)

    # training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--save_dir", type=str, default="exp", help="Directory to save checkpoints and logs")
    parser.add_argument("--balanced_sampling", action="store_true", help="Balance male/female in batches")

    return parser.parse_args()
