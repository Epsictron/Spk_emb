"""Autoresearch: Toy learning tasks (sine regression / spiral classification).

The AI agent may ONLY modify the `config` dict below.
Everything else (model architecture, data generation, evaluation) is fixed.
"""

import math
import time
import torch
import torch.nn as nn

# ============================================================
# CONFIG — The AI agent may ONLY modify values in this dict.
# ============================================================
config = {
    "task": "spiral",           # sine | spiral
    "hidden_size": 64,
    "num_layers": 3,
    "activation": "relu",       # relu | tanh | gelu | silu
    "dropout": 0.0,
    "lr": 0.001,
    "optimizer": "adam",        # adam | sgd | adamw
    "weight_decay": 0.0,
    "batch_size": 64,
    "train_points": 2000,
    "val_points": 500,
    "time_budget_sec": 60,
}
# ============================================================


# --- Fixed: Data generation ---
def make_sine_data(n_points, seed):
    torch.manual_seed(seed)
    x = torch.linspace(-2 * math.pi, 2 * math.pi, n_points).unsqueeze(1)
    y = torch.sin(x)
    return x, y


def make_spiral_data(n_points, seed, n_classes=2, noise=0.8):
    """Two interleaved spirals in 2D. Returns (x, y) with x:(N,2) y:(N,)."""
    torch.manual_seed(seed)
    points_per_class = n_points // n_classes
    x_list, y_list = [], []
    for c in range(n_classes):
        r = torch.linspace(0.2, 1.0, points_per_class)
        theta = torch.linspace(c * math.pi, c * math.pi + 3 * math.pi, points_per_class)
        x1 = r * torch.cos(theta) + torch.randn(points_per_class) * noise * 0.1
        x2 = r * torch.sin(theta) + torch.randn(points_per_class) * noise * 0.1
        x_list.append(torch.stack([x1, x2], dim=1))
        y_list.append(torch.full((points_per_class,), c, dtype=torch.long))
    x = torch.cat(x_list)
    y = torch.cat(y_list)
    perm = torch.randperm(x.size(0))
    return x[perm], y[perm]


# --- Fixed: Model ---
class Net(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_size, num_layers, activation, dropout):
        super().__init__()
        act_fn = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU, "silu": nn.SiLU}
        Act = act_fn.get(activation, nn.ReLU)

        layers = [nn.Linear(in_dim, hidden_size), Act()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(Act())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_size, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# --- Fixed: Training and evaluation ---
def main():
    cfg = config
    task = cfg["task"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    if task == "sine":
        x_train, y_train = make_sine_data(cfg["train_points"], seed=42)
        x_val, y_val = make_sine_data(cfg["val_points"], seed=123)
        in_dim, out_dim = 1, 1
        is_classification = False
    elif task == "spiral":
        x_train, y_train = make_spiral_data(cfg["train_points"], seed=42)
        x_val, y_val = make_spiral_data(cfg["val_points"], seed=123)
        in_dim, out_dim = 2, 2
        is_classification = True
    else:
        raise ValueError(f"Unknown task: {task}")

    x_train, y_train = x_train.to(device), y_train.to(device)
    x_val, y_val = x_val.to(device), y_val.to(device)

    # Model
    model = Net(
        in_dim, out_dim,
        cfg["hidden_size"], cfg["num_layers"],
        cfg["activation"], cfg["dropout"],
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"task: {task}")
    print(f"params: {param_count}")
    print(f"device: {device}")
    print(f"config: {cfg}")

    # Optimizer
    opt_name = cfg["optimizer"]
    if opt_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"],
                                     weight_decay=cfg["weight_decay"])
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                                      weight_decay=cfg["weight_decay"])
    elif opt_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg["lr"],
                                    weight_decay=cfg["weight_decay"])
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")

    if is_classification:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()

    batch_size = cfg["batch_size"]
    n_train = x_train.size(0)

    # Train for fixed time budget
    start = time.time()
    epoch = 0

    while True:
        elapsed = time.time() - start
        if elapsed >= cfg["time_budget_sec"]:
            break

        model.train()
        perm = torch.randperm(n_train, device=device)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = x_train[idx], y_train[idx]

            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        epoch += 1

        # Log every 50 epochs
        if epoch % 50 == 0:
            model.eval()
            with torch.no_grad():
                val_pred = model(x_val)
                val_loss = criterion(val_pred, y_val).item()
                if is_classification:
                    val_acc = (val_pred.argmax(dim=1) == y_val).float().mean().item()
            train_loss = epoch_loss / max(n_batches, 1)
            if is_classification:
                print(f"epoch: {epoch} | train_loss: {train_loss:.6f} | val_loss: {val_loss:.6f} | val_acc: {val_acc:.4f}")
            else:
                print(f"epoch: {epoch} | train_mse: {train_loss:.6f} | val_mse: {val_loss:.6f}")

    # Final evaluation
    model.eval()
    with torch.no_grad():
        val_pred = model(x_val)
        final_loss = criterion(val_pred, y_val).item()
        if is_classification:
            final_acc = (val_pred.argmax(dim=1) == y_val).float().mean().item()

    total_time = time.time() - start
    print(f"\n--- Results ---")
    print(f"epochs: {epoch}")
    print(f"time: {total_time:.1f}s")
    if is_classification:
        print(f"val_loss: {final_loss:.8f}")
        print(f"val_acc: {final_acc:.6f}")
    else:
        print(f"val_mse: {final_loss:.8f}")


if __name__ == "__main__":
    main()
