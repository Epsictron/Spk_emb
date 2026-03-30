"""Autoresearch: Learn y = sin(x) with a small MLP.

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
def make_data(n_points, seed):
    torch.manual_seed(seed)
    x = torch.linspace(-2 * math.pi, 2 * math.pi, n_points).unsqueeze(1)
    y = torch.sin(x)
    return x, y


# --- Fixed: Model ---
class SineNet(nn.Module):
    def __init__(self, hidden_size, num_layers, activation, dropout):
        super().__init__()
        act_fn = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU, "silu": nn.SiLU}
        Act = act_fn.get(activation, nn.ReLU)

        layers = [nn.Linear(1, hidden_size), Act()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(Act())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_size, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# --- Fixed: Training and evaluation ---
def main():
    cfg = config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    x_train, y_train = make_data(cfg["train_points"], seed=42)
    x_val, y_val = make_data(cfg["val_points"], seed=123)
    x_train, y_train = x_train.to(device), y_train.to(device)
    x_val, y_val = x_val.to(device), y_val.to(device)

    # Model
    model = SineNet(
        cfg["hidden_size"], cfg["num_layers"],
        cfg["activation"], cfg["dropout"],
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
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

    criterion = nn.MSELoss()
    batch_size = cfg["batch_size"]
    n_train = x_train.size(0)

    # Train for fixed time budget
    start = time.time()
    epoch = 0
    best_val_mse = float("inf")

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

        # Validate every 50 epochs
        if epoch % 50 == 0:
            model.eval()
            with torch.no_grad():
                val_pred = model(x_val)
                val_mse = criterion(val_pred, y_val).item()
            if val_mse < best_val_mse:
                best_val_mse = val_mse
            train_loss = epoch_loss / max(n_batches, 1)
            print(f"epoch: {epoch} | train_mse: {train_loss:.6f} | val_mse: {val_mse:.6f}")

    # Final evaluation
    model.eval()
    with torch.no_grad():
        val_pred = model(x_val)
        final_val_mse = criterion(val_pred, y_val).item()

    total_time = time.time() - start
    print(f"\n--- Results ---")
    print(f"epochs: {epoch}")
    print(f"time: {total_time:.1f}s")
    print(f"val_mse: {final_val_mse:.8f}")


if __name__ == "__main__":
    main()
