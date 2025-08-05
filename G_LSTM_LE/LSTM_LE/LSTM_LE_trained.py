#!/usr/bin/env python
# Train 1‑layer LSTM on Sequential‑MNIST; save 20 learning curves.
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import os, math, argparse, pathlib, numpy as np, torch
import torch.nn as nn, torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm                           # progress bars

# ---------- SMNIST loaders (unchanged) ---------------------------------------
def get_loaders(batch=128, root="./torch_datasets"):
    tfm = transforms.ToTensor()
    root = pathlib.Path(root).expanduser()
    ds_tr = MNIST(root, train=True,  download=True, transform=tfm)
    ds_te = MNIST(root, train=False, download=True, transform=tfm)
    n_tr  = int(0.8*len(ds_tr))
    ds_tr, ds_va = random_split(ds_tr, [n_tr, len(ds_tr)-n_tr],
                                generator=torch.Generator().manual_seed(0))
    mk = lambda ds, shuf: DataLoader(ds, batch, shuffle=shuf, drop_last=False)
    return mk(ds_tr, True), mk(ds_va, False), mk(ds_te, False)

# ---------- 1‑layer LSTM (unchanged) ----------------------------------------
class LSTMSMNIST(nn.Module):
    def __init__(self, hidden=64, dropout=0.1):
        super().__init__()
        self.hidden = hidden
        self.lstm   = nn.LSTM(28, hidden, batch_first=True, bias=False)
        self.drop   = nn.Dropout(dropout)
        self.fc     = nn.Linear(hidden, 10, bias=False)
    def forward(self, x):
        B = x.size(0)
        seq = x.view(B, 28, 28)              # (B, time=28, feat=28)
        h0 = c0 = torch.zeros(1, B, self.hidden, device=x.device)
        y, _ = self.lstm(seq, (h0, c0))
        return self.fc(self.drop(y)[:, -1])  # last time step

# ---------- weight initialisation (unchanged) -------------------------------
def critical_lstm_init(model: nn.Module, g: float = 1.0):
    H = model.hidden
    gain = g / math.sqrt(H)
    nn.init.normal_(model.lstm.weight_hh_l0, 0.0, gain)
    nn.init.normal_(model.lstm.weight_ih_l0, 0.0, gain)
    return g

# ============================================================================  
#                              ==== NEW ====  
# ============================================================================  
def evaluate(net, loader, device):
    """Return accuracy on `loader`."""
    net.eval(); correct = 0; total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = net(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total   += y.numel()
    return correct / total

def train_one_trial(trial_id: int, args, tr_loader, va_loader):
    """Train for `args.epochs` epochs and return accuracy history."""
    device = torch.device(args.device)
    net     = LSTMSMNIST(args.hidden).to(device)
    critical_lstm_init(net, g=args.gain)
    optimzr = optim.Adam(net.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    acc_curve = []
    for epoch in range(1, args.epochs + 1):
        net.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimzr.zero_grad()
            loss = loss_fn(net(xb), yb)
            loss.backward()
            clip_grad_norm_(net.parameters(), 1.0)   # good for LSTMs
            optimzr.step()

        val_acc = evaluate(net, va_loader, device)
        acc_curve.append(val_acc)

        tqdm.write(f"[T{trial_id:02d}] epoch {epoch:02d}/{args.epochs} "
                   f"val‑acc={val_acc:.4f}")

    # ---- save epoch/accuracy as 2‑column CSV ------------------------------
    epochs = np.arange(1, args.epochs + 1)
    np.savetxt(f"acc_T{trial_id:02d}.csv",
               np.column_stack((epochs, acc_curve)),
               delimiter=',', header="epoch,accuracy", comments='')
    return np.array(acc_curve)

def plot_all_curves(all_curves, args):
    """Draw every run (transparent) + mean (solid)."""
    epochs = np.arange(1, args.epochs + 1)
    mean   = all_curves.mean(axis=0)

    plt.figure(figsize=(6,4))
    for curve in all_curves:
        plt.plot(epochs, curve, color="steelblue", alpha=0.30)  # translucent
    plt.plot(epochs, mean,  color="crimson", linewidth=2.5,
             label=f"mean of {args.trials} runs")
    plt.xlabel("Epoch"); plt.ylabel("Validation accuracy")
    plt.ylim(0,1); plt.grid(alpha=.3)
    plt.legend(); plt.tight_layout()
    plt.savefig("smnist_lstm_20trials_accuracy.png", dpi=300)
    plt.close()
    print("Saved  smnist_lstm_20trials_accuracy.png")

# ============================================================================  
#                                    main  
# ============================================================================  
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hidden",  type=int,   default=8)
    p.add_argument("--gain",    type=float, default=3.05)
    p.add_argument("--epochs",  type=int,   default=20)
    p.add_argument("--batch",   type=int,   default=128)
    p.add_argument("--lr",      type=float, default=1e-3)
    p.add_argument("--trials",  type=int,   default=20)
    p.add_argument("--device",  default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    tr_loader, va_loader, _ = get_loaders(batch=args.batch)

    all_curves = []
    for t in range(1, args.trials + 1):
        tqdm.write(f"\n=== Trial {t}/{args.trials} (g={args.gain}) ===")
        curve = train_one_trial(t, args, tr_loader, va_loader)
        all_curves.append(curve)

    all_curves = np.vstack(all_curves)
    np.save("acc_all_trials.npy", all_curves)
    plot_all_curves(all_curves, args)

if __name__ == "__main__":
    main()
