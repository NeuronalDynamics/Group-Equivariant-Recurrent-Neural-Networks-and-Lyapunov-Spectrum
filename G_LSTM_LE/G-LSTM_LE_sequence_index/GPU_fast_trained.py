#!/usr/bin/env python
# Train 1‑layer permutation‑equivariant G‑LSTM on Sequential‑MNIST.
# Saves 20 learning‑curve CSVs and one summary plot.
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import os, math, argparse, pathlib, numpy as np, torch
import torch.nn as nn, torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm                             # progress bars

# ---------------- SMNIST loaders (unchanged) ------------------------------
def get_loaders(batch=128, root="./torch_datasets"):
    tfm   = transforms.ToTensor()
    root  = pathlib.Path(root).expanduser()
    ds_tr = MNIST(root, train=True,  download=True, transform=tfm)
    ds_te = MNIST(root, train=False, download=True, transform=tfm)
    n_tr  = int(0.8 * len(ds_tr))
    ds_tr, ds_va = random_split(ds_tr,
                                [n_tr, len(ds_tr) - n_tr],
                                generator=torch.Generator().manual_seed(0))
    mk = lambda ds, shuf: DataLoader(ds, batch, shuffle=shuf, drop_last=False)
    return mk(ds_tr, True), mk(ds_va, False), mk(ds_te, False)

# ---------------- G‑LSTM network (identical to your original) -------------
class GLSTMCell(nn.Module):
    def __init__(self, K: int, bias: bool = False):
        super().__init__()
        self.K   = K
        self.W_x = nn.Linear(K, 4 * K, bias=bias)
        self.W_h = nn.Linear(K, 4 * K, bias=bias)
    def forward(self, x, h, c):
        gates     = self.W_x(x) + self.W_h(h)          # (B,G,4K)
        i, f, g, o = torch.chunk(gates, 4, dim=-1)
        i, f, o   = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g         = torch.tanh(g)
        c_next    = f * c + i * g
        h_next    = o * torch.tanh(c_next)
        return h_next, c_next

class GLSTMSMNIST(nn.Module):
    def __init__(self, group_size=28, K=4, dropout=.1):
        super().__init__()
        self.G, self.K = group_size, K
        self.in_proj   = nn.Linear(1, K, bias=False)
        self.cell      = GLSTMCell(K, bias=False)
        self.drop      = nn.Dropout(dropout)
        self.fc        = nn.Linear(group_size * K, 10, bias=False)
    def forward(self, x):
        B, = x.shape[:1]
        seq = x.view(B, 28, self.G)                    # (B,T=28,G=28)
        h = c = torch.zeros(B, self.G, self.K,
                            device=x.device, dtype=x.dtype)
        for t in range(seq.size(1)):
            x_emb = self.in_proj(seq[:, t].unsqueeze(-1))
            h, c  = self.cell(x_emb, h, c)
        return self.fc(self.drop(h.reshape(B, -1)))

# ---------------- Critical init (unchanged) -------------------------------
def critical_glstm_init(model: GLSTMSMNIST, g=1.0):
    std = g / math.sqrt(model.K)
    nn.init.normal_(model.cell.W_h.weight, 0.0, std)
    nn.init.normal_(model.cell.W_x.weight, 0.0, std)
    return g

# =======================================================================
#                              ==== NEW ====
# =======================================================================
def evaluate(net, loader, device):
    net.eval(); correct = total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred   = net(xb).argmax(1)
            correct += (pred == yb).sum().item()
            total   += yb.numel()
    return correct / total

def train_one_trial(tid, args, tr_loader, va_loader):
    device  = torch.device(args.device)
    net     = GLSTMSMNIST(args.group_size, args.K, dropout=.1).to(device)
    critical_glstm_init(net, g=args.gain)
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
            clip_grad_norm_(net.parameters(), 1.0)    # recurrent stability
            optimzr.step()
        acc = evaluate(net, va_loader, device)
        acc_curve.append(acc)
        tqdm.write(f"[T{tid:02d}] epoch {epoch:02d}/{args.epochs} "
                   f"val‑acc={acc:.4f}")

    # save epoch/accuracy CSV -------------------------------------------------
    epochs = np.arange(1, args.epochs + 1)
    np.savetxt(f"acc_GLSTM_T{tid:02d}.csv",
               np.column_stack((epochs, acc_curve)),
               delimiter=',', header="epoch,accuracy", comments='')
    return np.array(acc_curve)

def plot_curves(curves, args):
    epochs = np.arange(1, args.epochs + 1)
    mean   = curves.mean(axis=0)
    plt.figure(figsize=(6,4))
    for c in curves:
        plt.plot(epochs, c, color="steelblue", alpha=0.3)
    plt.plot(epochs, mean, color="crimson", linewidth=2.5,
             label=f"mean of {args.trials} runs")
    plt.xlabel("Epoch"); plt.ylabel("Validation accuracy")
    plt.ylim(0, 1); plt.grid(alpha=.3); plt.legend()
    plt.tight_layout()
    plt.savefig("glstm_smnist_20trials_accuracy.png", dpi=300)
    plt.close()
    print("Saved  glstm_smnist_20trials_accuracy.png")

# =======================================================================
#                                  main
# =======================================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--group_size", type=int, default=28)
    p.add_argument("--K",          type=int, default=8)
    p.add_argument("--gain",       type=float, default=8.45)
    p.add_argument("--epochs",     type=int, default=20)
    p.add_argument("--batch",      type=int, default=128)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--trials",     type=int, default=20)
    p.add_argument("--device",     default="cuda" if torch.cuda.is_available()
                                               else "cpu")
    args = p.parse_args()

    tr_loader, va_loader, _ = get_loaders(args.batch)

    all_curves = []
    for t in range(1, args.trials + 1):
        tqdm.write(f"\n=== Trial {t}/{args.trials} (g={args.gain}) ===")
        curve = train_one_trial(t, args, tr_loader, va_loader)
        all_curves.append(curve)

    all_curves = np.vstack(all_curves)
    np.save("acc_GLSTM_all_trials.npy", all_curves)
    plot_curves(all_curves, args)

if __name__ == "__main__":
    main()
