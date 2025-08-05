#!/usr/bin/env python
# Kronecker‑G‑LSTM on Sequential‑MNIST: 20‑trial learning‑curve logger
# ====================================================================
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import os, math, argparse, pathlib, numpy as np, torch
import torch.nn as nn, torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

# --------------------------------------------------------------------------------
#                         DATA  (unchanged)
# --------------------------------------------------------------------------------
def get_loaders(batch=128, root="./torch_datasets"):
    tfm = transforms.ToTensor()
    root = pathlib.Path(root).expanduser()
    ds_tr = MNIST(root, train=True,  download=True, transform=tfm)
    ds_te = MNIST(root, train=False, download=True, transform=tfm)
    n_tr  = int(0.8 * len(ds_tr))
    ds_tr, ds_va = random_split(ds_tr, [n_tr, len(ds_tr) - n_tr],
                                generator=torch.Generator().manual_seed(0))
    mk = lambda ds, shuf: DataLoader(ds, batch,
                                     shuffle=shuf, drop_last=False)
    return mk(ds_tr, True), mk(ds_va, False), mk(ds_te, False)

# --------------------------------------------------------------------------------
#                    KRONECKER‑SUM  G‑LSTM  (unchanged)
# --------------------------------------------------------------------------------
class KronEqLinear(nn.Module):
    def __init__(self, K, out_features, perm, taps=(0,), bias=False):
        super().__init__()
        self.taps = tuple(taps)
        self.register_buffer("perm", perm.clone())
        self.As   = nn.ModuleList([nn.Linear(K, out_features, bias=bias)
                                   for _ in self.taps])
    def _apply_perm(self, x, r):
        if r == 0:                              # identity
            return x
        idx = self.perm
        for _ in range(r - 1):                  # P^r
            idx = self.perm[idx]
        return x[:, idx]
    def forward(self, x):
        y = 0
        for r, A_r in zip(self.taps, self.As):
            y = y + A_r(self._apply_perm(x, r))
        return y

class GLSTMCell(nn.Module):
    def __init__(self, K, perm, taps=(0,), bias=False):
        super().__init__()
        self.K     = K
        self.W_x   = KronEqLinear(K, 4 * K, perm, taps, bias)
        self.W_h   = KronEqLinear(K, 4 * K, perm, taps, bias)
    def forward(self, x, h, c):
        gates = self.W_x(x) + self.W_h(h)
        i, f, g, o = torch.chunk(gates, 4, dim=-1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g       = torch.tanh(g)
        c_next  = f * c + i * g
        h_next  = o * torch.tanh(c_next)
        return h_next, c_next

class GLSTMSMNIST(nn.Module):
    def __init__(self, group_size=28, K=4, dropout=.1):
        super().__init__()
        self.G, self.K = group_size, K
        self.in_proj   = nn.Linear(1, K, bias=False)
        perm           = torch.roll(torch.arange(group_size), shifts=1)
        self.cell      = GLSTMCell(K, perm=perm, taps=(0,), bias=False)
        self.drop      = nn.Dropout(dropout)
        self.fc        = nn.Linear(group_size * K, 10, bias=False)
    def forward(self, x):
        B = x.size(0)
        seq = x.view(B, 28, self.G)                         # (B,T,G)
        h = c = torch.zeros(B, self.G, self.K,
                            device=x.device, dtype=x.dtype)
        for t in range(seq.size(1)):
            x_emb = self.in_proj(seq[:, t].unsqueeze(-1))   # (B,G,K)
            h, c  = self.cell(x_emb, h, c)
        return self.fc(self.drop(h.reshape(B, -1)))

# --------------------------------------------------------------------------------
#             critical initialisation (unchanged logic)
# --------------------------------------------------------------------------------
def critical_glstm_init(model: GLSTMSMNIST, g=1.0):
    std = g / math.sqrt(model.K)
    for mod in model.cell.modules():
        if isinstance(mod, nn.Linear):
            nn.init.normal_(mod.weight, 0.0, std)
    return g

# =================================================================================
#                               ==== NEW ====
#                         Training‑loop utilities
# =================================================================================
@torch.inference_mode()
def evaluate(net, loader, device):
    net.eval(); correct = total = 0
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

    acc_hist = []
    for ep in range(1, args.epochs + 1):
        net.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimzr.zero_grad()
            loss = loss_fn(net(xb), yb)
            loss.backward()
            clip_grad_norm_(net.parameters(), 1.0)      # good for RNNs
            optimzr.step()
        acc = evaluate(net, va_loader, device)
        acc_hist.append(acc)
        tqdm.write(f"[T{tid:02d}] epoch {ep:02d}/{args.epochs} "
                   f"val‑acc={acc:.4f}")

    # ---------- save CSV (epoch,accuracy) --------------------------------
    epochs = np.arange(1, args.epochs + 1)
    np.savetxt(f"acc_KronGLSTM_T{tid:02d}.csv",
               np.column_stack((epochs, acc_hist)),
               delimiter=',',
               header="epoch,accuracy",
               comments='')
    return np.array(acc_hist)

def plot_curves(curve_arr, args):
    epochs = np.arange(1, args.epochs + 1)
    mean   = curve_arr.mean(axis=0)

    plt.figure(figsize=(6,4))
    for c in curve_arr:
        plt.plot(epochs, c, color="steelblue", alpha=0.3)
    plt.plot(epochs, mean, color="crimson", linewidth=2.5,
             label=f"mean of {args.trials} runs")
    plt.xlabel("Epoch"); plt.ylabel("Validation accuracy")
    plt.ylim(0, 1); plt.grid(alpha=.3); plt.legend()
    plt.tight_layout()
    plt.savefig("kron_glstm_smnist_accuracy.png", dpi=300)
    plt.close()
    print("Saved  kron_glstm_smnist_accuracy.png")

# =================================================================================
#                                         main
# =================================================================================
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
    np.save("acc_KronGLSTM_all_trials.npy", all_curves)
    plot_curves(all_curves, args)

if __name__ == "__main__":
    main()
