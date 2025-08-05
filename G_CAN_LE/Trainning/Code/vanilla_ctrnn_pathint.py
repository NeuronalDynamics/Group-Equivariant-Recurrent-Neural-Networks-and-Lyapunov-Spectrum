#!/usr/bin/env python
# ======================================================================
#  vanilla_ctrnn_pathint.py  – Continuous‑time RNN on 1‑D path‑integration
# ======================================================================
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import os, argparse, math, random, numpy as np, matplotlib.pyplot as plt
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

# ──────────────────────────────────────────────────────────────────────
#  Path‑integration synthetic dataset
# ──────────────────────────────────────────────────────────────────────
class PathInt1D(Dataset):
    """Velocity  →  absolute position (mod 1) on a ring."""
    def __init__(self, n_seq=10_000, T=100, dt=0.02, v_max=0.1, seed=0):
        super().__init__();  rng = np.random.default_rng(seed)
        v = rng.uniform(-v_max, v_max, size=(n_seq, T, 1))       # (N,T,1)
        s = np.mod(np.cumsum(v*dt, axis=1), 1.0)                 # (N,T,1)
        self.x = torch.tensor(v, dtype=torch.float32)            # vel
        self.y = torch.tensor(s, dtype=torch.float32)            # pos
    def __len__(self):  return len(self.x)
    def __getitem__(self,i): return self.x[i], self.y[i]

# ──────────────────────────────────────────────────────────────────────
#  Vanilla continuous‑time RNN
# ──────────────────────────────────────────────────────────────────────
class CTRNN(torch.nn.Module):
    def __init__(self, hidden=100, dt=0.02, g=1.5):
        super().__init__(); self.N, self.dt = hidden, dt
        self.J   = torch.nn.Parameter(g * torch.randn(hidden, hidden) /
                                      math.sqrt(hidden))
        self.B   = torch.nn.Parameter(torch.randn(hidden, 1) /
                                      math.sqrt(hidden))
        self.readout = torch.nn.Linear(hidden, 1, bias=False)
    def forward(self, vel):                                    # (B,T,1)
        B, T, _ = vel.shape
        h = torch.zeros(B, self.N, device=vel.device)
        outs = []
        for t in range(T):
            h = h*(1-self.dt) + self.dt*(torch.tanh(h) @ self.J.T +
                                          vel[:, t] @ self.B.T)
            outs.append(self.readout(torch.tanh(h)))
        return torch.stack(outs, 1)                            # (B,T,1)

# ──────────────────────────────────────────────────────────────────────
#  Training helpers
# ──────────────────────────────────────────────────────────────────────
def evaluate(net, loader, device):
    net.eval(); errs=[]
    with torch.no_grad():
        for v,s in loader:
            v,s = v.to(device), s.to(device)
            y = net(v)
            errs.append(((y-s)**2).mean().sqrt())
    return torch.stack(errs).mean().item()

def run_experiment(args):
    dev = torch.device(args.device)
    tr_ds = PathInt1D(n_seq=8000, T=args.T, dt=args.dt)
    va_ds = PathInt1D(n_seq=2000, T=args.T, dt=args.dt, seed=1)
    tr_ld = DataLoader(tr_ds, args.batch, shuffle=True)
    va_ld = DataLoader(va_ds, args.batch, shuffle=False)

    curves = []
    for trial in range(1, args.trials+1):
        net = CTRNN(hidden=args.hidden, dt=args.dt, g=args.gain).to(dev)

        # ── FREEZE recurrent matrix  (+ optionally B)  ──────────────
        net.J.requires_grad_(False)                       # always frozen
        if args.freeze_B:
            net.B.requires_grad_(False)

        opt = torch.optim.Adam(net.parameters(), lr=args.lr)
        hist=[]
        for epoch in range(1, args.epochs+1):
            net.train()
            for v,s in tr_ld:
                v,s = v.to(dev), s.to(dev)
                opt.zero_grad()
                loss = ((net(v)-s)**2).mean().sqrt()
                loss.backward(); clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
            rmse = evaluate(net, va_ld, dev)
            hist.append(rmse)
            tqdm.write(f"[CTRNN T{trial:02d}] epoch {epoch:02d}/{args.epochs} "
                       f"val‑RMSE={rmse:.4f}")
        curves.append(np.array(hist))
        np.savetxt(f"rmse_CTRNN_T{trial:02d}.csv",
                   np.column_stack([np.arange(1,args.epochs+1), hist]),
                   header="epoch,rmse", delimiter=',', comments='')

    curves = np.vstack(curves);  np.save("rmse_CTRNN_all.npy", curves)
    plot_curves(curves, "CTRNN")

def plot_curves(mat, tag):
    ep = np.arange(1, mat.shape[1]+1)
    for row in mat:
        plt.plot(ep, row, color="steelblue", alpha=.3)
    plt.plot(ep, mat.mean(0), color="crimson", lw=2.5, label="mean")
    plt.xlabel("Epoch"); plt.ylabel("RMSE"); plt.grid(alpha=.3); plt.legend()
    plt.tight_layout(); plt.savefig(f"rmse_{tag}.png", dpi=300); plt.close()

# ──────────────────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.manual_seed(0);  np.random.seed(0);  random.seed(0)
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--trials", type=int, default=10) #20
    ap.add_argument("--batch",  type=int, default=128)
    ap.add_argument("--hidden", type=int, default=100)
    ap.add_argument("--gain",   type=float, default=1.0)
    ap.add_argument("--lr",     type=float, default=5e-5)
    ap.add_argument("--dt",     type=float, default=0.02)
    ap.add_argument("--T",      type=int,   default=100)
    ap.add_argument("--freeze_B",action="store_true",
                    help="also freeze input projection B (default: train it)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                                              else "cpu")
    run_experiment(ap.parse_args())
