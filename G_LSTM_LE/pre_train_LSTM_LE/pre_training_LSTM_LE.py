# lstm_smnist_lyap_pretrain.py
#
# Train-free Lyapunov spectrum for a 1-layer LSTM on SMNIST
# Re-implements Fig. 7 (pre-training) from Vogt 2024.

import os, argparse, pathlib, numpy as np, torch
import torch.nn as nn, torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import transforms
from tqdm import tqdm
from torch.autograd.functional import jacobian
import matplotlib.pyplot as plt

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# ---------- data loader (unchanged) ----------
def get_loaders(batch=128, root="./torch_datasets"):
    tfm = transforms.ToTensor()
    root = pathlib.Path(root).expanduser()
    ds_tr = MNIST(root, train=True,  download=True, transform=tfm)
    ds_te = MNIST(root, train=False, download=True, transform=tfm)
    n_tr  = int(0.8*len(ds_tr))
    ds_tr, ds_va = random_split(ds_tr, [n_tr, len(ds_tr)-n_tr])
    mk = lambda ds, shuf: DataLoader(ds, batch, shuffle=shuf, drop_last=True)
    return mk(ds_tr, True), mk(ds_va, False), mk(ds_te, False)

# ---------- 1-layer LSTM ----------
class LSTMSMNIST(nn.Module):
    def __init__(self, hidden=64, dropout=0.1):
        super().__init__()
        self.hidden = hidden
        self.lstm   = nn.LSTM(28, hidden, batch_first=True, bias=False)
        self.drop   = nn.Dropout(dropout)
        self.fc     = nn.Linear(hidden, 10, bias=False)
    def forward(self, x):
        B = x.size(0)
        seq = x.view(B, 28, 28)
        h0 = c0 = torch.zeros(1, B, self.hidden, device=x.device, dtype=x.dtype)
        y, _ = self.lstm(seq, (h0, c0))
        return self.fc(self.drop(y)[:, -1])

# ---------- init U(-p,p) ----------
def init_uniform(model, p_min=0.1, p_max=3.0):
    p = float(torch.round((torch.rand(1)*(p_max-p_min)+p_min)*1e3)/1e3)
    for w in (model.lstm.weight_ih_l0, model.lstm.weight_hh_l0):
        nn.init.uniform_(w, -p, p)
    return p

# ---------- Jacobian (2H×2H) ----------
def lstm_jac_cell(cell, x_t, hc_prev, H):
    hc_prev = hc_prev.detach().requires_grad_(True)
    def _f(hc):
        h, c = torch.split(hc, H)
        h1, c1 = cell(x_t, (h, c))
        return torch.cat([h1, c1])
    return jacobian(_f, hc_prev, create_graph=False).detach()

def lyap_spectrum(model, seq, *, warm=500):
    H = model.hidden; dev, dty = seq.device, seq.dtype
    cell = nn.LSTMCell(28, H, bias=False, device=dev, dtype=dty)
    cell.load_state_dict({'weight_ih': model.lstm.weight_ih_l0,
                          'weight_hh': model.lstm.weight_hh_l0}, strict=False)
    hc = torch.zeros(2*H, device=dev, dtype=dty)
    Q  = torch.eye(2*H, device=dev, dtype=dty)
    le = torch.zeros(2*H, device=dev, dtype=dty)

    for t in range(warm):
        h,c = torch.split(hc, H); hc = torch.cat(cell(seq[t], (h,c)))

    eps, steps = 1e-12, 0
    for t in range(warm, seq.size(0)):
        J = lstm_jac_cell(cell, seq[t], hc, H)
        Q, R = torch.linalg.qr(J @ Q)
        le  += torch.log(torch.clamp(torch.abs(torch.diagonal(R)), min=eps))
        h,c = torch.split(hc, H); hc = torch.cat(cell(seq[t], (h,c)))
        steps += 1
    return (le/steps).cpu()

def make_driver(batch=15, T=600, device='cpu', dtype=torch.float64):
    return torch.rand(batch, T, 28, device=device, dtype=dtype)

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hidden', type=int, default=64)
    ap.add_argument('--trials', type=int, default=200)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    dev = torch.device(args.device)
    all_LE, all_p = [], []

    for run in range(1, args.trials+1):
        net = LSTMSMNIST(args.hidden).to(dev)
        p   = init_uniform(net);  all_p.append(p)          # ← measure BEFORE any training

        # ------- Lyapunov spectrum (double precision) -------
        torch.set_default_dtype(torch.float64); net = net.double()
        driver = make_driver(device=dev)
        spectra = [lyap_spectrum(net, seq, warm=500).numpy() for seq in driver]
        LE = np.mean(spectra, axis=0)
        all_LE.append(LE)
        np.save(f"lyap_pre_T{run}.npy", LE)
        print(f"[{run:03d}] λ₁={LE[0]:+.5f}, λ_{2*args.hidden}={LE[-1]:+.5f}")

    mean_LE = np.mean(all_LE, axis=0); np.save("lyap_pre_mean.npy", mean_LE)
    plt.figure(figsize=(4.5,3.0))
    plt.plot(range(1,len(mean_LE)+1), mean_LE, 'o-', lw=1.5, ms=3)
    plt.axhline(0, color='k', ls='--', lw=.8)
    plt.xlabel('$i$'); plt.ylabel(r'$\lambda_i$')
    plt.title('LSTM Lyapunov spectrum (initial weights)')
    plt.tight_layout(); plt.savefig('lyap_pre_mean.png', dpi=300)
    print("\nSaved  lyap_pre_mean.npy and lyap_pre_mean.png")

if __name__ == '__main__':
    main()
