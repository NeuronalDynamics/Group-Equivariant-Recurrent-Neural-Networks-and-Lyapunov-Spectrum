#!/usr/bin/env python
# ================================================================
#  rnn_lyap_untrained_gain.py
#  ---------------------------------------------------------------
#  Reproduces the “before-training” Lyapunov-spectrum curve for a
#  vanilla RNN (tanh) as in Vogt et al. 2024, but with *standard*
#  Gaussian initialisation   W ~ N(0, g² / H)   instead of the
#  authors’ per-trial box U(−p,p).  No training is performed.
# ================================================================

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import os, math, argparse, numpy as np, torch
import torch.nn as nn
from torch.autograd.functional import jacobian
from tqdm import tqdm
import matplotlib.pyplot as plt

# ---------------------------  model  ----------------------------
class RNNSMNIST(nn.Module):
    """Single-layer Elman RNN, 28 inputs ➜ H hidden units ➜ 10-way head."""
    def __init__(self, hidden=64):
        super().__init__()
        self.hidden = hidden
        self.rnn = nn.RNN(28, hidden, batch_first=True, bias=True)  # tanh
        self.fc  = nn.Linear(hidden, 10, bias=False)

    def forward(self, x):
        B = x.size(0)
        seq = x.view(B, 28, 28)                      # row-wise unfolding
        h0  = torch.zeros(1, B, self.hidden, device=x.device, dtype=x.dtype)
        y, _ = self.rnn(seq, h0)
        return self.fc(y[:, -1])                     # last time-step

# --------------------  Gaussian gain initialiser  ----------------
def init_gain(model: nn.Module, g: float = 1.5) -> float:
    """
    Draw all recurrent and input weights i.i.d.  N(0, g²/H).
    Biases are set to zero.  Returns the gain g.
    """
    H = model.hidden
    std = g / math.sqrt(H)
    for w in (model.rnn.weight_ih_l0, model.rnn.weight_hh_l0):
        nn.init.normal_(w, mean=0.0, std=std)
    nn.init.zeros_(model.rnn.bias_ih_l0)
    nn.init.zeros_(model.rnn.bias_hh_l0)
    return g

# ----------------------  Jacobian helper  -----------------------
def rnn_J(cell: nn.RNNCell, x_t: torch.Tensor, h_prev: torch.Tensor):
    """J_t = ∂h_t / ∂h_{t−1} for one RNNCell step (autograd)."""
    h_prev = h_prev.detach().requires_grad_(True)
    return jacobian(lambda h: cell(x_t, h),
                    h_prev,
                    create_graph=False, strict=True).detach()

# -------------------  Lyapunov spectrum  ------------------------
def lyap_spectrum(model, seq, *, warm=500):
    """
    Full Lyapunov spectrum for one input sequence (T × 28 tensor).
    Discards `warm` steps, then QR-accumulates exponents.
    """
    H, dev, dty = model.hidden, seq.device, seq.dtype
    # clone weights into an explicit RNNCell
    cell = nn.RNNCell(28, H, bias=True, device=dev, dtype=dty)
    cell.load_state_dict({
        'weight_ih': model.rnn.weight_ih_l0,
        'weight_hh': model.rnn.weight_hh_l0,
        'bias_ih'  : model.rnn.bias_ih_l0,
        'bias_hh'  : model.rnn.bias_hh_l0
    })

    h = torch.zeros(H, device=dev, dtype=dty)
    Q = torch.eye(H, device=dev, dtype=dty)
    le_sum = torch.zeros(H, device=dev, dtype=dty)
    steps  = 0
    eps = 1e-12

    # warm-up
    for t in range(warm):
        h = cell(seq[t], h)

    # QR accumulation
    for t in range(warm, seq.size(0)):
        J = rnn_J(cell, seq[t], h)          # (H×H)
        Q, R = torch.linalg.qr(J @ Q)       # reduced QR
        le_sum += torch.log(torch.clamp(torch.abs(torch.diagonal(R)), min=eps))
        h = cell(seq[t], h)
        steps += 1

    return (le_sum / steps).cpu()           # (H,)

# --------------------  random driver signal  --------------------
def make_driver(batch=15, seq_len=600, *, device, dtype):
    """15 sequences of i.i.d. U(0,1) vectors (length seq_len, dim 28)."""
    return torch.rand(batch, seq_len, 28, device=device, dtype=dtype)

# ----------------------------  main  ----------------------------
def main():
    torch.set_default_dtype(torch.float64)          # double precision
    ap = argparse.ArgumentParser()
    ap.add_argument('--hidden', type=int,   default=64,
                    help='hidden size H')
    ap.add_argument('--gain',   type=float, default=1.5,
                    help='g in  W~N(0, g²/H)')
    ap.add_argument('--trials', type=int,   default=40,
                    help='number of independent random draws')
    ap.add_argument('--device', default='cuda'
                    if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    dev = torch.device(args.device)

    all_LE, all_g = [], []
    for run in range(1, args.trials + 1):
        net = RNNSMNIST(args.hidden).to(dev).double()
        g   = init_gain(net, args.gain)
        all_g.append(g)
        print(f"\n=== untrained trial {run}/{args.trials}   g = {g:.2f} ===")

        driver = make_driver(device=dev, dtype=torch.float64)
        LE_batch = []
        for seq in tqdm(driver, desc='Lyap-QR', leave=False):
            LE = lyap_spectrum(net, seq, warm=500)
            LE_batch.append(LE.numpy())
        LE = np.mean(LE_batch, axis=0)
        all_LE.append(LE)
        np.save(f"LE_trial_{run:03d}.npy", LE)
        print(f"  λ₁ = {LE[0]:+.6f}   λ_H = {LE[-1]:+.6f}")

    # average over trials
    mean_LE = np.mean(all_LE, axis=0)
    np.save("LE_mean.npy", mean_LE)

    plt.figure(figsize=(4.5, 3.2))
    plt.plot(range(1, len(mean_LE)+1), mean_LE,
             marker='o', ms=3, lw=1.4, label='mean spectrum')
    plt.axhline(0, color='k', ls='--', lw=.8)
    plt.xlabel('exponent index  $i$')
    plt.ylabel(r'$\bar{\lambda}_i$')
    plt.title(f'Untrained RNN  (g = {args.gain},  H = {args.hidden})')
    plt.tight_layout()
    plt.savefig('LE_mean.png', dpi=300)
    print("\nSaved  LE_mean.npy  and  LE_mean.png")
    print("g values:", ', '.join(f"{x:.2f}" for x in all_g))

if __name__ == '__main__':
    main()
