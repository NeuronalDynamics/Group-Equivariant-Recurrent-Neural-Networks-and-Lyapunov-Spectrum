# gblock_lstm.py  ── feature-orbit permutation-equivariant G-LSTM
# --------------------------------------------------------------
# hidden-to-hidden weights are block-constant
#   W = A ⊗ I_k + B ⊗ (J_k − I_k)
# so any permutation of the k orbits commutes with the update.
#
# Usage examples (identical CLI):
#   python gblock_lstm.py --calibrate --device cuda
#   python gblock_lstm.py --gain 5.0 --trials 40 --device cuda
#
# --------------------------------------------------------------
import os, argparse, pathlib, math, copy, numpy as np, torch
import torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.autograd.functional import jacobian
torch.set_default_dtype(torch.float32)
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# ───────────────────────────── data loader ─────────────────────────────
def get_loaders(batch=128, root="./torch_datasets"):
    tfm = transforms.ToTensor()
    root = pathlib.Path(root).expanduser()
    ds_tr = MNIST(root,  train=True,  download=True, transform=tfm)
    ds_te = MNIST(root,  train=False, download=True, transform=tfm)
    n_tr  = int(0.8 * len(ds_tr))
    ds_tr, ds_va = random_split(ds_tr, [n_tr, len(ds_tr) - n_tr])
    mk = lambda ds, shuf: DataLoader(ds, batch, shuffle=shuf, drop_last=True)
    return mk(ds_tr, True), mk(ds_va, False), mk(ds_te, False)

# ─────────────────────── block-constant linear map ─────────────────────
class BlockEqLinear(nn.Module):
    """
    Builds  W = A ⊗ I_k + B ⊗ (J_k − I_k)   (gate_dim blocks).
    Works for GRU (3 gates) and LSTM (4 gates).
    """
    def __init__(self, gate_dim: int, k: int, h: int, bias: bool = False):
        super().__init__()
        self.k, self.h = k, h
        self.A = nn.Parameter(torch.empty(gate_dim, h, h))
        self.B = nn.Parameter(torch.empty(gate_dim, h, h))
        self.b = nn.Parameter(torch.zeros(gate_dim * k * h)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.A)
        nn.init.xavier_normal_(self.B)        # ← independent B
        # bias already zeros if present

    def _weight(self) -> torch.Tensor:
        k, h, g = self.k, self.h, self.A.size(0)
        eye_k   = torch.eye(k,  device=self.A.device, dtype=self.A.dtype)
        ones_k  = torch.ones_like(eye_k)
        off_k   = ones_k - eye_k              # J_k − I_k
        blocks = []
        for g_idx in range(g):
            Wg = torch.kron(eye_k, self.A[g_idx]) + torch.kron(off_k,
                                                                self.B[g_idx])
            blocks.append(Wg)                 # (H,H)
        return torch.cat(blocks, dim=0)       # (g·H , H)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self._weight(), self.b)

# ─────────────────────── feature-orbit G-LSTM cell ─────────────────────
class GBlockLSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int,
                 k: int, bias: bool = False):
        super().__init__()
        assert hidden_size % k == 0, "hidden_size must be divisible by k"
        self.k, self.h = k, hidden_size // k
        self.in_lin  = nn.Linear(input_size, 4 * hidden_size, bias=bias)
        self.hh_lin  = BlockEqLinear(4, k, self.h, bias=bias)

    def forward(self, x_t, hc_prev):
        h_prev, c_prev = hc_prev
        gates = self.in_lin(x_t) + self.hh_lin(h_prev)   # (B,4H)
        i, f, g, o = gates.chunk(4, dim=-1)
        i = torch.sigmoid(i); f = torch.sigmoid(f); o = torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, c

# ─────────────────────────── SMNIST wrapper ────────────────────────────
class GBlockLSTMSMNIST(nn.Module):
    def __init__(self, hidden: int = 112, k: int = 4, dropout: float = .1):
        super().__init__()
        self.hidden, self.k = hidden, k
        self.cell = GBlockLSTMCell(28, hidden, k, bias=False)
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(hidden, 10, bias=False)

    def forward(self, x):
        B = x.size(0)
        seq = x.view(B, 28, 28)                          # 28 rows ⇒ 28 steps
        h = c = torch.zeros(B, self.hidden,
                            device=x.device, dtype=x.dtype)
        for t in range(28):
            h, c = self.cell(seq[:, t], (h, c))
        return self.fc(self.drop(h))

# ─────────────────────── critical-gain initialiser ─────────────────────
def critical_gblock_lstm_init(model: 'GBlockLSTMSMNIST',
                              g: float,
                              scheme: str = "orthogonal"):
    """
    Initialise so that   Var[W_hh] = (g / √H)²   in *both* irreducible
    subspaces (mean orbit and deviation orbit).  That keeps the spectral
    radius comparable to a vanilla LSTM regardless of k.

    If you want strictly identical spectra, set `force_zero_mean=True`
    below (Patch B).  Otherwise the default (Patch A) gives independent
    A and B with B rescaled by 1/√k.
    """
    H, k  = model.hidden, model.k
    h     = H // k
    σ_A   = g / math.sqrt(H)           # same σ as vanilla
    σ_B   = σ_A / math.sqrt(k)         # ← key rescale (Patch A)

    A, B  = model.cell.hh_lin.A, model.cell.hh_lin.B
    W_in  = model.cell.in_lin.weight

    with torch.no_grad():
        # ----- hidden‑to‑hidden blocks -----
        if scheme == "orthogonal":
            nn.init.orthogonal_(A);  A.mul_(σ_A)
            nn.init.orthogonal_(B);  B.mul_(σ_B)
        else:                                        # 'gaussian'
            nn.init.normal_(A, 0.0, σ_A)
            nn.init.normal_(B, 0.0, σ_B)

        # ----- input‑to‑hidden -----
        nn.init.normal_(W_in, 0.0, σ_A)
        '''
        # ------------------------------------------------------------------
        # OPTIONAL Patch B: set the **mean‑orbit eigenvalue to zero** exactly
        # by forcing  B = −A/(k−1)  (comment‑in ONE of the two lines below).
        # ------------------------------------------------------------------
        # force_zero_mean = True
        force_zero_mean = False
        # ------------------------------------------------------------------
        if force_zero_mean:
            B.copy_(A)                # start identical shapes
            B.mul_(-1.0 / (k - 1))    # B = −A/(k−1)  →  λ_mean = 0
        '''
    return g

# ───────────────────── equivariance sanity check ──────────────────────
@torch.no_grad()
def check_equivariance(model: GBlockLSTMSMNIST, tol: float = 1e-6):
    """
    Permute the k feature-orbits (P_k⊗I_h) and verify that
        f(x)        = model(x)
        f_P(x)      = model with permuted weights (mp)(x)
    are identical.  Only W_in rows and fc columns need permutation.
    """
    k, h  = model.k, model.hidden // model.k
    H     = model.hidden
    dev   = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    # 1️⃣ build a random orbit permutation
    P_k  = torch.eye(k, device=dev, dtype=dtype)[torch.randperm(k, device=dev)]
    Pbig = torch.kron(P_k, torch.eye(h, device=dev, dtype=dtype))       # (H,H)
    P4   = torch.block_diag(Pbig, Pbig, Pbig, Pbig)                     # (4H,4H)

    # 2️⃣ evaluation mode for deterministic logits (no dropout)
    model_eval = model.eval()
    x     = torch.randn(4, 1, 28, 28, device=dev, dtype=dtype)
    y_ref = model_eval(x)

    # 3️⃣ clone & permute the relevant weights
    mp = copy.deepcopy(model_eval)      # already in eval mode
    mp.fc.weight.data          = mp.fc.weight @ Pbig.T
    mp.cell.in_lin.weight.data = P4 @ mp.cell.in_lin.weight
    if mp.cell.in_lin.bias is not None:
        mp.cell.in_lin.bias.data = P4 @ mp.cell.in_lin.bias

    y_perm = mp(x)

    # 4️⃣ compare
    assert torch.allclose(y_ref, y_perm, atol=tol), "equivariance broken!"

# ─────────────────────── Lyapunov helpers (unchanged) ──────────────────
def rnn_jacobian_autograd(cell, x_t, h_prev, c_prev):
    h_prev = h_prev.detach().requires_grad_(True)
    c_prev = c_prev.detach().requires_grad_(True)
    H = h_prev.numel()
    def _f(hc):
        h, c = hc[:H], hc[H:]
        h, c = h.view_as(h_prev), c.view_as(c_prev)
        h1, c1 = cell(x_t, (h, c))
        return torch.cat([h1.flatten(), c1.flatten()])
    hc_prev = torch.cat([h_prev.flatten(), c_prev.flatten()])
    J = jacobian(_f, hc_prev, create_graph=False, strict=True)
    return J.detach()

def make_driver(batch=15, seq_len=600, device='cpu',
                dtype=torch.float64):
    return torch.rand(batch, seq_len, 28, device=device, dtype=dtype)

@torch.no_grad()
def lambda_max(net, warm=500, T=300, device="cuda"):
    torch.set_default_dtype(torch.float64)
    net = net.double().to(device)
    cell = net.cell
    H = 2 * net.hidden                                 # (h,c) flat
    h = c = torch.zeros(net.hidden, device=device, dtype=torch.float64)
    q = torch.randn(H, 1, device=device); q /= q.norm()
    drv = make_driver(1, warm + T, device=device)[0]
    log_r = 0.0
    for t in range(warm + T):
        J = rnn_jacobian_autograd(cell, drv[t], h, c)
        v = J @ q; r = v.norm(); q = v / (r + 1e-12)
        if t >= warm: log_r += torch.log(r + 1e-12)
        h, c = cell(drv[t], (h, c))
    return (log_r / T).item()

# full-spectrum QR (same pattern)
def lyap_spectrum(net, seq, warm=500):
    dev, dty = seq.device, seq.dtype
    cell = net.cell.to(dev).to(dty)
    H = 2 * net.hidden
    h = c = torch.zeros(net.hidden, device=dev, dtype=dty)
    Q = torch.eye(H, device=dev, dtype=dty)
    le = torch.zeros(H, device=dev, dtype=dty)
    # warm-up
    for t in range(warm):
        h, c = cell(seq[t], (h, c))
    eps, steps = 1e-12, 0
    for t in range(warm, seq.size(0)):
        J = rnn_jacobian_autograd(cell, seq[t], h, c)
        Q, R = torch.linalg.qr(J @ Q)
        le += torch.log(torch.clamp(torch.abs(torch.diagonal(R)), min=eps))
        h, c = cell(seq[t], (h, c)); steps += 1
    return (le / steps).cpu()

# ───────────────────────────── main CLI ────────────────────────────────
def calibrate(g_grid, device, hidden=112, k=4, n_repeat=10, driver_T=300):
    λs = []
    for g in g_grid:
        vals = []
        for _ in range(n_repeat):
            net = GBlockLSTMSMNIST(hidden, k).to(device)
            critical_gblock_lstm_init(net, g, scheme="gaussian")
            if _ == 0: check_equivariance(net)
            vals.append(lambda_max(net, T=driver_T, device=device))
        λ̄ = np.mean(vals); λs.append(λ̄)
        print(f"gain {g:4.2f}  →  λ_max = {λ̄:+7.4f}")
    plt.figure(figsize=(4,3)); plt.plot(g_grid, λs, marker="o")
    plt.axhline(0, color="k", ls="--", lw=.8)
    plt.xlabel("gain  g"); plt.ylabel(r"$\lambda_{\max}$")
    plt.title("G-Block-LSTM calibration"); plt.tight_layout()
    plt.savefig("lambda_max_vs_gain_gblstm.png", dpi=200)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hidden', type=int, default=112)
    ap.add_argument('--k',      type=int, default=8) #2x2x2x2x7=112 4 8 16 28
    ap.add_argument('--gain',   type=float, default=8.45)
    ap.add_argument('--trials', type=int, default=40)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--calibrate', action='store_true')
    args = ap.parse_args()
    dev = torch.device(args.device)

    if args.calibrate:
        g_grid = np.arange(args.gain - 1.0, args.gain + 1.0, 0.1)
        calibrate(g_grid, dev, hidden=args.hidden, k=args.k)
        return

    torch.set_default_dtype(torch.float64)
    all_LE = []
    for run in range(1, args.trials + 1):
        net = GBlockLSTMSMNIST(args.hidden, args.k).to(dev)
        g = critical_gblock_lstm_init(net, g=args.gain, scheme="gaussian")
        check_equivariance(net)
        driver = make_driver(15, 600, dev)
        spectra = [lyap_spectrum(net, seq, warm=500).numpy() for seq in driver]
        LE = np.mean(spectra, axis=0); all_LE.append(LE)
        np.save(f"lyap_pre_T{run}.npy", LE)
        print(f"[{run:03d}] λ₁={LE[0]:+.4f}, λ_H={LE[-1]:+.4f}")
    mean_LE = np.mean(all_LE, axis=0); np.save("lyap_pre_mean.npy", mean_LE)
    plt.figure(figsize=(4.5,3)); plt.plot(range(1, len(mean_LE)+1), mean_LE,
         marker='o', ms=3, lw=1.5); plt.axhline(0, c='k', ls='--', lw=.8)
    plt.xlabel('$i$'); plt.ylabel(r'$\lambda_i$')
    plt.title('G-Block-LSTM Lyapunov spectrum'); plt.tight_layout()
    plt.savefig("lyap_pre_mean.png", dpi=300)
    print("Saved  lyap_pre_mean.npy / .png")

if __name__ == "__main__":
    main()
