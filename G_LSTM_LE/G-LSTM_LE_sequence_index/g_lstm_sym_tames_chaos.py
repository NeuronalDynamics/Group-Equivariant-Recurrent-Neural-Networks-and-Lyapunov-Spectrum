# ---------------------------------------------------------------------
#  g_lstm.py  –  permutation-equivariant 1-layer G-LSTM for SMNIST
#               (exact Supplement-B equations)
# ---------------------------------------------------------------------

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import os, argparse, pathlib, numpy as np, torch
import torch.nn as nn, torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import transforms
from tqdm import tqdm
from torch.autograd.functional import jacobian
import matplotlib.pyplot as plt
import math
import itertools

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


# ---------------- G-LSTM *cell* --------------------------------------
class GLSTMCell(nn.Module):
    r"""
    Supplement-B equations, written in tensor form.
    All signals are |G|×K tensors (here: 28×K) storing a
    *function on the permutation group*  g ↦ R^K.

        i = σ( x * ψ_ii + h * ψ_ih )
        f = σ( x * ψ_fi + h * ψ_fh )
        g = τ( x * ψ_gi + h * ψ_gh )
        o = σ( x * ψ_oi + h * ψ_oh )
        c' = f ⊙ c  +  i ⊙ g
        h' = o ⊙ τ(c')

    Because the linear maps are group-convolutions with a *delta*
    support, they reduce to **shared linear layers applied slot-wise**.
    """
    def __init__(self, K: int, bias: bool = False):
        super().__init__()
        self.K = K
        # shared weights – identical for every group element
        self.W_x = nn.Linear(K, 4 * K, bias=bias)   # ψ_{•i}
        self.W_h = nn.Linear(K, 4 * K, bias=bias)   # ψ_{•h}

    def forward(self, x, h, c):
        # x, h, c :  (B, |G|, K)
        gates = self.W_x(x) + self.W_h(h)           # (B, |G|, 4K)
        i, f, g, o = torch.chunk(gates, 4, dim=-1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g       = torch.tanh(g)
        c_next  = f * c + i * g
        h_next  = o * torch.tanh(c_next)
        return h_next, c_next
# ---------------------------------------------------------------------

# ---------------- Full SMNIST network --------------------------------
class GLSTMSMNIST(nn.Module):
    """
    * Input    : 28×28 image, read row-by-row  (T = |G| = 28 steps)
    * Hidden   : |G| × K    (default 28 × 4  ⇒ 112 channels total)
    * Output   : 10-way classifier (same as before – unused for LE)
    """
    def __init__(self, group_size: int = 28, K: int = 4, dropout: float = .1):
        super().__init__()
        self.G         = group_size         # |G|
        self.K         = K                 # “channels’’ per slot
        # 1-D “embedding’’   R¹ → R^K   (scalar pixel → K-dim feature)
        self.in_proj    = nn.Linear(1, K, bias=False)
        self.cell       = GLSTMCell(K, bias=False)
        self.drop       = nn.Dropout(dropout)
        self.fc         = nn.Linear(group_size * K, 10, bias=False)

    # ---------- helper used by Lyapunov code ----------
    def step(self, x_row, hc_flat):
        """
        One recurrent step **without** autograd side-effects.
        x_row    : (|G|,)               – raw pixel row
        hc_flat  : (2*|G|*K,)           – flattened [h, c]
        returns  : new (2*|G|*K,)       – flattened [h', c']
        """
        B = 1                                            # Lyapunov driver is batch-1
        h, c   = hc_flat.view(2, self.G, self.K)         # (|G|,K)
        x_row  = x_row.unsqueeze(-1)                     # (|G|,1)
        x_emb  = self.in_proj(x_row)                     # (|G|,K)
        h, c   = self.cell(x_emb.unsqueeze(0),           # add batch dim
                           h.unsqueeze(0),
                           c.unsqueeze(0))
        return torch.cat([h.squeeze(0), c.squeeze(0)]).view(-1)

    def forward(self, x):
        """
        Standard SMNIST inference (only used if you *train* the model).
        """
        B = x.size(0)
        seq = x.view(B, 28, self.G)                      # (B,T,G)
        h = c = torch.zeros(B, self.G, self.K,
                            dtype=x.dtype, device=x.device)
        for t in range(seq.size(1)):
            x_emb = self.in_proj(seq[:, t].unsqueeze(-1))        # (B,G,K)
            h, c  = self.cell(x_emb, h, c)
        out = self.fc(self.drop(h.reshape(B, -1)))
        return out
# ---------------------------------------------------------------------

# ---------- init U(-p,p) ----------
def init_scaled_uniform(model: nn.Module, g: float = 1.0) -> float:
    """
    Uniform[-g/sqrt(H), +g/sqrt(H)] on all recurrent & input weights.
    Returns the gain actually used so the caller can log it.
    """
    H  = model.hidden
    lim = g / math.sqrt(H)          # scale factor
    for w in (model.lstm.weight_ih_l0, model.lstm.weight_hh_l0):
        nn.init.uniform_(w, -lim, lim)
    #nn.init.zeros_(model.lstm.bias_ih_l0)
    #nn.init.zeros_(model.lstm.bias_hh_l0)
    return g                         # purely for bookkeeping

# ---------------- Critical initialisation ----------------------------
def critical_glstm_init(model: GLSTMSMNIST, *,
                        g: float = 1.0, scheme: str = "gaussian"):
    """
    Initialise every filter so that  Var(W) = (g / √K)²,
    matching the √H scaling you used for the vanilla LSTM.
    """
    K   = model.K
    std = g / math.sqrt(K)

    if scheme == "orthogonal":
        nn.init.orthogonal_(model.cell.W_h.weight)
        model.cell.W_h.weight.data.mul_(g)
        nn.init.orthogonal_(model.cell.W_x.weight)
        model.cell.W_x.weight.data.mul_(g)
    else:   # 'gaussian'
        nn.init.normal_(model.cell.W_h.weight, 0.0, std)
        nn.init.normal_(model.cell.W_x.weight, 0.0, std)

    # ⇢ keep derivatives alive: bias_f = +1  (indices K:2K)
    if model.cell.W_x.bias is not None:
        K = model.K
        model.cell.W_x.bias.data[K:2*K].fill_(1.0)
        model.cell.W_h.bias.data[K:2*K].fill_(1.0)

    # (no bias parameters to reset)
    return g
# ---------------------------------------------------------------------

# ---------- Convenience wrappers for your Lyapunov code -------------
@torch.no_grad()
def row_to_group_feats(net: GLSTMSMNIST, row):
    """Helper used by glstm_jac_cell in your script."""
    return net.in_proj(row.unsqueeze(-1))           # (|G|,K)

def step(net: GLSTMSMNIST, row, hc_flat):
    """One step for λ-max estimation (Benettin QR)."""
    return net.step(row, hc_flat)

# Jacobian of the G-LSTM cell (2·|G|·K  ×  2·|G|·K)
def glstm_jac_cell(cell: GLSTMCell, x_emb, hc_flat):
    """
    Autograd Jacobian used by your Lyapunov spectrum code.
    """
    G, K = x_emb.size(0), x_emb.size(-1)
    hc_flat = hc_flat.detach().requires_grad_(True)
    h, c = hc_flat.view(2, G, K)

    def _f(hc):
        h0, c0 = hc.view(2, G, K)
        h1, c1 = cell(x_emb.unsqueeze(0),      # add batch dim
                      h0.unsqueeze(0), c0.unsqueeze(0))
        return torch.cat([h1.squeeze(0), c1.squeeze(0)]).view(-1)

    return torch.autograd.functional.jacobian(_f, hc_flat,
                                              create_graph=False).detach()
# ---------------------------------------------------------------------

# ================================================================
#  1-column Benettin QR   →   λ_max  (G-LSTM version only)
# ================================================================
@torch.no_grad()
def lambda_max(net, *, warm=500, T=300, device="cuda"):
    """
    Largest Lyapunov exponent of a G-LSTM.
    * Works for full fp64 reproducibility.
    * Evaluates J BEFORE the state update (Benettin order).
    """
    import copy, math, torch
    torch.set_default_dtype(torch.float64)
    net  = net.double().to(device)
    cell = copy.deepcopy(net.cell).to(device)

    H   = 2 * net.G * net.K                         # (h,c) flat dim
    z   = torch.zeros(H,  device=device)            # state  (h,c)
    q   = torch.randn(H, 1, device=device); q /= q.norm()

    drv = make_driver(1, warm+T, device=device, dtype=torch.float64)[0]
    log_r = 0.0

    for t in range(warm+T):
        # Jacobian at current state
        x_emb = row_to_group_feats(net, drv[t])
        J     = glstm_jac_cell(cell, x_emb, z)

        v = J @ q
        r = v.norm()
        q = v / (r + 1e-12)
        if t >= warm:
            log_r += torch.log(r + 1e-12)

        # advance the system AFTER tangent update
        z = step(net, drv[t], z)

    return (log_r / T).item()


# ================================================================
#  Calibrate g  →  λ_max    (identical call-pattern as GRU code)
# ================================================================
def calibrate(g_grid, *, device, group_size=28, K=4,
              n_repeat=30, driver_T=300):
    λs = []
    #############################################################################
    '''
    for g in g_grid:
        vals = []
        for _ in range(n_repeat):
            net = GLSTMSMNIST(group_size, K).to(device)
            critical_glstm_init(net, g=g, scheme="gaussian")
            ok, err = check_equivariance(net, n_trials=32)
            print(f"Permutation-equivariant?  {ok}   (max |Δ| = {err:.2e})")
            vals.append(lambda_max(net, T=driver_T, device=device))
        λ̄ = np.mean(vals); λs.append(λ̄)
        print(f"gain {g:4.2f}  →  λ_max = {λ̄:+7.4f}")
    '''
    for g in g_grid:
        vals = []
        for _ in range(n_repeat):
            # ➋ build a *new* network each repetition
            net = GLSTMSMNIST(group_size, K).to(device)
            critical_glstm_init(net, g=g, scheme="gaussian")

            # optional: check equivariance only once per g
            if _ == 0:
                ok, err = check_equivariance(net, n_trials=8)
                print(f"g={g:.2f}  equivariance ok={ok}  maxΔ={err:.1e}")

            lam = lambda_max(net, warm=1000, T=1000, device=device)
            vals.append(lam)

        λ̄ = np.mean(vals)
        σ   = np.std(vals) / np.sqrt(n_repeat)
        λs.append(λ̄)                               # ➌ store for plot
        print(f"g={g:.2f} → λ_max = {λ̄:+.4f} ± {σ:.4f}")
    #############################################################################

    plt.figure(figsize=(4,3))
    plt.plot(g_grid, λs, marker="o")
    plt.axhline(0, color="k", ls="--", lw=.8)
    plt.xlabel("gain  g"); plt.ylabel(r"$\lambda_{\max}$")
    plt.title("Calibration – G-LSTM")
    plt.tight_layout(); plt.savefig("lambda_max_vs_gain_glstm.png", dpi=200)
    print("saved  lambda_max_vs_gain_glstm.png")


# ================================================================
#  Full spectrum  λ₁ … λ_{2·|G|·K}  (G-LSTM only)
# ================================================================
def lyap_spectrum_glstm(net, seq, *, warm=500):
    G, K  = net.G, net.K
    H     = 2 * G * K
    dev   = seq.device
    dty   = seq.dtype
    cell  = net.cell.to(dev).to(dty)

    z  = torch.zeros(H, device=dev, dtype=dty)   # state
    Q  = torch.eye(H, device=dev, dtype=dty)     # basis
    le = torch.zeros(H, device=dev, dtype=dty)

    # burn-in
    for t in range(warm):
        z = step(net, seq[t], z)

    eps, steps = 1e-12, 0
    for t in range(warm, seq.size(0)):
        # 1️⃣ J BEFORE advancing z
        x_emb = row_to_group_feats(net, seq[t])
        J     = glstm_jac_cell(cell, x_emb, z)

        # 2️⃣–3️⃣ tangent update + QR
        Q, R  = torch.linalg.qr(J @ Q)
        le   += torch.log(torch.clamp(torch.abs(torch.diagonal(R)), min=eps))

        # 4️⃣ advance state
        z = step(net, seq[t], z)
        steps += 1

    return (le / steps).cpu()


def make_driver(batch=15, T=600, device='cpu', dtype=torch.float64):
    return torch.rand(batch, T, 28, device=device, dtype=dtype)

# ================================================================
#  permutation-equivariance checker  (G-LSTM only)
# ================================================================
@torch.no_grad()
def check_equivariance(net: GLSTMSMNIST, n_trials=16,
                       atol=1e-6, rtol=1e-6,
                       device="cuda" if torch.cuda.is_available() else "cpu"):
    import random, math, torch
    net = net.to(device).double()
    G, K = net.G, net.K
    dtype = net.in_proj.weight.dtype        # <── added

    def _permute_vec(vec, perm):
        h, c = vec.view(2, G, K)
        return torch.cat([h[perm], c[perm]]).view(-1)

    max_err = 0.0
    for _ in range(n_trials):
        perm = torch.randperm(G, device=device)

        hc_flat  = torch.randn(2*G*K, device=device, dtype=dtype)   # <── dtype
        hc_perm  = _permute_vec(hc_flat, perm)

        row      = torch.randn(G, device=device, dtype=dtype)       # <── dtype
        row_perm = row[perm]

        hc_next      = step(net, row,      hc_flat)
        hc_next_perm = step(net, row_perm, hc_perm)

        hc_next_perm_by_hand = _permute_vec(hc_next, perm)
        err = torch.max(torch.abs(hc_next_perm - hc_next_perm_by_hand)).item()
        max_err = max(max_err, err)

        if err > atol + rtol * torch.max(torch.abs(hc_next_perm)).item():
            return False, max_err

    return True, max_err

# ─────────────────────────  SPECTRUM UTILITIES  ──────────────────────────
def one_step_jacobian_zero_glstm(net):
    """
    Full 2·|G|·K × 2·|G|·K Jacobian of a single G-LSTM step
    around the silent state (h=0,c=0) and zero input row.
    """
    G, K  = net.G, net.K
    dev   = next(net.parameters()).device
    dty   = next(net.parameters()).dtype

    hc0   = torch.zeros(2*G*K, device=dev, dtype=dty)
    row   = torch.zeros(G,     device=dev, dtype=dty)       # x_t = 0
    x_emb = row_to_group_feats(net, row)                    # (G,K)

    return glstm_jac_cell(net.cell, x_emb, hc0)             # 2GK × 2GK


def hidden_jacobian_zero_glstm(net):
    """
    dh_{t+1}/dh_t block used in Engelken Fig. 1.
    Shape: (|G|·K) × (|G|·K)
    """
    J2H = one_step_jacobian_zero_glstm(net)
    H   = net.G * net.K
    return J2H[:H, :H]          # top-left block


def plot_eigcloud(eigs, g, *, save=True, fname_prefix='eigcloud_glstm'):
    plt.figure(figsize=(4, 4))
    plt.scatter(eigs.real, eigs.imag, s=8, alpha=.7)
    plt.axvline(0, c='k', lw=.6); plt.axhline(0, c='k', lw=.6)
    plt.xlabel('Re λ'); plt.ylabel('Im λ'); plt.gca().set_aspect('equal')
    plt.title(f'Eigenvalues of dh/dh   (g={g:.2f})')
    if save:
        fn = f'{fname_prefix}_g{g:.2f}.png'
        plt.tight_layout(); plt.savefig(fn, dpi=250)
        print(f'saved  {fn}')
    else:
        plt.show()

# ─────────────────────  intra-block sorter  ──────────────────────
def sort_blocks(vec, block):
    """
    vec   : 1-D tensor or NumPy array of length 2*block
            (first block = h, second block = c)
    block : |G|*K   – size of each degenerate block
    Returns a NumPy array where both halves are sorted descending.
    """
    v = np.asarray(vec)
    h_sorted = np.sort(v[:block])[::-1]        # descending
    c_sorted = np.sort(v[block:])[::-1]
    return np.concatenate([h_sorted, c_sorted])

# ────────────────────────────────────────────────────────────────
#  Stair‑case detection utilities
# ────────────────────────────────────────────────────────────────
def staircase_blocks(spectrum, tol=1e-3):
    """
    Group a *descending* Lyapunov spectrum into contiguous blocks whose
    within‑block differences are ≤ tol.

    Parameters
    ----------
    spectrum : 1‑D NumPy array (already sorted descending)
    tol      : float           – equality threshold

    Returns
    -------
    blocks : list[tuple]
        [(start_idx, end_idx, lambda_value), ...]   1‑based indices
    """
    blocks, start = [], 0
    for i in range(1, len(spectrum)):
        if abs(spectrum[i] - spectrum[i-1]) > tol:
            blocks.append((start, i-1, spectrum[start]))
            start = i
    blocks.append((start, len(spectrum)-1, spectrum[start]))  # last block
    return [(s+1, e+1, val) for s, e, val in blocks]          # 1‑based idx

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--group_size', type=int, default=28,
                    help='|G| – number of permutation slots (28 for SMNIST)')
    ap.add_argument('--K', type=int, default=4, #4 8 16 28
                    help='channels per slot – hidden size is |G|·K')
    ap.add_argument('--trials', type=int, default=40)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--gain', type=float, default=6.4) #4.95 edge of chaos #10.0 chaotics lazy high gain #2.0 ordered rich low gain
    ap.add_argument('--calibrate', action='store_true',
                    help='sweep g→λ_max curve and exit')
    ap.add_argument('--eigpre',    action='store_true',
                    help='plot eigen-cloud + radius-vs-gain and exit')
    ap.add_argument('--tol', type=float, default=1e-3,
                help='two consecutive λ are considered equal if |Δ|≤tol')
    args = ap.parse_args()

    dev = torch.device(args.device)

    if args.calibrate:
        g_grid = np.arange(5.5, 25.5, 0.5)
        calibrate(g_grid,
                  device=dev,
                  group_size=args.group_size,
                  K=args.K)
        return

    if args.eigpre:
        # (a) eigenvalue cloud for a single gain
        net = GLSTMSMNIST(args.group_size, args.K).to(args.device)
        critical_glstm_init(net, g=args.gain, scheme='gaussian')
        D_h = hidden_jacobian_zero_glstm(net)
        eig = torch.linalg.eigvals(D_h).cpu().numpy()
        #print(eig)
        plot_eigcloud(eig, args.gain)
        print(f"saved  eigcloud_h_g{args.gain:.2f}.png")
        return

    all_LE, all_g = [], []
    for run in range(1, args.trials + 1):
        net = GLSTMSMNIST(args.group_size, args.K).to(dev)
        g   = critical_glstm_init(net, g=args.gain, scheme="gaussian")
        all_g.append(g)
        ok, err = check_equivariance(net, n_trials=32)
        print(f"Permutation-equivariant?  {ok}   (max |Δ| = {err:.2e})")
        print(f"\n=== Trial {run}/{args.trials}  (gain g = {g:.3f}) ===")

        torch.set_default_dtype(torch.float64); net = net.double()
        driver   = make_driver(device=dev)
        ################################################################
        spectra  = [lyap_spectrum_glstm(net, seq, warm=500).numpy()
                    for seq in driver]
        #LE       = np.mean(spectra, axis=0)
        LE_raw   = np.mean(spectra, axis=0)

        # -------- NEW: sort inside each degenerate block ----------
        block    = args.group_size * args.K        # |G|·K
        LE       = sort_blocks(LE_raw, block)
        all_LE.append(LE)
        ################################################################
        '''
        block   = args.group_size * args.K
        spectra = []

        for seq in driver:
            LE_run = lyap_spectrum_glstm(net, seq, warm=500).numpy()
            LE_run = sort_blocks(LE_run, block)   # sort *here*
            spectra.append(LE_run)

        LE = np.mean(spectra, axis=0)             # already aligned
        all_LE.append(LE)
        '''
        ################################################################
        np.save(f"lyap_pre_T{run}.npy", LE)
        print(f"[{run:03d}] λ₁={LE[0]:+.5f}, λ_{2*args.group_size*args.K}={LE[-1]:+.5f}")

    mean_LE = np.mean(all_LE, axis=0); np.save("lyap_pre_mean.npy", mean_LE)
    
    blocks = staircase_blocks(mean_LE, tol=args.tol)

    print(f"\nDetected {len(blocks)} staircase block(s) with tol={args.tol}:")
    for k, (s, e, val) in enumerate(blocks, 1):
        mult = e - s + 1
        print(f"  Block {k:>2}:  λ ≈ {val:+.4f}   multiplicity = {mult:>3} "
              f"(indices {s}–{e})")

    plt.figure(figsize=(4.5,3.0))
    plt.plot(range(1, len(mean_LE) + 1), mean_LE, 'o-', lw=1.5, ms=3)
    plt.axhline(0, color='k', ls='--', lw=.8)
    plt.xlabel('$i$'); plt.ylabel(r'$\lambda_i$')
    plt.title('G-LSTM Lyapunov spectrum (initial weights)')

    # annotate each plateau with its multiplicity
    for s, e, val in blocks:
        mid = (s + e) / 2        # x‑coordinate (already 1‑based)
        mult = e - s + 1
        plt.text(mid, val + 0.15, str(mult),
                 ha='center', va='bottom', fontsize=8)

    plt.tight_layout(); plt.savefig('lyap_pre_mean.png', dpi=300)
    print("\nSaved  lyap_pre_mean.npy and lyap_pre_mean.png")

if __name__ == '__main__':
    main()
