# lstm_le.py
#
# Train-free Lyapunov spectrum for a 1-layer LSTM on SMNIST
# Re-implements Fig. 7 (pre-training) from Vogt 2024.

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
import matplotlib.pyplot as plt

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

# ================================================================
#  Critical initialisation  (vanilla LSTM)
# ================================================================
def critical_lstm_init(model: nn.Module, g: float = 1.0,
                       scheme: str = "orthogonal"):
    """
    Initialise a 1-layer LSTM so that Var(W) = (g/√H)².
    """
    H = model.hidden
    gain = g / math.sqrt(H)
    if scheme == "orthogonal":
        with torch.no_grad():
            nn.init.orthogonal_(model.lstm.weight_hh_l0)
            model.lstm.weight_hh_l0.mul_(g)          # scale safely
    else:
        nn.init.normal_(model.lstm.weight_hh_l0, 0.0, gain)

    nn.init.normal_(model.lstm.weight_ih_l0, 0.0, gain)
    #  optional: forget-gate bias trick (cf. Jozefowicz-15)
    '''
    with torch.no_grad():
        H4 = H                          # one gate = H
        model.lstm.bias_hh_l0[H4:2*H4] += 1.0
    '''
    return g

# ================================================================
#  λ_max  (works for LSTM – detects architecture)
# ================================================================
def lambda_max(net, *, warm=500, T=300, device="cuda"):
    """
    Fast 1-column QR method (Benettin) to estimate the largest
    Lyapunov exponent for either LSTM or G-LSTM.
    """
    torch.set_default_dtype(torch.float64)
    net = net.double().to(device)
    H = (net.G * net.K * 2) if hasattr(net, "G") else net.hidden*2

    # choose the correct cell clone
    if isinstance(net, LSTMSMNIST):
        cell = nn.LSTMCell(28, net.hidden, bias=False,
                           device=device, dtype=torch.float64)
        cell.load_state_dict({"weight_ih": net.lstm.weight_ih_l0,
                              "weight_hh": net.lstm.weight_hh_l0},
                             strict=False)
    else:                                    # G-LSTM
        cell = copy.deepcopy(net.cell).to(device).double()

    drv = make_driver(1, warm+T, device=device, dtype=torch.float64)[0]
    hc = torch.zeros(H, device=device)
    q  = torch.randn(H, 1, device=device); q /= q.norm()

    log_r = 0.0
    for t in range(warm+T):
        # step state
        if isinstance(net, LSTMSMNIST):
            h,c = torch.split(hc, net.hidden); hc = torch.cat(cell(drv[t], (h,c)))
        else:
            hc = step(net, drv[t], hc)
        if t < warm: continue
        # Jacobian
        if isinstance(net, LSTMSMNIST):
            J = lstm_jac_cell(cell, drv[t], hc, net.hidden)
        else:
            x_emb = row_to_group_feats(net, drv[t])
            J = glstm_jac_cell(cell, x_emb, hc)
        v = J @ q; r = v.norm(); q = v / (r + 1e-12)
        log_r += torch.log(r + 1e-12)
    '''
    for t in range(warm+T):
        # --------- 1. current state BEFORE advancing ---------
        h, c = torch.split(hc, net.hidden, dim=0)
        hc_curr = torch.cat([h, c])          # z_t  (requires no grad)

        # --------- 2. Jacobian for step  t → t+1 --------------
        if isinstance(net, LSTMSMNIST):
            J = lstm_jac_cell(cell, drv[t], hc_curr, net.hidden)
        else:
            x_emb = row_to_group_feats(net, drv[t])
            J     = glstm_jac_cell(cell, x_emb, hc_curr)

        # --------- 3. Benettin QR update ----------------------
        v = J @ q
        r = v.norm()
        q = v / (r + 1e-12)
        if t >= warm:
            log_r += torch.log(r + 1e-12)

        # --------- 4. advance to z_{t+1} -----------------------
        if isinstance(net, LSTMSMNIST):
            hc = torch.cat(cell(drv[t], (h, c)))     # z_{t+1}
        else:
            hc = step(net, drv[t], hc)
    '''
    return (log_r / T).item()

# ================================================================
#  Calibrate g  →  λ_max    (identical call-pattern as GRU code)
# ================================================================
def calibrate(hidden, g_grid, device,
              model_type="lstm",
              n_repeat=40, driver_T=300):
    λs = []
    for g in g_grid:
        vals = []
        for _ in range(n_repeat):
            net = LSTMSMNIST(hidden).to(device)
            critical_lstm_init(net, g=g, scheme="gaussian")
            vals.append(lambda_max(net, T=driver_T, device=device))
        λ̄ = np.mean(vals); λs.append(λ̄)
        print(f"gain {g:4.2f}  →  λ_max = {λ̄:+7.4f}")

    # plot
    plt.figure(figsize=(4,3))
    plt.plot(g_grid, λs, marker="o")
    plt.axhline(0, color="k", ls="--", lw=.8)
    plt.xlabel("gain  g"); plt.ylabel(r"$\lambda_{\max}$")
    plt.title(f"Calibration – {model_type.upper()}")
    plt.tight_layout(); plt.savefig(f"lambda_max_vs_gain_{model_type}.png", dpi=200)
    print("saved  lambda_max_vs_gain_*.png")

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


# ---------- spectrum utilities -------------------------------------------------
def one_step_jacobian_zero(net):
    """
    Return the 2H×2H Jacobian of one LSTM step around the origin (h=0, c=0, x=0).
    This is the matrix whose eigenvalues produce Engelken‑style spectra.
    """
    H  = net.hidden
    dev, dty = net.lstm.weight_ih_l0.device, net.lstm.weight_ih_l0.dtype
    cell = nn.LSTMCell(28, H, bias=False, device=dev, dtype=dty)
    cell.load_state_dict({'weight_ih': net.lstm.weight_ih_l0,
                          'weight_hh': net.lstm.weight_hh_l0}, strict=False)

    x   = torch.zeros(28,   device=dev, dtype=dty)
    hc  = torch.zeros(2*H, device=dev, dtype=dty)  # (h, c)=0
    return lstm_jac_cell(cell, x, hc, H)           # already 2H × 2H

# ---------- hidden-state (H×H) Jacobian block ----------  
def hidden_jacobian_zero(net):
    """
    dh_{t+1}/dh_t around (h=0, c=0, x=0).
    Returns an H×H matrix – this is what Engelken Fig. 1 uses.
    """
    J2H = one_step_jacobian_zero(net)   # 2H × 2H
    H   = net.hidden
    return J2H[:H, :H]                  # top–left block

def plot_eigcloud(eigs, g, *, save=True):
    plt.figure(figsize=(4,4))
    plt.scatter(eigs.real, eigs.imag, s=8, alpha=.7)
    plt.axvline(0, c='k', lw=.6); plt.axhline(0, c='k', lw=.6)
    plt.title(f'Eigenvalues of D  (g={g:.2f})')
    plt.xlabel('Re λ'); plt.ylabel('Im λ'); plt.gca().set_aspect('equal')
    if save:
        plt.tight_layout(); plt.savefig(f'eigcloud_g{g:.2f}.png', dpi=250)
    else:
        plt.show()

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hidden', type=int, default=224) #112 64
    ap.add_argument('--trials', type=int, default=40)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--gain', type=float, default=3.05,
                help='initialisation gain g (W~U[-g/sqrt(H),g/sqrt(H)])') #4.05 112 #3.05 224
    ap.add_argument('--calibrate', action='store_true',
                help='sweep g→λ_max and exit')
    ap.add_argument('--eigpre', action='store_true',
                help='compute & save eigen‑spectrum before training')
    args = ap.parse_args()


    if args.calibrate:
        g_grid = np.arange(2.5, 6.5, 0.1)    # or any range you like
        calibrate(args.hidden, g_grid,
                  device=torch.device(args.device),
                  model_type="lstm")
        return

    # -----------------------------------------------------------
    #  (optional) hidden‑state eigenvalue spectrum  –  pre‑training
    # -----------------------------------------------------------
    if args.eigpre:
        # (i) complex-eigenvalue cloud of dh/dh
        net = LSTMSMNIST(args.hidden).to(args.device)
        critical_lstm_init(net, g=args.gain, scheme="gaussian")
        D_h = hidden_jacobian_zero(net)
        eig = torch.linalg.eigvals(D_h).cpu().numpy()
        plot_eigcloud(eig, args.gain)     # unchanged helper
        print(f"saved  eigcloud_h_g{args.gain:.2f}.png")
        return


    dev = torch.device(args.device)
    all_LE, all_g = [], []

    for run in range(1, args.trials+1):
        net = LSTMSMNIST(args.hidden).to(dev)
        g   = critical_lstm_init(net, g=args.gain, scheme="gaussian")#init_scaled_uniform(net, args.gain)   # << new
        all_g.append(g)   
        print(f"\n=== Trial {run}/{args.trials}  (gain g = {g:.3f}) ===")

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
