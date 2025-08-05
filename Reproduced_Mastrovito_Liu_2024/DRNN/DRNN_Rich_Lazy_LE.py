# replicate_mastrovito_fig6.py  – full sweep for Mastrovito & Liu 2024 Fig. 6
import math, itertools, pathlib, json, torch, torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.func import vmap, jacrev
from tqdm.auto import tqdm

# ---------- constants -------------------------------------------------
GAINS       = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25,
               2.5, 2.75, 3.0, 5.0]
SEEDS       = range(3)                       # paper uses 10
MODEL_LIST  = ['Gaussian198']#, 'Gaussian28','Meso198', 'MesoSparse28']

DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
HIDDEN      = 198
INPUT_DIM   = 28          # one MNIST row per step
OUT_DIM     = 10
N_STEPS     = 28
T_LYAP      = 500
DTYPE       = torch.float32

# ---------- data ------------------------------------------------------
def get_data(batch=128):
    tfm   = transforms.ToTensor()
    root  = './torch_data'
    ds_tr = MNIST(root, train=True,  download=True, transform=tfm)
    ds_te = MNIST(root, train=False, download=True, transform=tfm)
    return (DataLoader(ds_tr, batch_size=batch, shuffle=True),
            DataLoader(ds_te, batch_size=batch, shuffle=False))

# ---------- helpers ---------------------------------------------------
def make_gaussian_mask(n_nonzero, size=HIDDEN, seed=None):
    rng = np.random.default_rng(seed)
    mask = np.zeros((size, size), dtype=bool)
    for i in range(size):
        mask[i, rng.choice(size, n_nonzero, replace=False)] = True
    return torch.from_numpy(mask)

def load_meso(thr=None):
    m = torch.from_numpy(np.load('mouse_meso.npy')).to(DTYPE)
    if thr is not None:
        m = m.where(m.abs() >= thr, torch.zeros_like(m))
    signs = torch.randint_like(m, low=0, high=2, dtype=torch.bool)
    return m * (1 - 2*signs)                 # random ±

# ---------- model -----------------------------------------------------
class SimpleRNN(nn.Module):
    def __init__(self, H, U, A, b, c):
        super().__init__()
        self.H = nn.Parameter(H)
        self.U = nn.Parameter(U)
        self.A = nn.Parameter(A)
        self.b = nn.Parameter(b)
        self.c = nn.Parameter(c)

    # logits for classification
    def forward(self, x):                    # x: (T,B,28)
        h = torch.zeros(x.size(1), HIDDEN, device=x.device, dtype=DTYPE)
        for t in range(x.size(0)):
            h = torch.tanh(h @ self.H.T + x[t] @ self.U + self.b)
        return h @ self.A.T + self.c         # (B,10)

    # hidden state after last time-step (for RA)
    def hidden(self, x):
        h = torch.zeros(x.size(1), HIDDEN, device=x.device, dtype=DTYPE)
        for t in range(x.size(0)):
            h = torch.tanh(h @ self.H.T + x[t] @ self.U + self.b)
        return h                             # (B,198)

# ---------- Lyapunov --------------------------------------------------
def lyapunov_max(model, steps=T_LYAP, show_bar=False):
    h = torch.zeros(HIDDEN, device=DEVICE, dtype=DTYPE)

    def step(hv):
        return torch.tanh(hv @ model.H.T + model.b.squeeze(0))

    Q    = torch.eye(HIDDEN, device=DEVICE, dtype=DTYPE)
    lyap = torch.zeros(HIDDEN, device=DEVICE)

    iterator = tqdm(range(steps), leave=False, desc="Lyap") if show_bar else range(steps)
    for _ in iterator:
        J    = torch.autograd.functional.jacobian(step, h)   # (H,H)
        J    = J @ Q
        Q, R = torch.linalg.qr(J, mode='reduced')
        lyap += torch.log(torch.diag(R).abs() + 1e-12)
        h     = step(h)
    return (lyap / steps).max().item()

# ---------- training --------------------------------------------------
def train(model, loader, lr=1e-3, epochs=100):
    model.train()
    opt      = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn  = nn.CrossEntropyLoss()
    for ep in range(epochs):
        for x, y in tqdm(loader, leave=False,
                         desc=f"Epoch {ep+1:03d}/{epochs}", unit="batch"):
            x = x.squeeze(1).permute(2,0,1).to(DEVICE)         # 28×B×28
            y = y.to(DEVICE)
            opt.zero_grad()
            loss_fn(model(x), y).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

# ---------- metrics ---------------------------------------------------
@torch.no_grad()
def accuracy(model, loader):
    model.eval()
    hits = total = 0
    for x, y in loader:
        x = x.squeeze(1).permute(2,0,1).to(DEVICE)
        y = y.to(DEVICE)
        hits  += (model(x).argmax(1) == y).sum().item()
        total += y.numel()
    return hits / total

def rep_alignment(Z0, Z1):
    R0, R1 = Z0.T @ Z0, Z1.T @ Z1
    return torch.trace(R1 @ R0) / (torch.linalg.norm(R1) * torch.linalg.norm(R0))

# ---------- main sweep ------------------------------------------------
def run_sweep():
    loader_tr, loader_te = get_data()
    results = []

    total = len(MODEL_LIST) * len(GAINS) * len(SEEDS)
    for model_type, g, seed in tqdm(itertools.product(MODEL_LIST, GAINS, SEEDS),
                                    total=total, desc="Total sweep", unit="run"):
        torch.manual_seed(seed); np.random.seed(seed)

        # recurrent weight matrix -----------------------------------
        if model_type.startswith('Gaussian'):
            k    = 198 if '198' in model_type else 28
            mask = make_gaussian_mask(k, seed=seed)
            H    = torch.randn_like(mask, dtype=DTYPE) * (g / math.sqrt(k))
            H    = H * mask
        elif model_type == 'Meso198':
            H = load_meso()
            H *= g / H.abs().std()
        else:                          # MesoSparse28
            H = load_meso(thr=0.0188)
            H *= g / H.abs().std()

        # other layers ---------------------------------------------
        U = torch.randn(INPUT_DIM, HIDDEN, dtype=DTYPE) * 0.05
        A = torch.zeros(OUT_DIM, HIDDEN, dtype=DTYPE)
        b = torch.zeros(1, HIDDEN, dtype=DTYPE)
        c = torch.zeros(1, OUT_DIM, dtype=DTYPE)

        net = SimpleRNN(H.to(DEVICE), U.to(DEVICE), A.to(DEVICE),
                        b.to(DEVICE), c.to(DEVICE))

        # evaluation batch (same for RA before / after) ------------
        x_eval, _ = next(iter(loader_te))
        x_eval = x_eval.squeeze(1).permute(2,0,1).to(DEVICE)     # 28×B×28

        λ0          = lyapunov_max(net)
        acts_before = net.hidden(x_eval).detach()

        # ------ training -----------------------------------------
        train(net, loader_tr)

        λT          = lyapunov_max(net)
        acc         = accuracy(net, loader_te)
        acts_after  = net.hidden(x_eval).detach()
        RA          = rep_alignment(acts_before, acts_after).item()
        ΔH          = (net.H.detach() - H.to(DEVICE)).norm().item()

        results.append(dict(model=model_type, g=g, seed=seed,
                            lam0=λ0, lamT=λT, acc=acc, RA=RA, dH=ΔH))

        tqdm.write(f"{model_type:14s} g={g:<4} seed={seed} "
                   f"λ0={λ0:+.3f} λT={λT:+.3f}  "
                   f"acc={acc*100:5.1f}%  RA={RA:.3f}  ΔH={ΔH:.4f}")

    pathlib.Path("fig6_results.json").write_text(json.dumps(results, indent=2))

# ---------------------------------------------------------------------
if __name__ == "__main__":
    run_sweep()
