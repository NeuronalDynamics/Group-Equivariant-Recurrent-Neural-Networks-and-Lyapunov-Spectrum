# lyap_trials.py
import argparse, importlib.util, os, sys, itertools, gc, time
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"       # matplotlib + MKL fix on Windows

# --- numerics & plotting ------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.func import vmap, jvp
from torch.linalg import qr
from tqdm.auto import tqdm

# ────────────────────────────── CLI ───────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--trials", type=int, default=10,
                    help="number of independent realisations to average over")
parser.add_argument("--base-seed", type=int, default=1,
                    help="seed offset (each trial uses base_seed+i)")
parser.add_argument("--no-compile", action="store_true",
                    help="disable torch.compile even if Triton is present")
parser.add_argument("--no-plot", action="store_true",
                    help="skip all plotting (still prints the mean spectrum)")
args = parser.parse_args()

# ───────────────────────────── device & dtype ────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(DEVICE)
DTYPE  = torch.float32
if DEVICE.type == "cuda":
    print("Running on", torch.cuda.get_device_name(0))

# ───────────────────────────── model parameters ───────────────────────────────
N, g        = 100, 1.        # network size, coupling
DT, T_SIM   = 1e-1, 1e3
N_LE        = 100
T_ONS       = 1.0
STEP_ONS    = int(T_ONS / DT)
N_STEP      = int(T_SIM / DT)
t_total     = (N_STEP // STEP_ONS) * T_ONS

# ───────────────────────────── single‑trial kernel ────────────────────────────
def lyap_single(seed: int) -> np.ndarray:
    """Return Lyapunov spectrum for one random realisation (size N_LE)."""
    gen_net = torch.Generator(device=DEVICE).manual_seed(seed)
    gen_ic  = torch.Generator(device=DEVICE).manual_seed(seed + 10_000)
    gen_ons = torch.Generator(device=DEVICE).manual_seed(seed + 20_000)

    # random connectivity
    J = g * torch.randn((N, N), generator=gen_net, dtype=DTYPE) / np.sqrt(N)
    J.fill_diagonal_(0.)

    def f(h):
        return h*(1-DT) + DT * J @ torch.tanh(h)

    # optional TorchInductor compile
    if (DEVICE.type == "cuda"
        and importlib.util.find_spec("triton") is not None
        and not args.no_compile):
        try:
            f = torch.compile(f)
        except Exception as e:
            print("[WARN] torch.compile failed – using eager:", str(e).split("\n")[0])

    # initial state & tangent basis
    q, _ = qr(torch.randn(N, N_LE, generator=gen_ons, dtype=DTYPE))
    Ls   = torch.zeros(N_LE, dtype=DTYPE)
    h    = (g-1) * torch.randn(N, generator=gen_ic, dtype=DTYPE, device=DEVICE)

    # main loop
    for step in range(1, N_STEP + 1):
        h = f(h)

        # batched JVP
        tangents = q.T                                     # (N_LE, N)
        Dq = vmap(lambda v: jvp(f, (h,), (v,))[1])(tangents).T
        q  = Dq

        if step % STEP_ONS == 0:
            q, r = qr(q, mode="reduced")
            Ls += torch.log(torch.abs(torch.diag(r)))

    return (Ls / t_total).cpu().numpy()

# ───────────────────────────── run all trials ────────────────────────────────
all_specs = []
for k in tqdm(range(args.trials), desc="Trials", unit="trial"):
    spec = lyap_single(args.base_seed + k)
    all_specs.append(spec)
    # free GPU memory before next trial
    torch.cuda.empty_cache() if DEVICE.type == "cuda" else None
    gc.collect()

all_specs   = np.stack(all_specs)                    # shape (trials, N_LE)
mean_spec   = all_specs.mean(axis=0)
std_spec    = all_specs.std(axis=0)                  # useful for error bars

# ───────────────────────────── results & optional plot ───────────────────────
print("\nMean Lyapunov spectrum:")
print(mean_spec)

if not args.no_plot:
    x = np.arange(N_LE) / N_LE
    plt.figure(figsize=(6, 4))
    plt.plot(x, mean_spec, "o-", ms=3, lw=1, label="Autograd (mean)")
    plt.fill_between(x, mean_spec-std_spec, mean_spec+std_spec,
                     alpha=0.2, label=r"$\pm$1 s.d.")
    plt.axhline(0, linestyle=":", color="gray", lw=1)
    plt.xlabel(r"$i/N$")
    plt.ylabel(r"$\langle \lambda_i \rangle$  $(1/\tau)$")
    plt.title(f"Lyapunov spectrum averaged over {args.trials} trials")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
