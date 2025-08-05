import argparse, importlib.util, os, sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"  # matplotlib + MKL fix on Windows

# --- plotting & numerics ------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import torch
import math
from torch.func import vmap, jvp              # PyTorch≥2.1
from torch.linalg import qr
from tqdm.auto import tqdm


# ----------  path to Engelken spectrum  ----------
spec_path = "Lyapunov_spectrum.npy"        # ← adjust if you used another name
engelken_spec = None
if os.path.exists(spec_path):
    engelken_spec = np.load(spec_path)
    N_LE_saved     = engelken_spec.size
else:
    print(f"[INFO]   '{spec_path}' not found – will plot only autograd spectrum.")

# ----------  CLI -------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--no-compile", action="store_true",
                    help="disable torch.compile even if Triton is present")
args = parser.parse_args()

# ----------  device & dtype ---------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(DEVICE)
DTYPE  = torch.float32            # FP32 is ∼30× faster than FP64 on consumer GPUs
if DEVICE.type == "cuda":
    print("Running on", torch.cuda.get_device_name(0))

# ------------------------------------------------------------------
# 0. choose translation-equivariant parameters
# ------------------------------------------------------------------
sigma   = 40.0                     # tuning width  (deg)
rho_deg = 0.5                      # neuron density (cells / deg)
N       = 180                      # must match rho_deg * 360
dx      = 1 / rho_deg
wr      = 1.05 * 0.896             # 5 % above critical weight (see Eq. S11)
k       = 5e-4
rho_w   = rho_deg                  # same symbol as in the paper

# positions on the ring
xs = torch.linspace(-180, 180-dx, N, device=DEVICE)

# translation-invariant kernel W_r (Toeplitz-circulant)
Δx   = (xs[:,None] - xs[None,:] + 180) % 360 - 180
Wmat = (wr/(math.sqrt(2*math.pi)*sigma) *
        torch.exp(-(Δx**2)/(2*sigma**2))) * dx      # includes integral weight
Wmat = Wmat.to(DTYPE)


'''
J = g*torch.randn((N, N), generator=GEN_IC, dtype=DTYPE, device=DEVICE) / np.sqrt(N)
J.fill_diagonal_(0.)
def f(h):                     # Euler step
    return h*(1-DT) + DT * J @ torch.tanh(h)
'''
# ------------------------------------------------------------------
# 1. replace J & f by the convolutional version
# ------------------------------------------------------------------
def f(h):
    """Euler step for translation-equivariant CAN"""
    return h*(1-DT) + DT * (rho_w * (Wmat @ torch.tanh(h)))





# ----------  optional TorchInductor compile ---------------------------------
TRITON_AVAIL = importlib.util.find_spec("triton") is not None
if DEVICE.type == "cuda" and TRITON_AVAIL and not args.no_compile:
    try:
        f = torch.compile(f)
        print("[INFO] torch.compile → Inductor/Triton enabled")
    except Exception as e:
        print("[WARN] torch.compile unavailable (", str(e).split("\n")[0], ") – falling back to eager.")
else:
    if DEVICE.type == "cuda" and not TRITON_AVAIL and not args.no_compile:
        print("[WARN] Triton not installed – run  pip install triton  to try again.")


#N, g       = 100, 4.
DT, T_SIM   = 1e-1, 1e3
N_LE        = 100
T_ONS       = 1.0

GEN_IC     = torch.Generator(device=DEVICE).manual_seed(1)
GEN_ONS    = torch.Generator(device=DEVICE).manual_seed(1)
GEN_NET    = torch.Generator(device=DEVICE).manual_seed(1)


# ----------  initial state & tangent basis ----------------------------------
q, _ = qr(torch.randn(N, N_LE, generator=GEN_ONS, dtype=DTYPE))
Ls   = torch.zeros(N_LE, dtype=DTYPE)

STEP_ONS = int(T_ONS / DT)
N_STEP   = int(T_SIM / DT)


#h        = (g-1) * torch.randn(N, generator=GEN_IC, dtype=DTYPE, device=DEVICE)
# initial bump-like state at x=0 (helps reach attractor quickly)
Upeak = wr*(1+math.sqrt(1-0.896**2/wr**2))/(4*math.sqrt(math.pi)*k*sigma)
h     = Upeak * torch.exp(-(xs**2)/(4*sigma**2))


with torch.no_grad():
    g_star = (1 - torch.tanh(h).pow(2)).mean().item()
print("g* =", g_star)      # → about 0.4 in your run


# ----------  main integration loop ------------------------------------------
for step in tqdm(range(1, N_STEP + 1), total=N_STEP, desc="Batched JVP run", unit="step"):
    h = f(h)

    # batched forward‑mode JVP  – one kernel for all tangents
    tangents = q.T  # shape (N_LE, N)
    Dq = vmap(lambda v: jvp(f, (h,), (v,))[1])(tangents).T  # removed create_graph kwarg
    q  = Dq

    if step % STEP_ONS == 0:
        q, r = qr(q, mode="reduced")
        Ls += torch.log(torch.abs(torch.diag(r)))

# ----------  spectrum --------------------------------------------------------
t_total = (N_STEP // STEP_ONS) * T_ONS
Lyap    = (Ls / t_total).cpu().numpy()
print("Lyapunov spectrum", Lyap[:])





plt.figure(figsize=(6, 4))

# (i) autograd spectrum
plt.plot(np.arange(N_LE)/N_LE, Lyap,
         "o-", label="TE-network", ms=3)
print(Lyap)

# (ii) Engelken spectrum, if available
if engelken_spec is not None:
    plt.plot(np.arange(N_LE_saved)/N_LE_saved, engelken_spec,
             ".--", label="Engelken (saved)", ms=4)
    print(engelken_spec)

# horizontal zero-line for reference
plt.axhline(0, linestyle=":", color="gray", lw=1)

plt.xlabel(r"$i/N$")
plt.ylabel(r"$\lambda_i \; (1/\tau)$")
plt.title("Lyapunov spectrum – dense vs translation-equivariant")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
