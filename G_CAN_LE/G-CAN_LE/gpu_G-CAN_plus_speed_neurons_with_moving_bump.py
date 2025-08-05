#!/usr/bin/env python
# ----------------------------------------------------------------------
#  te_can.py – Translation-equivariant continuous-attractor network
#              (NeurIPS-2022 “translation-equivariant representation …’’)
# ----------------------------------------------------------------------
import argparse, importlib.util, os, math
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# ─── numerics & plotting ──────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.func import vmap, jvp
from torch.linalg import qr
from tqdm.auto import tqdm

# ─── CLI ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--no-compile", action="store_true")
parser.add_argument("--speed", type=float, default=0.0,
                    help="translation speed v  (deg / τ): 0 → static spectrum")
args = parser.parse_args()

# ─── device & dtype ───────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(DEVICE)
DTYPE  = torch.float32
if DEVICE.type == "cuda":
    print("Running on", torch.cuda.get_device_name(0))

# ─── optional Engelken spectrum ───────────────────────────────────────
spec_path = "Lyapunov_spectrum.npy"
engelken_spec, N_LE_saved = (None, None)
if os.path.exists(spec_path):
    engelken_spec = np.load(spec_path)
    N_LE_saved    = engelken_spec.size
else:
    print(f"[INFO] '{spec_path}' not found – autograd spectrum only.")

# ══════════════════════════════════════════════════════════════════════
# 0.  Parameters  (matches paper & supplement)
# ══════════════════════════════════════════════════════════════════════
sigma      = 40.0
rho_deg    = 0.5
N          = 180
dx         = 1 / rho_deg
wr         = 1.05 * 0.896
k          = 5e-4
rho_w      = rho_deg
tau        = 1.0

Delta      = 22.0
g_v        = 10.0
w_vs       = 0.2

# ─── ring coordinates & helpers ───────────────────────────────────────
xs = torch.linspace(-180, 180-dx, N, device=DEVICE)

def circ(a, b):
    return (a - b + 180.) % 360. - 180.

# ----------------------------------------------------------------------
# 1.  Convolutional kernels
# ----------------------------------------------------------------------
Δx   = circ(xs[:, None], xs[None, :])
W_r = (wr / (math.sqrt(2*math.pi)*sigma) *
       torch.exp(-(Δx**2) / (2*sigma**2))) * dx
W_r = W_r.to(DTYPE)

def shift_kernel(d):
    return torch.exp(-(d**2) / (2*sigma**2))

# ----------------------------------------------------------------------
# 2.  Stimulus-ring rate non-linearity
# ----------------------------------------------------------------------
def rates_s(u_s):
    up  = torch.clamp(u_s, 0.)
    num = up**2
    return num / (1 + k*rho_w*torch.sum(num))

# ----------------------------------------------------------------------
# 3.  Stationary bump initial state
# ----------------------------------------------------------------------
U_peak = wr * (1 + math.sqrt(1 - 0.896**2 / wr**2)) / (4*math.sqrt(math.pi)*k*sigma)
u_s0   = U_peak * torch.exp(-(xs**2) / (4*sigma**2))
u_p0   = torch.zeros_like(u_s0)
u_m0   = torch.zeros_like(u_s0)
h_init = torch.cat((u_s0, u_p0, u_m0))

# ----------------------------------------------------------------------
# 4.  Calibrate  w_sv  via Eq. 22
# ----------------------------------------------------------------------
with torch.no_grad():
    R_peak = rates_s(u_s0).max()
    coeff  = math.sqrt(2) * rho_w * w_vs * R_peak * Delta
    w_sv   = (tau * U_peak) / coeff

# ----------------------------------------------------------------------
# 5.  Speed-projection kernels   *** SIGN FIX ↓ ***
# ----------------------------------------------------------------------
W_plus  = w_sv * shift_kernel(circ(xs[:, None], xs[None, :] + Delta)) * dx
W_minus = w_sv * shift_kernel(circ(xs[:, None], xs[None, :] - Delta)) * dx
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# 6.  Network dynamics  (Eqs. 19a-b)
# ----------------------------------------------------------------------
def step(h, v_now):
    u_s, u_p, u_m = h[:N], h[N:2*N], h[2*N:]

    r_s = rates_s(u_s)
    r_p = torch.clamp((g_v + v_now) * u_p, 0.)
    r_m = torch.clamp((g_v - v_now) * u_m, 0.)

    du_s = (-u_s + rho_w*(W_r @ r_s + W_plus @ r_p + W_minus @ r_m)) / tau
    du_p = (-u_p + w_vs * r_s) / tau
    du_m = (-u_m + w_vs * r_s) / tau
    return h + DT * torch.cat((du_s, du_p, du_m))

f_static = lambda h: step(h, v_now=0.0)

# ─── optional TorchInductor compile ───────────────────────────────────
TRITON_AVAIL = importlib.util.find_spec("triton") is not None
if DEVICE.type == "cuda" and TRITON_AVAIL and not args.no_compile:
    try:
        f_static = torch.compile(f_static)
        step     = torch.compile(step)
        print("[INFO] torch.compile → Inductor/Triton enabled")
    except Exception as e:
        print("[WARN] torch.compile unavailable (", str(e).split('\n')[0], ") – eager mode.")

# ══════════════════════════════════════════════════════════════════════
# 7.  Simulation settings
# ══════════════════════════════════════════════════════════════════════
DT        = 1e-1
T_SIM     = 1e3
N_LE      = 100
T_ONS     = 1.0
GEN_ONS   = torch.Generator(device=DEVICE).manual_seed(1)

# ----------------------------------------------------------------------
# 7A.  Lyapunov-spectrum mode
# ----------------------------------------------------------------------
if abs(args.speed) < 1e-12:
    print("=== Lyapunov-spectrum mode (static bump) ===")
    q, _ = qr(torch.randn(3*N, N_LE, generator=GEN_ONS, dtype=DTYPE))
    Ls   = torch.zeros(N_LE, dtype=DTYPE)

    STEP_ONS = int(T_ONS / DT)
    N_STEP   = int(T_SIM / DT)
    h        = h_init.clone()

    for step_idx in tqdm(range(1, N_STEP+1), total=N_STEP,
                         desc="Batched JVP run", unit="step"):
        h = f_static(h)
        tangents = q.T
        Dq = vmap(lambda v: jvp(f_static, (h,), (v,))[1])(tangents).T
        q  = Dq
        if step_idx % STEP_ONS == 0:
            q, r = qr(q, mode="reduced")
            Ls += torch.log(torch.abs(torch.diag(r)))

    t_total = (N_STEP // STEP_ONS) * T_ONS
    Lyap    = (Ls / t_total).cpu().numpy()

    plt.figure(figsize=(6, 4))
    plt.plot(np.arange(N_LE)/N_LE, Lyap, "o-", ms=3, label="TE-network")
    if engelken_spec is not None:
        plt.plot(np.arange(N_LE_saved)/N_LE_saved, engelken_spec,
                 ".--", ms=4, label="Engelken (saved)")
    plt.axhline(0, linestyle=":", color="gray", lw=1)
    plt.xlabel(r"$i/N$");  plt.ylabel(r"$\lambda_i \; (1/\tau)$")
    plt.title("Lyapunov spectrum – dense vs TE-network")
    plt.grid(True, alpha=0.3);  plt.legend();  plt.tight_layout();  plt.show()

# ----------------------------------------------------------------------
# 7B.  Moving-bump mode
# ----------------------------------------------------------------------
else:
    v_cmd = torch.tensor(args.speed, dtype=DTYPE, device=DEVICE)
    print(f"=== Moving-bump demo  (v = {v_cmd.item():.3f} deg / τ) ===")

    # continuous (un-wrapped) population-vector decoder
    def pop_vector_unwrapped(r, state):
        z   = torch.sum(r * torch.exp(1j*xs*math.pi/180))
        phi = math.atan2(z.imag, z.real)            # (-π, π]
        if state:                                   # subsequent sample
            delta = ((phi - state['prev'] + math.pi) % (2*math.pi)) - math.pi
            state['cum'] += delta
        else:                                       # first sample
            state['cum'] = phi
        state['prev'] = phi
        return math.degrees(state['cum']), state

    N_STEP = int(T_SIM / DT)
    h      = h_init.clone()
    state  = {}          # decoder memory
    traj   = []          # un-wrapped angles

    # ── Lyapunov-spectrum bookkeeping ───────────────────────────────
    q, _ = qr(torch.randn(3*N, N_LE, generator=GEN_ONS, dtype=DTYPE))
    Ls   = torch.zeros(N_LE, dtype=DTYPE)
    STEP_ONS = int(T_ONS / DT)

    # fixed-speed step for JVP
    step_v = torch.compile(lambda x: step(x, v_cmd)) \
             if (DEVICE.type=="cuda" and not args.no_compile and TRITON_AVAIL) \
             else (lambda x: step(x, v_cmd))

    for step_idx in tqdm(range(1, N_STEP+1), desc="Moving bump", unit="step"):
        h = step_v(h)

        # -------- Lyapunov JVP (batched) ----------------------------
        tangents = q.T
        Dq = vmap(lambda v: jvp(step_v, (h,), (v,))[1])(tangents).T
        q  = Dq
        if step_idx % STEP_ONS == 0:
            q, r = qr(q, mode="reduced")
            Ls += torch.log(torch.abs(torch.diag(r)))

        # -------- bump decoder --------------------------------------
        if step_idx % 10 == 0:
            angle, state = pop_vector_unwrapped(rates_s(h[:N]), state)
            traj.append(angle)

    # ─── trajectory plot ──────────────────────────────────────────────
    plt.figure(figsize=(6, 3))
    t = np.arange(len(traj))*DT*10
    plt.plot(t, traj, label="decoded bump")
    plt.plot(t, traj[0] + v_cmd.item()*t, "--", label="theory  s₀ + v t")
    plt.xlabel("time (τ)");  plt.ylabel("stimulus  s  (deg)")
    plt.legend();  plt.tight_layout();  plt.show()

    # ─── Lyapunov spectrum plot ─────────────────────────────────────
    t_total = (N_STEP // STEP_ONS) * T_ONS
    Lyap_mv = (Ls / t_total).cpu().numpy()

    plt.figure(figsize=(6, 4))
    plt.plot(np.arange(N_LE)/N_LE, Lyap_mv, "o-", ms=3,
             label=f"TE-network  (v = {v_cmd.item():g})")
    plt.axhline(0, linestyle=":", color="gray", lw=1)
    plt.xlabel(r"$i/N$");  plt.ylabel(r"$\lambda_i \; (1/\tau)$")
    plt.title("Lyapunov spectrum – moving bump")
    plt.grid(True, alpha=0.3);  plt.legend();  plt.tight_layout();  plt.show()