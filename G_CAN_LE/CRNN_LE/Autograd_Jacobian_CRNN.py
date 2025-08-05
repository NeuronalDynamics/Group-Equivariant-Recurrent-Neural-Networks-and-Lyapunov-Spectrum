# ------------------------------------------------------------
# 0.  (OPTIONAL)  Load Engelken–style spectrum saved earlier
# ------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd.functional import jvp
from torch.linalg import qr
from tqdm.auto import tqdm
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# ----------  path to Engelken spectrum  ----------
spec_path = "Lyapunov_spectrum.npy"        # ← adjust if you used another name
engelken_spec = None
if os.path.exists(spec_path):
    engelken_spec = np.load(spec_path)
    nLE_saved     = engelken_spec.size
else:
    print(f"[INFO]   '{spec_path}' not found – will plot only autograd spectrum.")

# ------------------------------------------------------------
# 1.  Run the autograd/JVP version to produce `Lspec`
# ------------------------------------------------------------
device     = "cpu"
dtype      = torch.float64
N, g       = 100, 4.
dt, tSim   = 1e-1, 1e3
nLE        = 100
tONS       = 1.0
rng_ic     = torch.Generator().manual_seed(1)
rng_net    = torch.Generator().manual_seed(1)
rng_ons    = torch.Generator().manual_seed(1)

J = g*torch.randn((N, N), generator=rng_net, dtype=dtype, device=device) / np.sqrt(N)
J.fill_diagonal_(0.)
def f(h):                     # Euler step
    return h*(1-dt) + dt * J @ torch.tanh(h)

q,_ = qr(torch.randn(N, nLE, generator=rng_ons, dtype=dtype, device=device))
Ls        = torch.zeros(nLE, dtype=dtype)
nstepONS  = int(tONS / dt)
nStep     = int(tSim / dt)
h         = (g-1) * torch.randn(N, generator=rng_ic, dtype=dtype, device=device)
nStepTransient      = int(100/dt)

for n in tqdm(range(1, nStep+1), total=nStep, desc="Autograd run", unit="step"):
    h = f(h)

    def net_step(x):          # wrapper for jvp
        return f(x)

    Dq = torch.empty_like(q)
    for k in range(nLE):
        _, Jv = jvp(net_step, (h,), (q[:, k],), create_graph=False)
        Dq[:, k] = Jv

    q = Dq                      # just propagate for this step
    if n % nstepONS == 0:       # ← QR only every 1 τ
        q, r = qr(q, mode="reduced")          # ← move QR inside the block
        Ls += torch.log(torch.abs(torch.diag(r)))

t_total = (nStep // nstepONS) * tONS
Lspec   = (Ls / t_total).cpu().numpy()

# ------------------------------------------------------------
# 2.  Over-plot both spectra
# ------------------------------------------------------------
plt.figure(figsize=(6, 4))

# (i) autograd spectrum
plt.plot(np.arange(nLE)/nLE, Lspec,
         "o-", label="Autograd / JVP", ms=3)
print(Lspec)

# (ii) Engelken spectrum, if available
if engelken_spec is not None:
    plt.plot(np.arange(nLE_saved)/nLE_saved, engelken_spec,
             ".--", label="Engelken (saved)", ms=4)
    print(engelken_spec)

# horizontal zero-line for reference
plt.axhline(0, linestyle=":", color="gray", lw=1)

plt.xlabel(r"$i/N$")
plt.ylabel(r"$\lambda_i \; (1/\tau)$")
plt.title("Lyapunov spectra comparison")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
