# Continuous Attractor Network – step‑1 analyses
#
# (1) plot the translation-invariant connection kernel (Fig. 2B);
# 
# (2) evolve the CAN for 30 τ from a centred Gaussian bump (Fig. 2C);
#
# (3) compute the time-averaged bump profile (Fig. 2D).

import numpy as np, math, matplotlib.pyplot as plt

# ------------------------
# 1. parameters (Table 1)
# ------------------------
N        = 180          # neurons
rho      = 0.5          # cells per degree
sigma    = 40.0         # deg
k        = 5e-4
tau      = 1.0
dt       = 0.01 * tau
wc       = 2*math.sqrt(2)*(2*math.pi)**0.25 * math.sqrt(k*sigma/rho)   # Eq. S11
wr       = 1.05 * wc    # 5 % above threshold → stable bump

print("wc = ", wc)
print("wr = ", wr)
# ------------------------
# 2. geometry
# ------------------------
L   = N / rho                            # 360 deg domain
dx  = 1 / rho                            # 2 deg lattice spacing
xs  = np.linspace(-L/2, L/2, N, endpoint=False)

print("L  = ", L)
print("dx = ", dx)
#print("xs = ", xs)

def circ_dist(a,b):                      # minimal ring distance
    d = a - b
    return (d + L/2) % L - L/2

# recurrent kernel (Eq. 9)
dist   = circ_dist(xs[:,None], xs[None,:])
Wr_mat = wr / (math.sqrt(2*math.pi)*sigma) * np.exp(-(dist**2)/(2*sigma**2))

# Figure 2B ---------------------------------------------------------
disp = np.linspace(-180, 180, 361)
Wr_profile = wr / (math.sqrt(2*math.pi)*sigma) * np.exp(-(disp**2)/(2*sigma**2))
plt.figure(figsize=(4.5,3))
plt.plot(disp, Wr_profile, lw=2, color='tab:orange')
plt.xlabel("Tuning disparity $x-x'$ (°)")
plt.ylabel("$W_r$")
plt.title("Translation-invariant weight (Fig 2B)")
plt.tight_layout(); plt.show()

# ------------------------
# 3. simulation (Eq. 8)
# ------------------------
def G(u):
    up = np.maximum(u,0)
    num = up**2
    den = 1 + k*rho*np.sum(num)*dx
    return num/den

U_peak = wr*(1+math.sqrt(1-wc**2/wr**2))/(4*math.sqrt(math.pi)*k*sigma)   # Eq. S12
u = U_peak * np.exp(-(xs)**2/(4*sigma**2))                                 # centred bump
print("U_peak = ", U_peak)
#print("u      = ", u)

steps = int(30/dt)                # 30 τ as in the paper
R_t   = np.zeros((steps, N))

for t in range(steps):
    r  = G(u)
    R_t[t] = r                    # store for Fig 2C
    conv = Wr_mat @ r * dx
    u   += dt/tau * (-u + rho*conv)

# Figure 2C ----------------------------------------------------------
plt.figure(figsize=(6,4))
plt.imshow(R_t, aspect='auto', origin='lower',
           extent=[xs[0], xs[-1], 0, steps*dt],
           cmap='viridis')
plt.xlabel("Neuron preferred stimulus $x$ (°)")
plt.ylabel("time (τ)")
plt.colorbar(label="firing rate")
plt.title("Spatiotemporal firing (Fig 2C)")
plt.tight_layout(); plt.show()


# Figure 2C – time on x-axis, stimulus on y-axis
plt.figure(figsize=(6, 4))

# 1) transpose R_t  ⟹  shape: (space, time) → (time, space)
# 2) swap the two (min,max) pairs inside extent
plt.imshow(R_t.T,               # <-- transpose
           aspect='auto',
           origin='lower',
           extent=[0, steps*dt, xs[0], xs[-1]],   # <-- swapped order
           cmap='viridis')

plt.xlabel("time (τ)")
plt.ylabel("Neuron preferred stimulus $x$ (°)")
plt.colorbar(label="firing rate")
plt.title("Spatiotemporal firing (Fig 2C, axes swapped)")
plt.tight_layout()
plt.show()

# Figure 2D ----------------------------------------------------------
avg_r = R_t.mean(axis=0)
#print("avg_r = ", avg_r)
plt.figure(figsize=(5,3))
plt.plot(xs, avg_r, color='tab:orange', lw=3)
plt.xlabel("Neuron preferred stimulus $x$ (°)")
plt.ylabel("mean firing rate")
plt.title("Time-avg bump profile (Fig 2D)")
plt.tight_layout(); plt.show()




