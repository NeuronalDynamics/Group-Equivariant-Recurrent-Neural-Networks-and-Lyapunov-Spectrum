# Continuous Attractor Network – step‑2 analyses
#
# (1) Neutral‑stability test: initialise bump at ±30° and show centre stays put.
# (2) Jacobian eigen‑spectrum: verify one eigenvalue at 0 (λ=1 in discrete‑time)
#     and all others <0 (continuous) / <1 (discrete).
# (3) Full operator circuit with speed populations; show calibrated bump shift
#     under a velocity pulse.

import numpy as np, math, matplotlib.pyplot as plt
from numpy.linalg import eigvals, eig

# ------------------------
# 0. shared parameters
# ------------------------
N        = 180
rho      = 0.5
sigma    = 40.0
k        = 5e-4
tau      = 1.0
dt       = 0.01 * tau
wc       = 2*math.sqrt(2)*(2*math.pi)**0.25 * math.sqrt(k*sigma/rho)
wr       = 1.05 * wc          # 5 % above critical
w_sv     = 1.0
w_vs     = 0.2
Delta_x  = 22.0               # degrees

L   = N / rho
dx  = 1 / rho
xs  = np.linspace(-L/2, L/2, N, endpoint=False)
def circ_dist(a,b): return (a-b + L/2) % L - L/2

# translation‑invariant kernel
W_dist = circ_dist(xs[:,None], xs[None,:])
Wr_mat = wr / (math.sqrt(2*math.pi)*sigma) * np.exp(-(W_dist**2)/(2*sigma**2))

# helper: non‑linearity and its derivatives -------------------------
def G(u):
    """firing‑rate vector and denominator"""
    up = np.maximum(u, 0.0)
    num = up**2
    denom = 1.0 + k*rho*np.sum(num)*dx
    return num/denom, up, denom

def dG_du(u, up, denom):
    """Jacobian of r wrt u (N×N)"""
    # precompute 2*up/denom
    diag_part = 2*up/denom
    # common factor for denominator derivative
    common = 2*k*rho*dx / (denom**2)
    # build full Jacobian
    J = np.zeros((N,N))
    for i in range(N):
        # ith row, jth col
        for j in range(N):
            if up[i] == 0:                         # below threshold
                continue
            if j == i:
                J[i,j] = diag_part[i] - up[i]**2 * common * up[j]
            else:
                J[i,j] = - up[i]**2 * common * up[j]
    return J

# analytical bump height (Eq. S12)
def bump_height(wr):
    return wr*(1+math.sqrt(1-wc**2/wr**2))/(4*math.sqrt(math.pi)*k*sigma)

U_peak = bump_height(wr)

# ================================================================
# 1. Neutral‑stability test (±30° initial shift)
# ================================================================
def simulate_CAN(u0, steps=3000):
    Rcent = []
    u = u0.copy()
    for _ in range(steps):
        r, up, denom = G(u)
        # population‑vector estimate of bump centre
        vect = np.sum(r * np.exp(1j*np.deg2rad(xs))) / np.sum(r)
        centre = np.rad2deg(np.angle(vect))
        Rcent.append(centre)
        # Euler step
        conv = Wr_mat @ r * dx
        u += dt/tau * (-u + rho*conv)
    return np.asarray(Rcent)

# shift +30°
u_init_plus  = U_peak * np.exp(-(circ_dist(xs,  30.0))**2/(4*sigma**2))
# shift −30°
u_init_minus = U_peak * np.exp(-(circ_dist(xs, -30.0))**2/(4*sigma**2))

cent_plus  = simulate_CAN(u_init_plus)
cent_minus = simulate_CAN(u_init_minus)

t_axis = np.arange(cent_plus.size)*dt

plt.figure(figsize=(5,3))
plt.plot(t_axis, cent_plus,  label="+30° init")
plt.plot(t_axis, cent_minus, label="−30° init")
plt.xlabel("time (τ)")
plt.ylabel("bump centre (°)")
plt.title("Neutral stability along ring")
plt.legend(); plt.tight_layout()
plt.show()

# ================================================================
# 2. Jacobian eigen‑spectrum around centred bump
# ================================================================
u0 = U_peak * np.exp(-(xs)**2/(4*sigma**2))
r0, up0, denom0 = G(u0)
Dr = dG_du(u0, up0, denom0)
Wr_dx = Wr_mat * dx           # ← multiply by lattice spacing
Jac   = (-np.eye(N) + rho * Wr_dx @ Dr) / tau   # continuous‑time Jacobian
eigvals_cont = np.sort(np.real(eigvals(Jac)))[::-1]   # descending
print(eigvals_cont[:])

K = rho * Wr_dx @ Dr
eig_K = np.sort(np.real(np.linalg.eigvals(K)))[::-1]
print(eig_K[:])

plt.figure(figsize=(4,3))
plt.plot(eig_K[:10], '.')
plt.yscale('log')
plt.axhline(0, color='k', lw=0.5)
plt.xlabel("eigen‑index")
plt.ylabel("λ (continuous‑time)")
plt.title("Jacobian spectrum")
plt.tight_layout()
plt.show()

# ================================================================
# 3. Full operator circuit with speed populations
# ================================================================
# shifted kernels W_+ and W_-
shift = Delta_x
W_plus  = wr / (math.sqrt(2*math.pi)*sigma) * np.exp(-(circ_dist(xs[:,None], xs[None,:]-shift))**2/(2*sigma**2))
W_minus = wr / (math.sqrt(2*math.pi)*sigma) * np.exp(-(circ_dist(xs[:,None], xs[None,:]+shift))**2/(2*sigma**2))

# stationary bump used for calibration
R0 = r0
# compute calibration factor C = sqrt(2)ρ w_sv w_vs R Δx / (τ U)
R_peak = R0.max()
C = math.sqrt(2)*rho*w_sv*w_vs*R_peak*Delta_x / (tau*U_peak)

# velocity protocol: 0 → +5° τ⁻¹ for 15 τ, then back to 0
T_total = 30.0
steps   = int(T_total/dt)
v_track = np.zeros(steps)
v_track[int(5/dt):int(20/dt)] = 5.0    # pulse from 5τ to 20τ

# state variables
u_s = u0.copy()
u_plus  = np.zeros(N)
u_minus = np.zeros(N)
centers = []

for t in range(steps):
    v = v_track[t]
    r_s, up_s, denom_s = G(u_s)
    r_plus  = np.maximum((w_vs*r_s + (10+v))*u_plus, 0)  # Eq. 19b (simplified)
    r_minus = np.maximum((w_vs*r_s + (10-v))*u_minus, 0)
    
    # update currents
    conv_s = rho*(Wr_mat @ r_s + W_plus @ r_plus + W_minus @ r_minus)*dx
    u_s   += dt/tau * (-u_s + conv_s)
    u_plus  += dt/tau * (-u_plus  + w_vs*r_s)
    u_minus += dt/tau * (-u_minus + w_vs*r_s)
    
    vect = np.sum(r_s * np.exp(1j*np.deg2rad(xs))) / np.sum(r_s)
    centers.append(np.rad2deg(np.angle(vect)))

centers = np.unwrap(np.deg2rad(centers))
centers = np.rad2deg(centers)  # unwrap to avoid discontinuities
t_axis2 = np.arange(steps)*dt

plt.figure(figsize=(5,3))
plt.plot(t_axis2, centers)
plt.plot(t_axis2, np.cumsum(v_track)*dt, '--', label="∫v(t)dt")
plt.xlabel("time (τ)")
plt.ylabel("bump centre (°)")
plt.title(f"Path‑integration with speed populations (C={C:.2f})")
plt.legend()
plt.tight_layout()
plt.show()