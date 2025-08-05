import numpy as np, math, matplotlib.pyplot as plt

# ----------  CAN & weight parameters (same table as before) ----------
N, rho, sigma = 180, 0.5, 40.0
k, tau, dt    = 5e-4, 1.0, 0.01
wc            = 2*math.sqrt(2)*(2*math.pi)**.25 * math.sqrt(k*sigma/rho)
wr            = 1.05*wc                       # 5 % above critical
L, dx         = N/rho, 1/rho
xs            = np.linspace(-L/2, L/2, N, endpoint=False)
circ          = lambda a,b: (a-b+L/2)%L - L/2

# weight kernel W_r(x-x')
W0   = wr/(math.sqrt(2*math.pi)*sigma) * \
       np.exp(-circ(xs[:,None], xs[None,:])**2/(2*sigma**2))
dWdx = -(circ(xs[:,None], xs[None,:]))/sigma**2 * W0        # ∂/∂x W_r
W0  *= dx;   dWdx *= dx                                     # approximate integral

def G(u):                      # divisive normalisation (Eq.​8 b)
    up  = np.clip(u, 0, None)
    num = up**2
    return num / (1 + k*rho*np.sum(num))

def bump_height(wr):           # analytic peak height (Eq.​S12)
    return wr*(1+math.sqrt(1-wc**2/wr**2))/(4*math.sqrt(math.pi)*k*sigma)
U0 = bump_height(wr)

# ============ 3 a  —  derivative of the kernel =============
disp = np.linspace(-180, 180, 361)
W_prof  = wr/(math.sqrt(2*math.pi)*sigma)*np.exp(-(disp**2)/(2*sigma**2))
dW_prof = -(disp)/sigma**2 * W_prof
plt.figure(figsize=(5,3))
plt.plot(disp, dW_prof, lw=2)
plt.axhline(0, color='k', lw=.5)
plt.xlabel("Δx (°)"); plt.ylabel("∂x W_r")
plt.title("Fig 3a — derivative of weight kernel")
plt.tight_layout()
plt.show()
# ============ 3 b & 3 c — drift at v = 5°/τ =============
v = 5.0                       # commanded speed
T = 10.0;  steps = int(T/dt)
u      = U0 * np.exp(-circ(xs, 0)**2/(4*sigma**2))
r_ts   = np.zeros((steps, N))
centre = []

for t in range(steps):
    r_ts[t] = G(u)
    conv    = rho * (W0 - tau*v*dWdx) @ r_ts[t]
    u      += dt/tau * (-u + conv)
    vec     = np.sum(r_ts[t]*np.exp(1j*np.deg2rad(xs))) / np.sum(r_ts[t])
    centre.append(np.rad2deg(np.angle(vec)))

centre = np.unwrap(np.deg2rad(centre)); centre = np.rad2deg(centre)
t_vec  = np.arange(steps)*dt

# 3 b  — heat-map
plt.figure(figsize=(6,3))
plt.imshow(r_ts.T, aspect='auto', origin='lower',
           extent=[0, T, xs[0], xs[-1]], cmap='viridis')
plt.xlabel("time (τ)"); plt.ylabel("preferred stimulus (°)")
plt.title("Fig 3b — bump drifting at 5°/τ")
plt.tight_layout()
plt.show()
# 3 c  — centre vs time
plt.figure(figsize=(5,3))
plt.plot(t_vec, centre, label="measured")
plt.plot(t_vec, v*t_vec, '--', label="theory")
plt.xlabel("time (τ)"); plt.ylabel("bump centre (°)")
plt.title("Fig 3c — centre tracks v·t"); plt.legend(); plt.tight_layout()
plt.show()
# ============ 3 d — linearity of speed =============
v_tests = np.array([-6, -4, -2, 0, 2, 4, 6])
slopes  = []

for v_cmd in v_tests:
    u = U0 * np.exp(-circ(xs, 0)**2/(4*sigma**2))
    c = []
    for _ in range(steps):
        r   = G(u)
        conv = rho * (W0 - tau*v_cmd*dWdx) @ r
        u  += dt/tau * (-u + conv)
        vec = np.sum(r*np.exp(1j*np.deg2rad(xs))) / np.sum(r)
        c.append(np.rad2deg(np.angle(vec)))
    c = np.unwrap(np.deg2rad(c));  c = np.rad2deg(c)
    slopes.append(np.polyfit(t_vec, c, 1)[0])

plt.figure(figsize=(4,3))
plt.plot(v_tests, slopes, 'o', label="measured")
plt.plot(v_tests, v_tests, '--', label="ideal")
plt.xlabel("commanded v (°/τ)"); plt.ylabel("fitted slope (°/τ)")
plt.title("Fig 3d — linear speed-transfer"); plt.legend(); plt.tight_layout()
plt.show()