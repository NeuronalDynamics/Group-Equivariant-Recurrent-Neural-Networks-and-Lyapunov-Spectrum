import numpy as np, math, matplotlib.pyplot as plt

# --- parameters (identical to Table 1) ----------------------------
N, ρ, σ   = 180, 0.5, 40.0                   # neurons, density, tuning
k, τ, dt  = 5e-4, 1.0, 0.01                  # inhibition, time-constant
w_c       = 2*math.sqrt(2)*(2*math.pi)**.25*math.sqrt(k*σ/ρ)
w_r       = 1.05*w_c                         # 5 % above threshold
L, dx     = N/ρ, 1/ρ
xs        = np.linspace(-L/2, L/2, N, endpoint=False)
circ      = lambda a,b: (a-b+L/2)%L - L/2

# recurrent kernel  W_r(x-x')
W = w_r/(math.sqrt(2*math.pi)*σ) * np.exp(-(circ(xs[:,None],xs[None,:])**2)/(2*σ**2))

def G(u):
    up  = np.clip(u, 0, None)
    num = up**2
    den = 1 + k*ρ*np.sum(num)*dx
    return num / den

def bump_height(wr):
    return wr*(1+math.sqrt(1-w_c**2/wr**2)) / (4*math.sqrt(math.pi)*k*σ)

U = bump_height(w_r)

# --- simulate full rate matrix ------------------------------------
def simulate_rates(start_deg, T=30):
    u = U * np.exp(-(circ(xs, start_deg)**2)/(4*σ**2))
    steps = int(T/dt)
    rates = np.zeros((steps, N))
    for t in range(steps):
        r        = G(u)
        rates[t] = r
        u       += dt/τ * (-u + ρ * W @ r * dx)
    return rates

R_plus  = simulate_rates(+30)
R_minus = simulate_rates(-30)

# --- plotting (Fig 2C style) --------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True,
                         gridspec_kw={'hspace': 0.25})
vmax = max(R_plus.max(), R_minus.max())

for ax, R, title in zip(
        axes,
        (R_plus, R_minus),
        ('Bump initialised at +30°', 'Bump initialised at –30°')):
    im = ax.imshow(R, origin='lower', aspect='auto',
                   extent=[xs[0], xs[-1], 0, R.shape[0]*dt],
                   cmap='inferno', vmin=0, vmax=vmax)
    ax.set_ylabel('time (τ)')
    ax.set_title(title, fontsize=12)

axes[-1].set_xlabel('preferred stimulus $x$ (°)')
fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7,
             label='firing rate')
fig.suptitle('Neutral-stability test — spatiotemporal activity', fontsize=14)
#plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()