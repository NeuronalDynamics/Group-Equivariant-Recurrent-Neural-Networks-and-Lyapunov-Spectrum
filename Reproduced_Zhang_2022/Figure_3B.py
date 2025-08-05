import numpy as np, math, matplotlib.pyplot as plt

# Parameters (Table 1)
N, rho, sigma = 180, 0.5, 40.0
k, tau, dt = 5e-4, 1.0, 0.01
wc = 2*math.sqrt(2)*(2*math.pi)**0.25*math.sqrt(k*sigma/rho)
wr = 1.05*wc
v = 5.0                          # deg per tau (matches Fig 3B)
T = 10.0                         # simulate 10 tau
steps = int(T/dt)

# Grid and kernels ---------------------------------------------------
L, dx = N/rho, 1/rho
xs = np.linspace(-L/2, L/2, N, endpoint=False)
circ = lambda a,b: (a-b+L/2)%L - L/2
W0 = wr/(math.sqrt(2*math.pi)*sigma)*np.exp(-(circ(xs[:,None],xs[None,:])**2)/(2*sigma**2))*dx
dWdx = -(circ(xs[:,None],xs[None,:]))/sigma**2 * W0       # derivative already times dx

# Divisive-normalisation --------------------------------------------
def G(u):
    up = np.clip(u, 0, None)
    num = up**2
    return num / (1 + k*rho*np.sum(num))

def bump_height(wr):
    return wr*(1+math.sqrt(1-wc**2/wr**2))/(4*math.sqrt(math.pi)*k*sigma)

# Initial bump centred at x=0
u = bump_height(wr) * np.exp(-(xs**2)/(4*sigma**2))

# Simulation loop ----------------------------------------------------
R = np.zeros((steps, N))
centre = np.zeros(steps)

for t in range(steps):
    R[t] = G(u)
    drive = rho * (W0 - tau*v*dWdx) @ R[t]
    u += dt/tau * (-u + drive)

    # bump centre read‑out
    vec = np.sum(R[t]*np.exp(1j*np.deg2rad(xs))) / np.sum(R[t])
    centre[t] = np.rad2deg(np.angle(vec))

# Post‑processing for nicer plot -------------------------------------
centre_unwrap = np.rad2deg(np.unwrap(np.deg2rad(centre)))
t_axis = np.arange(steps)*dt

# Figure -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.5,3))
im = ax.imshow(R.T, aspect='auto', origin='lower',
               extent=[0, T, xs[0], xs[-1]],
               cmap='viridis', vmin=0, vmax=R.max())
cb = fig.colorbar(im, ax=ax, label='firing rate')

ax.plot(t_axis, centre_unwrap, color='w', lw=1.5, label='centre')
ax.plot(t_axis, v*t_axis, '--', color='b', lw=1, label=r'$s(t)=vt$')

ax.set_xlabel('time (τ)')
ax.set_ylabel('preferred stimulus x (°)')
ax.set_title('Numerical reproduction of Fig 3B – bump drift at 5° / τ')
ax.legend(loc='upper left')
plt.tight_layout()
plt.show()