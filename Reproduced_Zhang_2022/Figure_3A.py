import numpy as np, math, matplotlib.pyplot as plt

# real units (no normalisation)
N, rho, sigma = 180, 0.5, 40.0
k, tau, dt    = 5e-4, 1.0, 0.01
wc            = 2*math.sqrt(2)*(2*math.pi)**.25 * math.sqrt(k*sigma/rho)
wr            = 1.05*wc                       # 5 % above critical
v_demo = 5.0/tau          # 5°/τ as in the paper
Δx = np.linspace(-180, 180, 361)
W  = wr/(math.sqrt(2*math.pi)*sigma) * np.exp(-(Δx**2)/(2*sigma**2))
dW = -tau*v_demo * ( -Δx / sigma**2 ) * W   #  -τv ∂x W_r

plt.figure(figsize=(5.5,3))
plt.plot(Δx, W,  lw=3, color='tab:orange', label=r'$W_r(\Delta x)$')
plt.plot(Δx, dW, lw=2, color='tab:red',
         label=fr'$-\tau v\,\partial_x W_r\;$ ($v={v_demo}°/\tau$)')
plt.axhline(0, color='k', lw=.5)
plt.xlabel(r'$\Delta x$ (deg)')
plt.ylabel('weight (same units)')
plt.title('Stationary kernel vs. translation drive (true scale)')
plt.legend(); plt.tight_layout()
plt.show()