import numpy as np, math, matplotlib.pyplot as plt

# Parameters
N, rho, sigma = 180, 0.5, 40.0
k, tau, dt = 5e-4, 1.0, 0.01
wc = 2*math.sqrt(2)*(2*math.pi)**0.25*math.sqrt(k*sigma/rho)
wr = 1.05*wc
w_sv = 1.0
w_vs = 0.2
Delta = 22.0
g_v = 10.0

# grids
L, dx = N/rho, 1/rho
xs = np.linspace(-L/2, L/2, N, endpoint=False)
circ = lambda a,b: (a-b+L/2)%L - L/2

# kernels
W_r = wr/(math.sqrt(2*math.pi)*sigma)*np.exp(-(circ(xs[:,None], xs[None,:])**2)/(2*sigma**2))*dx
W_plus = w_sv/(math.sqrt(2*math.pi)*sigma)*np.exp(-(circ(xs[:,None], xs[None,:]-Delta)**2)/(2*sigma**2))*dx
W_minus= w_sv/(math.sqrt(2*math.pi)*sigma)*np.exp(-(circ(xs[:,None], xs[None,:]+Delta)**2)/(2*sigma**2))*dx

# figure 4C: plot kernels
disp = np.linspace(-180,180,361)
W_r_disp = wr/(math.sqrt(2*math.pi)*sigma)*np.exp(-(disp**2)/(2*sigma**2))
W_p_disp = w_sv/(math.sqrt(2*math.pi)*sigma)*np.exp(-((disp-Delta)**2)/(2*sigma**2))
W_m_disp = w_sv/(math.sqrt(2*math.pi)*sigma)*np.exp(-((disp+Delta)**2)/(2*sigma**2))

plt.figure(figsize=(5,3))
plt.plot(disp, W_r_disp, lw=3, label=r'$W_r$')
plt.plot(disp, W_p_disp, '--', lw=2, label=r'$W_{+}$ (shift +Δx)')
plt.plot(disp, W_m_disp, ':', lw=2, label=r'$W_{-}$ (shift −Δx)')
plt.axhline(0,color='k',lw=.5)
plt.xlabel(r'$\Delta x$ (deg)')
plt.ylabel('weight')
plt.title('Synaptic kernels (Fig 4C)')
plt.legend(); plt.tight_layout()
plt.show()