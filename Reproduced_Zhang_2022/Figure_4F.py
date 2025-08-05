import numpy as np, math, matplotlib.pyplot as plt

# parameters same as before
N, rho, sigma = 180, 0.5, 40.0
k, tau, dt = 5e-4, 1.0, 0.01
wc = 2*math.sqrt(2)*(2*math.pi)**0.25*math.sqrt(k*sigma/rho)
wr = 1.05*wc
w_sv=1.0; w_vs=0.2; Delta=22.0; g_v=10.0
L, dx = N/rho, 1/rho
xs = np.linspace(-L/2, L/2, N, endpoint=False)
circ = lambda a,b: (a-b+L/2)%L - L/2
G0 = lambda delta: (1/(math.sqrt(2*math.pi)*sigma))*np.exp(-(delta**2)/(2*sigma**2))*dx
W_r = wr*G0(circ(xs[:,None], xs[None,:]))
W_plus = w_sv*G0(circ(xs[:,None], xs[None,:]-Delta))
W_minus= w_sv*G0(circ(xs[:,None], xs[None,:]+Delta))
def G(u):
    up=np.clip(u,0,None); num=up**2
    return num/(1 + k*rho*np.sum(num))
def bump_height(wr):
    return wr*(1+math.sqrt(1-wc**2/wr**2))/(4*math.sqrt(math.pi)*k*sigma)
U0=bump_height(wr)

v_vals = np.linspace(-10,10,21)
Rplus_map=[]
Rminus_map=[]
for v_cmd in v_vals:
    u_s = U0*np.exp(-(xs**2)/(4*sigma**2))
    u_p = np.zeros(N)
    u_m = np.zeros(N)
    steps=int(3.0/dt)
    for _ in range(steps):
        r_s = G(u_s)
        r_p = np.clip((g_v + v_cmd)*u_p,0,None)
        r_m = np.clip((g_v - v_cmd)*u_m,0,None)
        conv = rho*(W_r@r_s + W_plus@r_p + W_minus@r_m)
        u_s += dt/tau*(-u_s+conv)
        u_p += dt/tau*(-u_p + w_vs*r_s)
        u_m += dt/tau*(-u_m + w_vs*r_s)
    R_p = np.clip((g_v + v_cmd)*u_p,0,None)
    R_m = np.clip((g_v - v_cmd)*u_m,0,None)
    # align to bump centre
    vec=np.sum(r_s*np.exp(1j*np.deg2rad(xs)))/np.sum(r_s)
    s=np.rad2deg(np.angle(vec))
    rel_x = circ(xs, s)
    order=np.argsort(rel_x)
    Rplus_map.append(R_p[order])
    Rminus_map.append(R_m[order])
rel_sorted = rel_x[order]
Rplus_map=np.array(Rplus_map)
Rminus_map=np.array(Rminus_map)

# plot heatmaps
fig,axs=plt.subplots(1,2,figsize=(9,4),sharey=True)
im1=axs[0].imshow(Rplus_map, aspect='auto', origin='lower',
                  extent=[rel_sorted[0], rel_sorted[-1], v_vals[0], v_vals[-1]],
                  cmap='viridis')
axs[0].set_title('V+ joint tuning')
axs[0].set_xlabel('x - s (deg)')
axs[0].set_ylabel('speed v (°/τ)')
fig.colorbar(im1, ax=axs[0], label='R+ rate')

im2=axs[1].imshow(Rminus_map, aspect='auto', origin='lower',
                  extent=[rel_sorted[0], rel_sorted[-1], v_vals[0], v_vals[-1]],
                  cmap='viridis')
axs[1].set_title('V- joint tuning')
axs[1].set_xlabel('x - s (deg)')
fig.colorbar(im2, ax=axs[1], label='R- rate')
plt.tight_layout()
plt.show()