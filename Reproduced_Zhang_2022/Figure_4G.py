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
G0 = lambda delta: (1/(math.sqrt(2*math.pi)*sigma))*np.exp(-(delta**2)/(2*sigma**2))*dx

W_r = wr * G0(circ(xs[:,None], xs[None,:]))
W_plus = w_sv * G0(circ(xs[:,None], xs[None,:]+Delta))  # shift negative relative coordinates?
W_minus= w_sv * G0(circ(xs[:,None], xs[None,:]-Delta))

def G(u):
    up=np.clip(u,0,None); num=up**2
    return num/(1 + k*rho*np.sum(num))

def bump_height(wr):
    return wr*(1+math.sqrt(1-wc**2/wr**2))/(4*math.sqrt(math.pi)*k*sigma)

U0=bump_height(wr)
T=6.0
steps=int(T/dt)
t_vec=np.arange(steps)*dt

def measure_bump_speed(v_cmd):
    u_s = U0*np.exp(-(xs**2)/(4*sigma**2))
    u_p = np.zeros(N)
    u_m = np.zeros(N)
    centres=[]
    for _ in range(steps):
        r_s = G(u_s)
        r_p = np.clip((g_v + v_cmd)*u_p,0,None)
        r_m = np.clip((g_v - v_cmd)*u_m,0,None)
        conv = rho*(W_r@r_s + W_plus@r_p + W_minus@r_m)
        u_s += dt/tau*(-u_s + conv)
        u_p += dt/tau*(-u_p + w_vs*r_s)
        u_m += dt/tau*(-u_m + w_vs*r_s)
        vec = np.sum(r_s*np.exp(1j*np.deg2rad(xs)))/np.sum(r_s)
        centres.append(np.rad2deg(np.angle(vec)))
    centres=np.unwrap(np.deg2rad(centres)); centres=np.rad2deg(centres)
    slope,_ = np.polyfit(t_vec, centres, 1)
    return slope

v_cmds = np.linspace(-10,10,21)
v_net = [measure_bump_speed(v) for v in v_cmds]
plt.figure(figsize=(4,4))
plt.plot(v_cmds, v_net, 'o')
plt.plot(v_cmds, v_cmds, '--')
plt.show()