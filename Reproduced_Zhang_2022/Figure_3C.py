import numpy as np, math, matplotlib.pyplot as plt

# Parameters
N, rho, sigma = 180, 0.5, 40.0
k, tau, dt = 5e-4, 1.0, 0.01
wc = 2*math.sqrt(2)*(2*math.pi)**0.25*math.sqrt(k*sigma/rho)
wr = 1.05*wc
# grid
L, dx = N/rho, 1/rho
xs = np.linspace(-L/2, L/2, N, endpoint=False)
circ = lambda a,b: (a-b+L/2)%L - L/2
W0 = wr/(math.sqrt(2*math.pi)*sigma)*np.exp(-(circ(xs[:,None],xs[None,:])**2)/(2*sigma**2))*dx
dWdx = -(circ(xs[:,None],xs[None,:]))/sigma**2 * W0

def G(u):
    up=np.clip(u,0,None)
    num=up**2
    return num/(1 + k*rho*np.sum(num))

def bump_height(wr):
    return wr*(1+math.sqrt(1-wc**2/wr**2))/(4*math.sqrt(math.pi)*k*sigma)

U0=bump_height(wr)

# simulation parameters
T=5.0
steps=int(T/dt)
t_vec=np.arange(steps)*dt

def measure_speed(vcmd):
    u = U0*np.exp(-(xs**2)/(4*sigma**2))
    centre=[]
    for t in range(steps):
        r = G(u)
        conv = rho*(W0 - tau*vcmd*dWdx) @ r
        u += dt/tau * (-u + conv)
        vec=np.sum(r*np.exp(1j*np.deg2rad(xs)))/np.sum(r)
        centre.append(np.angle(vec))
    centre=np.unwrap(centre)
    slope, _ = np.polyfit(t_vec, centre, 1)
    return np.rad2deg(slope)

v_tests=np.linspace(-50,50,21)
measured=[]
for v in v_tests:
    measured.append(measure_speed(v))
measured=np.array(measured)

plt.figure(figsize=(4,4))
plt.plot(v_tests, measured,'o',label='network')
plt.plot(v_tests, v_tests,'--',label='theory')
plt.xlabel('commanded speed v (°/τ)')
plt.ylabel('measured speed (°/τ)')
plt.title('Translation speed: network vs theory')
plt.legend(); plt.tight_layout()
plt.show()