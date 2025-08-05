#!/usr/bin/env python
# ======================================================================
#  te_can_pathint.py  – Translation‑equivariant CAN on 1‑D path‑integration
# ======================================================================
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import os, argparse, math, random, numpy as np, matplotlib.pyplot as plt
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

# ──────────────────────────────────────────────────────────────────────
#  Dataset
# ──────────────────────────────────────────────────────────────────────
class PathInt1D(Dataset):
    def __init__(self, n_seq=10_000, T=100, dt=0.02, v_max=0.1, seed=0):
        super().__init__(); rng=np.random.default_rng(seed)
        v = rng.uniform(-v_max, v_max, size=(n_seq, T, 1))
        s = np.mod(np.cumsum(v*dt, axis=1), 1.0)
        self.x = torch.tensor(v, dtype=torch.float32)
        self.y = torch.tensor(s, dtype=torch.float32)
    def __len__(self):  return len(self.x)
    def __getitem__(self,i): return self.x[i], self.y[i]

# ──────────────────────────────────────────────────────────────────────
#  TE‑CAN
# ──────────────────────────────────────────────────────────────────────
class TECAN(torch.nn.Module):
    def __init__(self, N=180, dt=0.02, sigma=40., v_pref_deg=22.,
                 rho_deg=0.5, k=5e-4, wr=0.94, w_vs=0.2):
        super().__init__()
        self.N, self.dt = N, dt
        xs = torch.linspace(-180, 180-1/rho_deg, N)
        self.register_buffer("xs", xs)

        # recurrent kernels -> buffers (not trainable)
        Δx = self._circ(xs[:,None], xs[None,:])
        W_r = (wr/(math.sqrt(2*math.pi)*sigma) *
               torch.exp(-(Δx**2)/(2*sigma**2))) * (1/rho_deg)
        self.register_buffer("W_r", W_r)

        shift = lambda d: torch.exp(-(d**2)/(2*sigma**2))
        coeff = (wr/(math.sqrt(2*math.pi)*sigma)) * (1/rho_deg)
        self.register_buffer("W_plus",
                             coeff*shift(self._circ(xs[:,None], xs[None,:]+v_pref_deg)))
        self.register_buffer("W_minus",
                             coeff*shift(self._circ(xs[:,None], xs[None,:]-v_pref_deg)))

        self.rho_w, self.k = 1/rho_deg, k
        self.w_vs, self.tau = w_vs, 1.0

        # trainable params
        self.B_v = torch.nn.Parameter(torch.randn(1)/10)
        self.readout = torch.nn.Linear(N,1,bias=False)

        # initial bump
        U_peak = wr*(1+math.sqrt(1-0.896**2/wr**2))/(4*math.sqrt(math.pi)*k*sigma)
        u_s0 = U_peak * torch.exp(-(xs**2)/(4*sigma**2))
        h0 = torch.cat((u_s0,
                        torch.zeros_like(u_s0),
                        torch.zeros_like(u_s0)))
        self.register_buffer("h_init", h0)

    @staticmethod
    def _circ(a,b): return (a-b+180.)%360.-180.
    def _rates_s(self,u_s):
        up = torch.clamp(u_s,0.)
        num = up**2
        return num/(1+self.k*self.rho_w*torch.sum(num,dim=-1,keepdim=True))

    def _step(self, h, v_now):
        u_s,u_p,u_m = torch.split(h,self.N,dim=-1)
        r_s = self._rates_s(u_s)
        r_p = torch.clamp((10.+v_now)*u_p,0.)
        r_m = torch.clamp((10.-v_now)*u_m,0.)
        du_s = (-u_s + self.rho_w*(r_s@self.W_r.T +
                                   r_p@self.W_plus.T +
                                   r_m@self.W_minus.T))/self.tau
        du_p = (-u_p + self.w_vs*r_s)/self.tau
        du_m = (-u_m + self.w_vs*r_s)/self.tau
        return h + self.dt*torch.cat((du_s,du_p,du_m),dim=-1)

    def forward(self, vel):                          # (B,T,1)
        B,T,_ = vel.shape
        h = self.h_init.unsqueeze(0).repeat(B,1)
        outs=[]
        for t in range(T):
            h = self._step(h, vel[:,t]*self.B_v)
            outs.append(self.readout(self._rates_s(h[:,:self.N])))
        return torch.stack(outs,1)

# ──────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────
def evaluate(net, loader, device):
    net.eval(); errs=[]
    with torch.no_grad():
        for v,s in loader:
            v,s = v.to(device), s.to(device)
            errs.append(((net(v)-s)**2).mean().sqrt())
    return torch.stack(errs).mean().item()

def run_experiment(args):
    dev=torch.device(args.device)
    tr_ds=PathInt1D(8000,args.T,args.dt)
    va_ds=PathInt1D(2000,args.T,args.dt,seed=1)
    tr_ld=DataLoader(tr_ds,args.batch,shuffle=True)
    va_ld=DataLoader(va_ds,args.batch,shuffle=False)

    curves=[]
    for trial in range(1,args.trials+1):
        seed = args.seed_base + trial          # 0, 1, 2, …
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        net=TECAN(N=args.N,dt=args.dt).to(dev)
        opt=torch.optim.Adam(net.parameters(),lr=args.lr)
        hist=[]
        for epoch in range(1,args.epochs+1):
            net.train()
            for v,s in tr_ld:
                v,s=v.to(dev),s.to(dev)
                opt.zero_grad()
                loss=((net(v)-s)**2).mean().sqrt()
                loss.backward(); clip_grad_norm_(net.parameters(),1.0)
                opt.step()
            rmse=evaluate(net,va_ld,dev); hist.append(rmse)
            tqdm.write(f"[TE‑CAN T{trial:02d}] epoch {epoch:02d}/{args.epochs} "
                       f"val‑RMSE={rmse:.4f}")
        curves.append(np.array(hist))
        np.savetxt(f"rmse_TECAN_T{trial:02d}.csv",
                   np.column_stack([np.arange(1,args.epochs+1),hist]),
                   header="epoch,rmse",delimiter=',',comments='')
    curves=np.vstack(curves); np.save("rmse_TECAN_all.npy",curves)
    plot_curves(curves,"TECAN")

def plot_curves(mat,tag):
    ep=np.arange(1,mat.shape[1]+1)
    for row in mat: plt.plot(ep,row,color="steelblue",alpha=.3)
    plt.plot(ep,mat.mean(0),color="crimson",lw=2.5,label="mean")
    plt.xlabel("Epoch"); plt.ylabel("RMSE"); plt.grid(alpha=.3); plt.legend()
    plt.tight_layout(); plt.savefig(f"rmse_{tag}.png",dpi=300); plt.close()

# ──────────────────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────────────────
if __name__=="__main__":
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    ap=argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--batch",  type=int, default=128)
    ap.add_argument("--N",      type=int, default=200)
    ap.add_argument("--lr",     type=float, default=1e-5)
    ap.add_argument("--dt",     type=float, default=0.02)
    ap.add_argument("--T",      type=int, default=100)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                                          else "cpu")
    ap.add_argument("--seed_base", type=int, default=0,
                help="base RNG seed; actual seed is seed_base + trial")
    run_experiment(ap.parse_args())
