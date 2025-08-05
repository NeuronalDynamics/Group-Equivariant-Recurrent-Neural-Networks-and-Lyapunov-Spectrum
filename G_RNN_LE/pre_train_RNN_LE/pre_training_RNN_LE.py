# ================================================================
#  rnn_smnist_lyap_untrained.py – Figure-7 “before-training” curve
# ================================================================
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import os, argparse, pathlib, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import transforms
from tqdm import tqdm
from torch.autograd.functional import jacobian
import matplotlib.pyplot as plt
# ---------------- data loader (unchanged) ----------------
def get_loaders(batch=128, root="./torch_datasets"):
    tfm = transforms.ToTensor()
    root = pathlib.Path(root).expanduser()
    ds_tr = MNIST(root, train=True,  download=True, transform=tfm)
    n_tr  = int(0.8*len(ds_tr))
    ds_tr, _ = random_split(ds_tr, [n_tr, len(ds_tr)-n_tr])
    return DataLoader(ds_tr, batch, shuffle=True, drop_last=True)
# ---------------- model ----------------
class RNNSMNIST(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.hidden = hidden
        self.rnn = nn.RNN(28, hidden, batch_first=True, bias=True)
        self.fc  = nn.Linear(hidden, 10, bias=False)
    def forward(self, x):
        B = x.size(0); seq = x.view(B,28,28)
        h0 = torch.zeros(1,B,self.hidden, device=x.device, dtype=x.dtype)
        y,_ = self.rnn(seq,h0); return self.fc(y[:,-1])
# ---------------- init U(−p,p) ----------------
def init_uniform(model,p_min=0.1,p_max=3.0):
    p = float(torch.round((torch.rand(1)*(p_max-p_min)+p_min)*1e3)/1e3)
    for w in (model.rnn.weight_ih_l0, model.rnn.weight_hh_l0):
        nn.init.uniform_(w,-p,p)
    nn.init.zeros_(model.rnn.bias_ih_l0); nn.init.zeros_(model.rnn.bias_hh_l0)
    return p
# ---------------- jacobian & LE ----------------
def rnn_J(cell,x_t,h_prev):
    h_prev = h_prev.detach().requires_grad_(True)
    return jacobian(lambda h: cell(x_t,h),h_prev,create_graph=False,strict=True)
def lyap_spectrum(model,seq,warm=500):
    H,dev,dty=model.hidden,seq.device,seq.dtype
    cell=nn.RNNCell(28,H,bias=True,device=dev,dtype=dty)
    cell.load_state_dict({'weight_ih':model.rnn.weight_ih_l0,
                          'weight_hh':model.rnn.weight_hh_l0,
                          'bias_ih'  :model.rnn.bias_ih_l0,
                          'bias_hh'  :model.rnn.bias_hh_l0})
    h=torch.zeros(H,device=dev,dtype=dty); Q=torch.eye(H,device=dev,dtype=dty)
    le_sum=torch.zeros(H,device=dev,dtype=dty);steps=0;eps=1e-12
    for t in range(warm): h = cell(seq[t],h)
    for t in range(warm,seq.size(0)):
        J=rnn_J(cell,seq[t],h); Q,R=torch.linalg.qr(J@Q)
        le_sum+=torch.log(torch.clamp(torch.abs(torch.diagonal(R)),min=eps))
        h=cell(seq[t],h); steps+=1
    return (le_sum/steps).cpu()
def make_driver(batch=15,T=600,device='cpu',dtype=torch.float64):
    return torch.rand(batch,T,28,device=device,dtype=dtype)
# ---------------- main ----------------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--trials',type=int,default=200)
    ap.add_argument('--hidden',type=int,default=64)
    ap.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    args=ap.parse_args(); dev=torch.device(args.device)
    torch.set_default_dtype(torch.float64)
    all_LE=[]; all_p=[]
    for run in range(1,args.trials+1):
        net=RNNSMNIST(args.hidden).to(dev).double()
        p=init_uniform(net); all_p.append(p)
        print(f"\n=== untrained trial {run}/{args.trials}   p={p:.3f} ===")
        driver=make_driver(device=dev)
        LEbatch=[lyap_spectrum(net,seq,warm=500).numpy() for seq in driver]
        LE=np.mean(LEbatch,axis=0); all_LE.append(LE)
        np.save(f"LE_trial{run:03d}.npy",LE)
        print(f"  λ₁={LE[0]:+.6f}   λ_H={LE[-1]:+.6f}")
    meanLE=np.mean(all_LE,axis=0); np.save("LE_mean.npy",meanLE)
    plt.plot(range(1,len(meanLE)+1),meanLE,'o-',ms=3,lw=1.4)
    plt.axhline(0,color='k',ls='--',lw=.8)
    plt.xlabel('index  $i$'); plt.ylabel(r'$\bar\lambda_i$')
    plt.title('Lyapunov spectrum – untrained RNN'); plt.tight_layout()
    plt.savefig('LE_mean.png',dpi=300)
    print("\nSaved LE_mean.npy  &  LE_mean.png")
if __name__=='__main__': main()
