import torch
import grcwa
from .utils import t_grad

L1 = [0.5,0]
L2 = [0,0.2]
nG = 100
method = 0
kx0 = torch.tensor(0.1,dtype=torch.float64)
ky0 = torch.tensor(0.2,dtype=torch.float64)

Lk1,Lk2 = grcwa.Lattice_Reciprocate(L1,L2)
G,nGout = grcwa.Lattice_getG(nG,Lk1,Lk2,method=method)
kx, ky = grcwa.Lattice_SetKs(G, kx0, ky0, Lk1, Lk2)
    
def test_bloch():
    assert nGout>0,'negative nG'
    assert nGout<=nG,'wrong nG'

Nx = 51
Ny = 71
dN = 1./Nx/Ny
tol = 1e-2


def test_fft():
    def fun(ep):
        epout = ep.reshape(Nx,Ny)
        epsinv, _ = grcwa.Epsilon_fft(dN,epout,G)
        return torch.real(torch.sum(epsinv))

    x = 1.+10.*torch.rand(Nx*Ny,dtype=torch.float64)
    ind = torch.randint(Nx*Ny,size=(1,)).item()
    FD, AD = t_grad(fun,x,1e-3,ind)
    ref = max(torch.abs(FD).item(),1e-12)
    assert torch.abs(FD-AD).item()<ref*tol,'wrong fft gradient'


def test_fft_aniso():
    def fun(ep):
        epout = [ep[x*Nx*Ny:(x+1)*Nx*Ny].reshape(Nx,Ny) for x in range(3)]
        _, eps2 = grcwa.Epsilon_fft(dN,epout,G)
        return torch.real(torch.sum(eps2))

    x = 1.+10.*torch.rand(3*Nx*Ny,dtype=torch.float64)
    ind = torch.randint(Nx*Ny*2,size=(1,)).item()
    FD, AD = t_grad(fun,x,1e-3,ind)
    ref = max(torch.abs(FD).item(),1e-12)
    assert torch.abs(FD-AD).item()<ref*tol,'wrong fft gradient'


def test_ifft():
    ix = torch.randint(Nx,size=(1,)).item()
    iy = torch.randint(Ny,size=(1,)).item()

    def fun(x):
        out = grcwa.get_ifft(Nx,Ny,x,G)
        return torch.real(out[ix,iy])

    x = 10.*torch.rand(nGout,dtype=torch.float64)
    ind = torch.randint(nGout,size=(1,)).item()
    FD, AD = t_grad(fun,x,1e-3,ind)
    ref = max(torch.abs(FD).item(),1e-12)
    assert torch.abs(FD-AD).item()<ref*tol,'wrong ifft gradient'
