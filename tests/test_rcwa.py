import torch
import grcwa
from .utils import t_grad

tol = 1e-2
tolS4 = 1e-3 # error tolerance for S4 v.s. this code

Nlayer = 1
nG = 101    
L1 = [0.1,0]
L2 = [0,0.1]
# all patterned layers below have the same griding structure: Nx*Ny
Nx = 100
Ny = 100

# now consider 3 layers: vacuum + patterned + vacuum
epsuniform0 = 1. # dielectric for layer 1 (uniform)
epsuniformN = 1.  # dielectric for layer N (uniform)

thick0 = 1. # thickness for vacuum layer 1
thickN = 1.

# frequency and angles
freq = 1.
theta = torch.pi/18
phi = torch.pi/9
Pscale = 1.

pthick = [0.2]    
# eps for patterned layer
radius = 0.4
epgrid = torch.ones((Nx,Ny),dtype=torch.float64)
x0 = torch.linspace(0,1.,Nx,dtype=torch.float64)
y0 = torch.linspace(0,1.,Ny,dtype=torch.float64)
x, y = torch.meshgrid(x0,y0,indexing='ij')
sphere = (x-.5)**2+(y-.5)**2<radius**2
epgrid[sphere] = 12.

planewave={'p_amp':1,'s_amp':0,'p_phase':0,'s_phase':0}

def rcwa_assembly(epgrid,freq,theta,phi,planewave,pthick,Pscale=1.):
    '''
    planewave:{'p_amp',...}
    '''
    obj = grcwa.obj(nG,L1,L2,freq,theta,phi,verbose=1)
    obj.Add_LayerUniform(thick0,epsuniform0)
    for i in range(Nlayer):
        obj.Add_LayerGrid(pthick[i],Nx,Ny)
    obj.Add_LayerUniform(thickN,epsuniformN)
    
    obj.Init_Setup(Pscale=Pscale,Gmethod=0)
    obj.MakeExcitationPlanewave(planewave['p_amp'],planewave['p_phase'],planewave['s_amp'],planewave['s_phase'],order = 0)
    obj.GridLayer_geteps(epgrid)
    
    return obj

   
def test_rcwa():
    planewave={'p_amp':1,'s_amp':0,'p_phase':0,'s_phase':0}
    obj=rcwa_assembly(epgrid,freq,theta,phi,planewave,pthick,Pscale=1.)
    R,T= obj.RT_Solve(normalize=0)
    assert abs(T.item()-0.85249901083265)<tolS4 * T.item()

    planewave={'p_amp':0,'s_amp':1,'p_phase':0,'s_phase':0}
    obj=rcwa_assembly(epgrid,freq,theta,phi,planewave,pthick,Pscale=1.)
    R,T= obj.RT_Solve(normalize=0)
    assert abs(T.item()-0.83900479939861)<tolS4 * T.item()

    ai,bi = obj.GetAmplitudes(1,0.)
    assert len(ai) == obj.nG*2

    e,h = obj.Solve_FieldOnGrid(1,0.)
    assert e[0].shape == (Nx,Ny)

    Mx = torch.real(obj.Patterned_epinv_list[0])
    val = obj.Volume_integral(1,Mx,Mx,Mx,normalize=1)
    assert torch.real(val).item()>0

    Tx,Ty,Tz = obj.Solve_ZStressTensorIntegral(0)
    assert Tz.item()<0


def test_epsgrad():
    def fun(x):
        obj=rcwa_assembly(x,freq,theta,phi,planewave,pthick,Pscale=1.)
        R,_= obj.RT_Solve(normalize=1)
        return R

    x = epgrid.flatten()
    ind = torch.randint(Nx*Ny*Nlayer,size=(1,)).item()
    FD, AD = t_grad(fun,x,1e-3,ind)
    ref = max(torch.abs(FD).item(),1e-12)
    assert torch.abs(FD-AD).item()<ref*tol,'wrong epsgrid gradient'


def test_thickgrad():
    def fun(x):
        obj=rcwa_assembly(epgrid.flatten(),freq,theta,phi,planewave,x,Pscale=1.)
        R,_= obj.RT_Solve(normalize=1)
        return R

    x = torch.tensor([0.1],dtype=torch.float64)
    FD, AD = t_grad(fun,x,1e-3,0)
    ref = max(torch.abs(FD).item(),1e-12)
    assert torch.abs(FD-AD).item()<ref*tol,'wrong thickness gradient'


def test_periodgrad():
    def fun(x):
        obj=rcwa_assembly(epgrid.flatten(),freq,theta,phi,planewave,pthick,Pscale=x)
        R,_= obj.RT_Solve(normalize=1)
        return R

    x = torch.tensor(1.0,dtype=torch.float64)
    FD, AD = t_grad(fun,x,1e-3,0)
    ref = max(torch.abs(FD).item(),1e-12)
    assert torch.abs(FD-AD).item()<ref*tol,'wrong thickness gradient'


def test_freqgrad():
    def fun(x):
        obj=rcwa_assembly(epgrid.flatten(),x,theta,phi,planewave,pthick,Pscale=1.)
        R,_= obj.RT_Solve(normalize=1)
        return R

    x = torch.tensor(1.0,dtype=torch.float64)
    FD, AD = t_grad(fun,x,1e-3,0)
    ref = max(torch.abs(FD).item(),1e-12)
    assert torch.abs(FD-AD).item()<ref*tol,'wrong thickness gradient'


def test_thetagrad():
    def fun(x):
        obj=rcwa_assembly(epgrid.flatten(),freq,x,phi,planewave,pthick,Pscale=1.)
        R,_= obj.RT_Solve(normalize=1)
        return R

    x = torch.tensor(torch.pi/10,dtype=torch.float64)
    FD, AD = t_grad(fun,x,1e-3,0)
    ref = max(torch.abs(FD).item(),1e-12)
    assert torch.abs(FD-AD).item()<ref*tol,'wrong thickness gradient'
