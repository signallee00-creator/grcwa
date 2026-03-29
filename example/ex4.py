"""Transmission and reflection of a hexagonal lattice of a hole."""
import grcwa
import torch

# Truncation order (actual number might be smaller)
nG = 101
# lattice constants
angle = torch.pi/3
L1 = [0.1,0]
L2 = [0.1*torch.cos(torch.tensor(angle)).item(),0.1*torch.sin(torch.tensor(angle)).item()]
# frequency and angles
freq = 1.
theta = 0.
phi = 0.
# to avoid singular matrix, alternatively, one can add fictitious small loss to vacuum
Qabs = float('inf')
freqcmp = freq*(1+1j/2/Qabs)
# the patterned layer has a griding: Nx*Ny
Nx = 1000
Ny = Nx

# now consider 3 layers: vacuum + patterned + vacuum
ep0 = 1. # dielectric for layer 1 (uniform)
epp = 4. # dielectric for patterned layer
epbkg = 1. # dielectric for holes in the patterned layer 
epN = 1.  # dielectric for layer N (uniform)

thick0 = 1. # thickness for vacuum layer 1
thickp = 0.4 # thickness of patterned layer
thickN = 1.

# eps for patterned layer
epgrid = torch.ones((Nx,Ny),dtype=torch.float64)*epp

## note the eps-matrix is defined in the non-orthogonal coordinate given by basis vectors. To define a sphere, we need to obtain the Cartesian coordinate, as shown below,

# coordinate in the non-orthogonal coordinate
u0 = torch.linspace(0,1.,Nx,dtype=torch.float64)
v0 = torch.linspace(0,1.,Ny,dtype=torch.float64)
u, v = torch.meshgrid(u0,v0,indexing='ij')

radius = 0.3
uc0 = 0.5 # center of sphere
vc0 = 0.5
# to define the sphere, let's transform to Cartesian coordinate
x = u + v * torch.cos(torch.tensor(angle))
y = v * torch.sin(torch.tensor(angle))

xc0 = uc0 + vc0 * torch.cos(torch.tensor(angle))
yc0 = vc0 * torch.sin(torch.tensor(angle))

sphere = (x-xc0)**2+(y-yc0)**2<radius**2
epgrid[sphere] = epbkg

######### setting up RCWA
obj = grcwa.obj(nG,L1,L2,freqcmp,theta,phi,verbose=1)
# input layer information
obj.Add_LayerUniform(thick0,ep0)
obj.Add_LayerGrid(thickp,Nx,Ny)
obj.Add_LayerUniform(thickN,epN)
obj.Init_Setup()

# planewave excitation
planewave={'p_amp':1,'s_amp':0,'p_phase':0,'s_phase':0}
obj.MakeExcitationPlanewave(planewave['p_amp'],planewave['p_phase'],planewave['s_amp'],planewave['s_phase'],order = 0)
# eps in patterned layer
obj.GridLayer_geteps(epgrid.flatten())

# compute reflection and transmission
R,T= obj.RT_Solve(normalize=1)
print('R=',R,', T=',T,', R+T=',R+T)
