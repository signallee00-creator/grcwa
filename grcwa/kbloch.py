import torch


def Lattice_Reciprocate(L1,L2,dtype=torch.float64,device=None):
    '''Given two lattice vectors L1,L2 in the form of (Lx,Ly), returns the
    reciprocal vectors Lk/(2*pi).
    '''

    assert len(L1) == 2,'Both x,y components of Lattice vector L1 are required.'
    assert len(L2) == 2,'Both x,y components of Lattice vector L2 are required.'

    L1 = torch.as_tensor(L1, dtype=dtype, device=device)
    L2 = torch.as_tensor(L2, dtype=dtype, device=L1.device)
    d = L1[0]*L2[1]-L1[1]*L2[0]

    Lk1 = torch.stack((L2[1]/d, -L2[0]/d))
    Lk2 = torch.stack((-L1[1]/d, L1[0]/d))

    return Lk1,Lk2


def Lattice_getG(nG,Lk1,Lk2,method=0):
    '''
    The G is defined to produce the following reciprocal vector:
    k = G[:,0] Lk1 + G[:,1] Lk2 (both k and Lk don't include the 2pi factor)

    method:0 for circular truncation, 1 for parallelogramic truncation
    '''
    assert type(nG) == int, 'nG must be integar'

    if method == 0:
        G,nG = Gsel_circular(nG, Lk1, Lk2)
    elif method == 1:
        G,nG = Gsel_parallelogramic(nG, Lk1, Lk2)
    else:
        raise Exception('Truncation scheme is not included')

    return G,nG


def Lattice_SetKs(G, kx0, ky0, Lk1, Lk2):
    '''
    Construct kx,ky including all relevant orders, given initial scalar kx,ky.
    The returned kx,ky include the 2pi factor.
    '''

    G = G.to(dtype=torch.long)
    device = G.device
    dtype = Lk1.dtype if torch.is_tensor(Lk1) else torch.float64
    Lk1 = torch.as_tensor(Lk1, dtype=dtype, device=device)
    Lk2 = torch.as_tensor(Lk2, dtype=dtype, device=device)

    kx = kx0 + 2*torch.pi*(Lk1[0]*G[:,0]+Lk2[0]*G[:,1])
    ky = ky0 + 2*torch.pi*(Lk1[1]*G[:,0]+Lk2[1]*G[:,1])

    return kx,ky


def Gsel_parallelogramic(nG, Lk1, Lk2):
    ''' From Liu's gsel.c '''
    dtype = Lk1.dtype if torch.is_tensor(Lk1) else torch.float64
    device = Lk1.device if torch.is_tensor(Lk1) else None
    Lk1 = torch.as_tensor(Lk1, dtype=dtype, device=device)
    Lk2 = torch.as_tensor(Lk2, dtype=dtype, device=Lk1.device)

    u = torch.linalg.norm(Lk1)
    v = torch.linalg.norm(Lk2)
    uv = torch.dot(Lk1,Lk2)

    NGroot = int(nG**0.5)
    if NGroot % 2 == 0:
        NGroot -= 1

    M = NGroot//2

    xG = torch.arange(-M,NGroot-M,device=Lk1.device,dtype=torch.long)
    G1,G2 = torch.meshgrid(xG,xG,indexing='ij')
    G1 = G1.reshape(-1)
    G2 = G2.reshape(-1)

    Gl2 = G1**2*u**2+G2**2*v**2+2*G2*G1*uv
    sort = torch.argsort(Gl2)
    G1 = G1[sort]
    G2 = G2[sort]

    nG = NGroot*NGroot
    G = torch.zeros((nG,2),dtype=torch.long,device=Lk1.device)
    G[:,0] = G1[:nG]
    G[:,1] = G2[:nG]

    return G, nG


def Gsel_circular(nG, Lk1, Lk2):
    '''From Liu's gsel.c.'''
    dtype = Lk1.dtype if torch.is_tensor(Lk1) else torch.float64
    device = Lk1.device if torch.is_tensor(Lk1) else None
    Lk1 = torch.as_tensor(Lk1, dtype=dtype, device=device)
    Lk2 = torch.as_tensor(Lk2, dtype=dtype, device=Lk1.device)

    u = torch.linalg.norm(Lk1)
    v = torch.linalg.norm(Lk2)
    uv = torch.dot(Lk1,Lk2)
    uxv = Lk1[0]*Lk2[1] - Lk1[1]*Lk2[0]
    circ_area = nG * torch.abs(uxv)
    circ_radius = torch.sqrt(circ_area/torch.pi) + u+v

    scale = torch.sqrt(1.-uv**2/(u*v)**2)
    u_extent = 1+int((circ_radius/(u*scale)).item())
    v_extent = 1+int((circ_radius/(v*scale)).item())

    uext21 = 2*u_extent+1
    vext21 = 2*v_extent+1

    xG = torch.arange(-u_extent,uext21-u_extent,device=Lk1.device,dtype=torch.long)
    yG = torch.arange(-v_extent,vext21-v_extent,device=Lk1.device,dtype=torch.long)
    G1,G2 = torch.meshgrid(xG,yG,indexing='ij')
    G1 = G1.reshape(-1)
    G2 = G2.reshape(-1)

    Gl2 = G1**2*u**2+G2**2*v**2+2*G2*G1*uv
    sort = torch.argsort(Gl2)
    G1 = G1[sort]
    G2 = G2[sort]
    Gl2 = Gl2[sort]

    nGtmp = uext21*vext21

    if nG < nGtmp:
        nGtmp = nG

    tol = 1e-10*max((u**2).item(),(v**2).item())
    for i in range(nGtmp-1,-1,-1):
        if abs((Gl2[i]-Gl2[i-1]).item())>tol:
            break
    nG = i

    G = torch.zeros((nG,2),dtype=torch.long,device=Lk1.device)
    G[:,0] = G1[:nG]
    G[:,1] = G2[:nG]

    return G,nG
