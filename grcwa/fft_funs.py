import torch


def Epsilon_fft(dN,eps_grid,G,dtype_f=torch.float64,dtype_c=torch.complex128,device=None):
    """Build convolution matrices for isotropic or diagonal-anisotropic epsilon grids."""
    if device is None and torch.is_tensor(G):
        device = G.device

    if isinstance(eps_grid,(list,tuple)) and len(eps_grid) == 3 and eps_grid[0].ndim == 2:
        epsx_fft = get_conv(dN,eps_grid[0],G,dtype_f=dtype_f,device=device)
        epsy_fft = get_conv(dN,eps_grid[1],G,dtype_f=dtype_f,device=device)
        epsz_fft = get_conv(dN,eps_grid[2],G,dtype_f=dtype_f,device=device)
        epsinv = torch.linalg.inv(epsz_fft.to(dtype=dtype_c))

        tmp1 = torch.vstack((epsx_fft,torch.zeros_like(epsx_fft)))
        tmp2 = torch.vstack((torch.zeros_like(epsx_fft),epsy_fft))
        eps2 = torch.hstack((tmp1,tmp2))

    elif eps_grid.ndim == 2:
        eps_fft = get_conv(dN,eps_grid,G,dtype_f=dtype_f,device=device)
        epsinv = torch.linalg.inv(eps_fft.to(dtype=dtype_c))

        tmp1 = torch.vstack((eps_fft,torch.zeros_like(eps_fft)))
        tmp2 = torch.vstack((torch.zeros_like(eps_fft),eps_fft))
        eps2 = torch.hstack((tmp1,tmp2))
    else:
        raise ValueError("Wrong eps_grid type")

    return epsinv.to(dtype=dtype_c), eps2.to(dtype=dtype_c)


def get_conv(dN,s_in,G,dtype_f=torch.float64,device=None):
    G = G.to(dtype=torch.long, device=device if device is not None else G.device)
    s_in = torch.as_tensor(s_in, dtype=dtype_f, device=G.device)
    nG,_ = G.shape
    sfft = torch.fft.fft2(s_in)*dN

    ix = torch.arange(nG,device=G.device,dtype=torch.long)
    ii,jj = torch.meshgrid(ix,ix,indexing='ij')
    s_out = sfft[G[ii,0]-G[jj,0], G[ii,1]-G[jj,1]]
    return s_out


def get_fft(dN,s_in,G,dtype_f=torch.float64,device=None):
    G = G.to(dtype=torch.long, device=device if device is not None else G.device)
    s_in = torch.as_tensor(s_in, dtype=dtype_f, device=G.device)
    sfft = torch.fft.fft2(s_in)*dN
    return sfft[G[:,0],G[:,1]]


def get_ifft(Nx,Ny,s_in,G,dtype_c=torch.complex128,device=None):
    G = G.to(dtype=torch.long, device=device if device is not None else G.device)
    s_in = torch.as_tensor(s_in, dtype=dtype_c, device=G.device)
    dN = 1./Nx/Ny

    s0 = torch.zeros((Nx,Ny),dtype=dtype_c,device=G.device)
    s0[torch.remainder(G[:,0],Nx), torch.remainder(G[:,1],Ny)] = s_in

    s_out = torch.fft.ifft2(s0)/dN
    return s_out


def get_ifft_batch(Nx,Ny,s_in,G,dtype_c=torch.complex128,device=None):
    G = G.to(dtype=torch.long, device=device if device is not None else G.device)
    s_in = torch.as_tensor(s_in, dtype=dtype_c, device=G.device)
    if s_in.shape[-1] != G.shape[0]:
        raise ValueError("Last dimension of s_in must match the number of G vectors")

    original_shape = s_in.shape[:-1]
    flat = s_in.reshape(-1, G.shape[0])
    dN = 1./Nx/Ny

    s0 = torch.zeros((flat.shape[0],Nx,Ny),dtype=dtype_c,device=G.device)
    gx = torch.remainder(G[:,0],Nx)
    gy = torch.remainder(G[:,1],Ny)
    s0[:,gx,gy] = flat

    s_out = torch.fft.ifft2(s0,dim=(-2,-1))/dN
    return s_out.reshape(*original_shape,Nx,Ny)


def get_ifft_xline(x_coords,y0,s_in,G,dtype_c=torch.complex128,device=None):
    out = get_ifft_xline_batch(x_coords,y0,s_in,G,dtype_c=dtype_c,device=device)
    return out


def get_ifft_yline(x0,y_coords,s_in,G,dtype_c=torch.complex128,device=None):
    out = get_ifft_yline_batch(x0,y_coords,s_in,G,dtype_c=dtype_c,device=device)
    return out


def get_ifft_xline_batch(x_coords,y0,s_in,G,dtype_c=torch.complex128,device=None):
    G = G.to(dtype=torch.long, device=device if device is not None else G.device)
    s_in = torch.as_tensor(s_in, dtype=dtype_c, device=G.device)
    if s_in.shape[-1] != G.shape[0]:
        raise ValueError("Last dimension of s_in must match the number of G vectors")

    coord_dtype = torch.float64 if dtype_c == torch.complex128 else torch.float32
    x_coords = torch.as_tensor(x_coords, dtype=coord_dtype, device=G.device)
    y0 = torch.as_tensor(y0, dtype=coord_dtype, device=G.device)

    flat = s_in.reshape(-1, G.shape[0])
    gx = G[:,0].to(dtype=coord_dtype)
    gy = G[:,1].to(dtype=coord_dtype)

    phase_y = torch.exp(2j*torch.pi*gy*y0)
    phase_x = torch.exp(2j*torch.pi*torch.outer(x_coords,gx))
    out = torch.matmul(flat*phase_y.unsqueeze(0), torch.transpose(phase_x,0,1))
    return out.reshape(*s_in.shape[:-1], x_coords.shape[0])


def get_ifft_yline_batch(x0,y_coords,s_in,G,dtype_c=torch.complex128,device=None):
    G = G.to(dtype=torch.long, device=device if device is not None else G.device)
    s_in = torch.as_tensor(s_in, dtype=dtype_c, device=G.device)
    if s_in.shape[-1] != G.shape[0]:
        raise ValueError("Last dimension of s_in must match the number of G vectors")

    coord_dtype = torch.float64 if dtype_c == torch.complex128 else torch.float32
    x0 = torch.as_tensor(x0, dtype=coord_dtype, device=G.device)
    y_coords = torch.as_tensor(y_coords, dtype=coord_dtype, device=G.device)

    flat = s_in.reshape(-1, G.shape[0])
    gx = G[:,0].to(dtype=coord_dtype)
    gy = G[:,1].to(dtype=coord_dtype)

    phase_x = torch.exp(2j*torch.pi*gx*x0)
    phase_y = torch.exp(2j*torch.pi*torch.outer(y_coords,gy))
    out = torch.matmul(flat*phase_x.unsqueeze(0), torch.transpose(phase_y,0,1))
    return out.reshape(*s_in.shape[:-1], y_coords.shape[0])
