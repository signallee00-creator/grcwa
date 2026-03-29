import numbers

import torch

from .fft_funs import Epsilon_fft, get_ifft
from .kbloch import Lattice_Reciprocate, Lattice_getG, Lattice_SetKs


def _is_scalar_like(value):
    return isinstance(value, numbers.Number) or (torch.is_tensor(value) and value.ndim == 0)


class obj:
    def __init__(self,nG,L1,L2,freq,theta,phi,verbose=1,device=None,dtype_f=torch.float64,dtype_c=torch.complex128):
        '''The time harmonic convention is exp(-i omega t), speed of light = 1.'''

        if device is None and torch.is_tensor(freq):
            device = freq.device
        self.device = torch.device(device) if device is not None else torch.device('cpu')
        self.dtype_f = dtype_f
        self.dtype_c = dtype_c

        freq_tensor = torch.as_tensor(freq, device=self.device)
        self.freq = freq_tensor.to(dtype=self.dtype_c if torch.is_complex(freq_tensor) else self.dtype_f)
        self.omega = 2*torch.pi*torch.as_tensor(freq, dtype=self.dtype_c, device=self.device)
        self.L1 = torch.as_tensor(L1, dtype=self.dtype_f, device=self.device)
        self.L2 = torch.as_tensor(L2, dtype=self.dtype_f, device=self.device)
        self.phi = torch.as_tensor(phi, dtype=self.dtype_f, device=self.device)
        self.theta = torch.as_tensor(theta, dtype=self.dtype_f, device=self.device)
        self.nG = int(nG)
        self.verbose = verbose
        self.Layer_N = 0

        self.thickness_list = []
        self.id_list = []

        self.kp_list = []
        self.q_list = []
        self.phi_list = []

        self.Uniform_ep_list = []
        self.Uniform_N = 0

        self.Patterned_N = 0
        self.Patterned_epinv_list = []
        self.Patterned_ep2_list = []

        self.GridLayer_N = 0
        self.GridLayer_Nxy_list = []

        self.FourierLayer_N = 0
        self.FourierLayer_params = []

    def Add_LayerUniform(self,thickness,epsilon):
        self.id_list.append([0,self.Layer_N,self.Uniform_N])
        epsilon_tensor = torch.as_tensor(epsilon, device=self.device)
        self.Uniform_ep_list.append(epsilon_tensor.to(dtype=self.dtype_c if torch.is_complex(epsilon_tensor) else self.dtype_f))
        self.thickness_list.append(torch.as_tensor(thickness, dtype=self.dtype_f, device=self.device))

        self.Layer_N += 1
        self.Uniform_N += 1

    def Add_LayerGrid(self,thickness,Nx,Ny):
        self.thickness_list.append(torch.as_tensor(thickness, dtype=self.dtype_f, device=self.device))
        self.GridLayer_Nxy_list.append([Nx,Ny])
        self.id_list.append([1,self.Layer_N,self.Patterned_N,self.GridLayer_N])

        self.Layer_N += 1
        self.GridLayer_N += 1
        self.Patterned_N += 1

    def Add_LayerFourier(self,thickness,params):
        self.thickness_list.append(torch.as_tensor(thickness, dtype=self.dtype_f, device=self.device))
        self.FourierLayer_params.append(params)
        self.id_list.append([2,self.Layer_N,self.Patterned_N,self.FourierLayer_N])

        self.Layer_N += 1
        self.Patterned_N += 1
        self.FourierLayer_N += 1

    def Init_Setup(self,Pscale=1.,Gmethod=0):
        ep0 = torch.as_tensor(self.Uniform_ep_list[0], dtype=self.dtype_c, device=self.device)
        kx0 = self.omega*torch.sin(self.theta)*torch.cos(self.phi)*torch.sqrt(ep0)
        ky0 = self.omega*torch.sin(self.theta)*torch.sin(self.phi)*torch.sqrt(ep0)

        self.Lk1, self.Lk2 = Lattice_Reciprocate(self.L1,self.L2,dtype=self.dtype_f,device=self.device)
        self.G,self.nG = Lattice_getG(self.nG,self.Lk1,self.Lk2,method=Gmethod)

        Pscale = torch.as_tensor(Pscale, dtype=self.dtype_f, device=self.device)
        self.Lk1 = self.Lk1/Pscale
        self.Lk2 = self.Lk2/Pscale
        self.kx,self.ky = Lattice_SetKs(self.G, kx0, ky0, self.Lk1, self.Lk2)

        self.normalization = torch.sqrt(torch.as_tensor(torch.real(ep0), dtype=self.dtype_f, device=self.device))/torch.cos(self.theta)

        if self.verbose>0:
            print('Total nG = ',self.nG)

        self.Patterned_ep2_list = [None]*self.Patterned_N
        self.Patterned_epinv_list = [None]*self.Patterned_N
        for i in range(self.Layer_N):
            if self.id_list[i][0] == 0:
                ep = self.Uniform_ep_list[self.id_list[i][2]]
                kp = MakeKPMatrix(self.omega,0,1./ep,self.kx,self.ky,dtype_c=self.dtype_c)
                self.kp_list.append(kp)

                q,phi = SolveLayerEigensystem_uniform(self.omega,self.kx,self.ky,ep,dtype_c=self.dtype_c)
                self.q_list.append(q)
                self.phi_list.append(phi)
            else:
                self.kp_list.append(None)
                self.q_list.append(None)
                self.phi_list.append(None)

    def MakeExcitationPlanewave(self,p_amp,p_phase,s_amp,s_phase,order=0,direction='forward'):
        self.direction = direction
        theta = self.theta
        phi = self.phi
        p_amp = torch.as_tensor(p_amp, dtype=self.dtype_f, device=self.device)
        p_phase = torch.as_tensor(p_phase, dtype=self.dtype_f, device=self.device)
        s_amp = torch.as_tensor(s_amp, dtype=self.dtype_f, device=self.device)
        s_phase = torch.as_tensor(s_phase, dtype=self.dtype_f, device=self.device)

        a0 = torch.zeros(2*self.nG,dtype=self.dtype_c,device=self.device)
        bN = torch.zeros(2*self.nG,dtype=self.dtype_c,device=self.device)
        if direction == 'forward':
            tmp1 = torch.zeros(2*self.nG,dtype=self.dtype_c,device=self.device)
            tmp1[order] = 1.0
            a0 = a0 + tmp1*(-s_amp*torch.cos(theta)*torch.cos(phi)*torch.exp(1j*s_phase) - p_amp*torch.sin(phi)*torch.exp(1j*p_phase))

            tmp2 = torch.zeros(2*self.nG,dtype=self.dtype_c,device=self.device)
            tmp2[order+self.nG] = 1.0
            a0 = a0 + tmp2*(-s_amp*torch.cos(theta)*torch.sin(phi)*torch.exp(1j*s_phase) + p_amp*torch.cos(phi)*torch.exp(1j*p_phase))
        elif direction == 'backward':
            tmp1 = torch.zeros(2*self.nG,dtype=self.dtype_c,device=self.device)
            tmp1[order] = 1.0
            bN = bN + tmp1*(-s_amp*torch.cos(theta)*torch.cos(phi)*torch.exp(1j*s_phase) - p_amp*torch.sin(phi)*torch.exp(1j*p_phase))

            tmp2 = torch.zeros(2*self.nG,dtype=self.dtype_c,device=self.device)
            tmp2[order+self.nG] = 1.0
            bN = bN + tmp2*(-s_amp*torch.cos(theta)*torch.sin(phi)*torch.exp(1j*s_phase) + p_amp*torch.cos(phi)*torch.exp(1j*p_phase))

        self.a0 = a0
        self.bN = bN

    def GridLayer_geteps(self,ep_all):
        ptri = 0
        ptr = 0
        for i in range(self.Layer_N):
            if self.id_list[i][0] != 1:
                continue

            Nx = self.GridLayer_Nxy_list[ptri][0]
            Ny = self.GridLayer_Nxy_list[ptri][1]
            dN = 1./Nx/Ny

            if isinstance(ep_all,(list,tuple)) and len(ep_all) == 3 and ep_all[0].ndim > 0:
                if ep_all[0].ndim == 2 and self.GridLayer_N == 1:
                    ep_grid = [torch.as_tensor(component, dtype=self.dtype_f, device=self.device).reshape(Nx,Ny) for component in ep_all]
                else:
                    ep_grid = [torch.as_tensor(component[ptr:ptr+Nx*Ny], dtype=self.dtype_f, device=self.device).reshape(Nx,Ny) for component in ep_all]
            else:
                if getattr(ep_all,'ndim',None) == 2 and self.GridLayer_N == 1:
                    ep_grid = torch.as_tensor(ep_all, dtype=self.dtype_f, device=self.device).reshape(Nx,Ny)
                else:
                    ep_source = torch.as_tensor(ep_all, dtype=self.dtype_f, device=self.device)
                    ep_grid = ep_source[ptr:ptr+Nx*Ny].reshape(Nx,Ny)

            epinv, ep2 = Epsilon_fft(dN,ep_grid,self.G,dtype_f=self.dtype_f,dtype_c=self.dtype_c,device=self.device)

            self.Patterned_epinv_list[self.id_list[i][2]] = epinv
            self.Patterned_ep2_list[self.id_list[i][2]] = ep2

            kp = MakeKPMatrix(self.omega,1,epinv,self.kx,self.ky,dtype_c=self.dtype_c)
            self.kp_list[self.id_list[i][1]] = kp

            q,phi = SolveLayerEigensystem(self.omega,self.kx,self.ky,kp,ep2)
            self.q_list[self.id_list[i][1]] = q
            self.phi_list[self.id_list[i][1]] = phi

            ptr += Nx*Ny
            ptri += 1

    def Return_eps(self,which_layer,Nx,Ny,component='xx'):
        i = which_layer
        if self.id_list[i][0] == 0:
            ep = self.Uniform_ep_list[self.id_list[i][2]]
            return torch.ones((Nx,Ny),dtype=ep.dtype,device=self.device)*ep

        if component == 'zz':
            epk = torch.linalg.inv(self.Patterned_epinv_list[self.id_list[i][2]])
        elif component == 'xx':
            epk = self.Patterned_ep2_list[self.id_list[i][2]][:self.nG,:self.nG]
        elif component == 'xy':
            epk = self.Patterned_ep2_list[self.id_list[i][2]][:self.nG,self.nG:]
        elif component == 'yx':
            epk = self.Patterned_ep2_list[self.id_list[i][2]][self.nG:,:self.nG]
        elif component == 'yy':
            epk = self.Patterned_ep2_list[self.id_list[i][2]][self.nG:,self.nG:]
        else:
            raise ValueError('Unknown epsilon component')

        return get_ifft(Nx,Ny,epk[0,:],self.G,dtype_c=self.dtype_c,device=self.device)

    def RT_Solve(self,normalize=0,byorder=0):
        aN, b0 = SolveExterior(self.a0,self.bN,self.q_list,self.phi_list,self.kp_list,self.thickness_list,dtype_c=self.dtype_c)
        _,bi = GetZPoyntingFlux(self.a0,b0,self.omega,self.kp_list[0],self.phi_list[0],self.q_list[0],byorder=byorder)
        fe,_ = GetZPoyntingFlux(aN,self.bN,self.omega,self.kp_list[-1],self.phi_list[-1],self.q_list[-1],byorder=byorder)

        if self.direction == 'forward':
            R = torch.real(-bi)
            T = torch.real(fe)
        elif self.direction == 'backward':
            R = torch.real(fe)
            T = torch.real(-bi)
        else:
            raise ValueError('Unknown excitation direction')

        if normalize == 1:
            R = R*self.normalization
            T = T*self.normalization
        return R,T

    def GetAmplitudes_noTranslate(self,which_layer):
        if which_layer == 0 :
            aN, b0 = SolveExterior(self.a0,self.bN,self.q_list,self.phi_list,self.kp_list,self.thickness_list,dtype_c=self.dtype_c)
            ai = self.a0
            bi = b0

        elif which_layer == self.Layer_N-1:
            aN, b0 = SolveExterior(self.a0,self.bN,self.q_list,self.phi_list,self.kp_list,self.thickness_list,dtype_c=self.dtype_c)
            ai = aN
            bi = self.bN

        else:
            ai, bi = SolveInterior(which_layer,self.a0,self.bN,self.q_list,self.phi_list,self.kp_list,self.thickness_list,dtype_c=self.dtype_c)
        return ai,bi

    def GetAmplitudes(self,which_layer,z_offset):
        if which_layer == 0 :
            aN, b0 = SolveExterior(self.a0,self.bN,self.q_list,self.phi_list,self.kp_list,self.thickness_list,dtype_c=self.dtype_c)
            ai = self.a0
            bi = b0

        elif which_layer == self.Layer_N-1:
            aN, b0 = SolveExterior(self.a0,self.bN,self.q_list,self.phi_list,self.kp_list,self.thickness_list,dtype_c=self.dtype_c)
            ai = aN
            bi = self.bN

        else:
            ai, bi = SolveInterior(which_layer,self.a0,self.bN,self.q_list,self.phi_list,self.kp_list,self.thickness_list,dtype_c=self.dtype_c)

        ai, bi = TranslateAmplitudes(self.q_list[which_layer],self.thickness_list[which_layer],z_offset,ai,bi,dtype_f=self.dtype_f)
        return ai,bi

    def Solve_FieldFourier(self,which_layer,z_offset):
        ai0,bi0 = self.GetAmplitudes_noTranslate(which_layer)
        zl = [z_offset] if _is_scalar_like(z_offset) else z_offset

        eh = []
        for zoff in zl:
            ai, bi = TranslateAmplitudes(self.q_list[which_layer],self.thickness_list[which_layer],zoff,ai0,bi0,dtype_f=self.dtype_f)
            fhxy = torch.matmul(self.phi_list[which_layer],ai+bi)
            fhx = fhxy[:self.nG]
            fhy = fhxy[self.nG:]

            tmp1 = (ai-bi)/self.omega/self.q_list[which_layer]
            tmp2 = torch.matmul(self.phi_list[which_layer],tmp1)
            fexy = torch.matmul(self.kp_list[which_layer],tmp2)
            fey = -fexy[:self.nG]
            fex = fexy[self.nG:]

            fhz = (self.kx*fey - self.ky*fex)/self.omega

            fez = (self.ky*fhx - self.kx*fhy)/self.omega
            if self.id_list[which_layer][0] == 0:
                fez = fez / self.Uniform_ep_list[self.id_list[which_layer][2]]
            else:
                fez = torch.matmul(self.Patterned_epinv_list[self.id_list[which_layer][2]],fez)
            eh.append([[fex,fey,fez],[fhx,fhy,fhz]])
        return eh

    def Solve_FieldOnGrid(self,which_layer,z_offset,Nxy=None):
        if Nxy is None:
            Nxy = self.GridLayer_Nxy_list[self.id_list[which_layer][3]]
        Nx = Nxy[0]
        Ny = Nxy[1]

        fehl = self.Solve_FieldFourier(which_layer,z_offset)

        eh = []
        for feh in fehl:
            fe = feh[0]
            fh = feh[1]
            ex = get_ifft(Nx,Ny,fe[0],self.G,dtype_c=self.dtype_c,device=self.device)
            ey = get_ifft(Nx,Ny,fe[1],self.G,dtype_c=self.dtype_c,device=self.device)
            ez = get_ifft(Nx,Ny,fe[2],self.G,dtype_c=self.dtype_c,device=self.device)

            hx = get_ifft(Nx,Ny,fh[0],self.G,dtype_c=self.dtype_c,device=self.device)
            hy = get_ifft(Nx,Ny,fh[1],self.G,dtype_c=self.dtype_c,device=self.device)
            hz = get_ifft(Nx,Ny,fh[2],self.G,dtype_c=self.dtype_c,device=self.device)
            eh.append([[ex,ey,ez],[hx,hy,hz]])
        if _is_scalar_like(z_offset):
            eh = eh[0]
        return eh

    def Volume_integral(self,which_layer,Mx,My,Mz,normalize=0):
        kp = self.kp_list[which_layer]
        q = self.q_list[which_layer]
        phi = self.phi_list[which_layer]

        if self.id_list[which_layer][0] == 0:
            epinv = 1. / self.Uniform_ep_list[self.id_list[which_layer][2]]
        else:
            epinv = self.Patterned_epinv_list[self.id_list[which_layer][2]]

        ai, bi = SolveInterior(which_layer,self.a0,self.bN,self.q_list,self.phi_list,self.kp_list,self.thickness_list,dtype_c=self.dtype_c)
        ab = torch.hstack((ai,bi))
        abMatrix = torch.outer(ab,torch.conj(ab))

        Mt = Matrix_zintegral(q,self.thickness_list[which_layer])
        abM = abMatrix * Mt

        Faxy = torch.matmul(torch.matmul(kp,phi), torch.diag(1./self.omega/q))
        Faz1 = 1./self.omega*torch.matmul(epinv,torch.diag(self.ky))
        Faz2 = -1./self.omega*torch.matmul(epinv,torch.diag(self.kx))
        Faz = torch.matmul(torch.hstack((Faz1,Faz2)),phi)

        tmp1 = torch.vstack((Faxy,Faz))
        tmp2 = torch.vstack((-Faxy,Faz))
        F = torch.hstack((tmp1,tmp2))

        Mx = torch.as_tensor(Mx, dtype=F.dtype, device=self.device)
        My = torch.as_tensor(My, dtype=F.dtype, device=self.device)
        Mz = torch.as_tensor(Mz, dtype=F.dtype, device=self.device)
        Mzeros = torch.zeros_like(Mx)
        Mtotal = torch.vstack((torch.hstack((Mx,Mzeros,Mzeros)),
                               torch.hstack((Mzeros,My,Mzeros)),
                               torch.hstack((Mzeros,Mzeros,Mz))))

        tmp = torch.matmul(torch.matmul(torch.conj(torch.transpose(F,0,1)),Mtotal),F)
        val = torch.trace(torch.matmul(abM,tmp))

        if normalize == 1:
            val = val*self.normalization
        return val

    def Solve_ZStressTensorIntegral(self,which_layer):
        eh = self.Solve_FieldFourier(which_layer,0.0)
        e = eh[0][0]
        h = eh[0][1]
        ex = e[0]
        ey = e[1]
        ez = e[2]

        hx = h[0]
        hy = h[1]
        hz = h[2]

        dz = (self.ky*hx - self.kx*hy)/self.omega

        if self.id_list[which_layer][0] == 0:
            dx = ex * self.Uniform_ep_list[self.id_list[which_layer][2]]
            dy = ey * self.Uniform_ep_list[self.id_list[which_layer][2]]
        else:
            exy = torch.hstack((-ey,ex))
            dxy = torch.matmul(self.Patterned_ep2_list[self.id_list[which_layer][2]],exy)
            dx = dxy[self.nG:]
            dy = -dxy[:self.nG]

        Tx = torch.sum(ex*torch.conj(dz)+hx*torch.conj(hz))
        Ty = torch.sum(ey*torch.conj(dz)+hy*torch.conj(hz))
        Tz = 0.5*torch.sum(ez*torch.conj(dz)+hz*torch.conj(hz)-ey*torch.conj(dy)-ex*torch.conj(dx)-torch.abs(hx)**2-torch.abs(hy)**2)

        return torch.real(Tx), torch.real(Ty), torch.real(Tz)


def MakeKPMatrix(omega,layer_type,epinv,kx,ky,dtype_c=torch.complex128):
    nG = len(kx)

    Jk = torch.vstack((torch.diag(-ky),torch.diag(kx)))
    eye = torch.eye(2*nG,dtype=dtype_c,device=kx.device)
    if layer_type == 0:
        JkkJT = torch.matmul(Jk,torch.transpose(Jk,0,1))
        kp = omega**2*eye - epinv*JkkJT
    else:
        tmp = torch.matmul(Jk,epinv)
        kp = omega**2*eye - torch.matmul(tmp,torch.transpose(Jk,0,1))

    return kp


def SolveLayerEigensystem_uniform(omega,kx,ky,epsilon,dtype_c=torch.complex128):
    nG = len(kx)
    q = torch.sqrt(torch.as_tensor(epsilon, dtype=dtype_c, device=kx.device)*omega**2 - kx**2 - ky**2)
    q = torch.where(torch.imag(q)<0.,-q,q)

    q = torch.concatenate((q,q))
    phi = torch.eye(2*nG,dtype=dtype_c,device=kx.device)
    return q,phi


def SolveLayerEigensystem(omega,kx,ky,kp,ep2):
    k = torch.vstack((torch.diag(kx),torch.diag(ky)))
    kkT = torch.matmul(k,torch.transpose(k,0,1))
    M = torch.matmul(ep2,kp) - kkT

    q,phi = torch.linalg.eig(M)
    q = torch.sqrt(q)
    q = torch.where(torch.imag(q)<0.,-q,q)
    return q,phi


def GetSMatrix(indi,indj,q_list,phi_list,kp_list,thickness_list,dtype_c=torch.complex128):
    nG2 = q_list[0].shape[0]
    device = q_list[0].device
    S11 = torch.eye(nG2,dtype=dtype_c,device=device)
    S12 = torch.zeros_like(S11)
    S21 = torch.zeros_like(S11)
    S22 = torch.eye(nG2,dtype=dtype_c,device=device)
    if indi == indj:
        return S11,S12,S21,S22
    if indi>indj:
        raise Exception('indi must be < indj')

    for l in range(indi,indj):
        lp1 = l+1

        Q = torch.matmul(torch.linalg.inv(phi_list[l]),  phi_list[lp1])
        P1 = torch.matmul(torch.diag(q_list[l]), torch.linalg.inv(torch.matmul(kp_list[l],phi_list[l])))
        P2 = torch.matmul(torch.matmul(kp_list[lp1],phi_list[lp1]), torch.diag(1./q_list[lp1]))
        P = torch.matmul(P1,P2)

        T11 = 0.5*(Q+P)
        T12 = 0.5*(Q-P)

        d1 = torch.diag(torch.exp(1j*q_list[l]*thickness_list[l]))
        d2 = torch.diag(torch.exp(1j*q_list[lp1]*thickness_list[lp1]))

        P1 = T11 - torch.matmul(torch.matmul(d1,S12),T12)
        P1 = torch.linalg.inv(P1)
        S11 = torch.matmul(torch.matmul(P1,d1),S11)

        P2 = torch.matmul(d1,torch.matmul(S12,T11))-T12
        S12 = torch.matmul(torch.matmul(P1,P2),d2)

        S21 = S21 + torch.matmul(S22,torch.matmul(T12,S11))

        P2 = torch.matmul(S22,torch.matmul(T12,S12))
        P1 = torch.matmul(S22,torch.matmul(T11,d2))
        S22 = P1 + P2

    return S11,S12,S21,S22


def SolveExterior(a0,bN,q_list,phi_list,kp_list,thickness_list,dtype_c=torch.complex128):
    Nlayer = len(thickness_list)
    S11, S12, S21, S22 = GetSMatrix(0,Nlayer-1,q_list,phi_list,kp_list,thickness_list,dtype_c=dtype_c)

    aN = torch.matmul(S11,a0) + torch.matmul(S12,bN)
    b0 = torch.matmul(S21,a0) + torch.matmul(S22,bN)

    return aN,b0


def SolveInterior(which_layer,a0,bN,q_list,phi_list,kp_list,thickness_list,dtype_c=torch.complex128):
    Nlayer = len(thickness_list)
    nG2 = q_list[0].shape[0]
    device = q_list[0].device

    S11, S12, S21, S22 = GetSMatrix(0,which_layer,q_list,phi_list,kp_list,thickness_list,dtype_c=dtype_c)
    pS11, pS12, pS21, pS22 = GetSMatrix(which_layer,Nlayer-1,q_list,phi_list,kp_list,thickness_list,dtype_c=dtype_c)

    tmp = torch.linalg.inv(torch.eye(nG2,dtype=dtype_c,device=device)-torch.matmul(S12,pS21))
    ai = torch.matmul(tmp, torch.matmul(S11,a0)+torch.matmul(S12,torch.matmul(pS22,bN)))
    bi = torch.matmul(pS21,ai) + torch.matmul(pS22,bN)

    return ai,bi


def TranslateAmplitudes(q,thickness,dz,ai,bi,dtype_f=torch.float64):
    thickness = torch.as_tensor(thickness, dtype=dtype_f, device=q.device)
    dz = torch.as_tensor(dz, dtype=dtype_f, device=q.device)
    aim = ai*torch.exp(1j*q*dz)
    bim = bi*torch.exp(1j*q*(thickness-dz))
    return aim,bim


def GetZPoyntingFlux(ai,bi,omega,kp,phi,q,byorder=0):
    n = ai.shape[0]//2
    A = torch.matmul(torch.matmul(kp,phi), torch.diag(1./omega/q))

    pa = torch.matmul(phi,ai)
    pb = torch.matmul(phi,bi)
    Aa = torch.matmul(A,ai)
    Ab = torch.matmul(A,bi)

    diff = 0.5*(torch.conj(pb)*Aa-torch.conj(Ab)*pa)
    forward_xy = torch.real(torch.conj(Aa)*pa) + diff
    backward_xy = -torch.real(torch.conj(Ab)*pb) + torch.conj(diff)

    forward = forward_xy[:n] + forward_xy[n:]
    backward = backward_xy[:n] + backward_xy[n:]
    if byorder == 0:
        forward = torch.sum(forward)
        backward = torch.sum(backward)

    return forward, backward


def Matrix_zintegral(q,thickness,shift=1e-12):
    nG2 = q.shape[0]
    qi,qj = Gmeshgrid(q)

    qij = qj-torch.conj(qi)+torch.eye(nG2,dtype=q.dtype,device=q.device)*shift
    Maa = (torch.exp(1j*qij*thickness)-1)/1j/qij

    qij2 = qj+torch.conj(qi)
    Mab = (torch.exp(1j*qj*thickness)-torch.exp(-1j*torch.conj(qi)*thickness))/1j/qij2

    tmp1 = torch.vstack((Maa,Mab))
    tmp2 = torch.vstack((Mab,Maa))
    Mt = torch.hstack((tmp1,tmp2))
    return Mt


def Gmeshgrid(x):
    N = x.shape[0]
    qj = x.reshape(1,N).repeat(N,1)
    qi = x.reshape(N,1).repeat(1,N)
    return qi,qj
