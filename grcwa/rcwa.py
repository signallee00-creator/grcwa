import copy
import numbers

import torch

from .fft_funs import Epsilon_fft, get_ifft
from .fft_funs import get_ifft_batch, get_ifft_xline_batch, get_ifft_yline_batch
from .kbloch import Lattice_Reciprocate, Lattice_getG, Lattice_SetKs
STATE_VERSION = 1


# Lightweight tensor and serialization helpers
def _clone_nested(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.clone()
    if isinstance(value, tuple):
        return tuple(_clone_nested(item) for item in value)
    if isinstance(value, list):
        return [_clone_nested(item) for item in value]
    if isinstance(value, dict):
        return {key: _clone_nested(item) for key, item in value.items()}
    if isinstance(value, set):
        return set(value)
    return copy.deepcopy(value)


def _coerce_grid_eps(ep_grid, Nx, Ny, dtype_f, device):
    if isinstance(ep_grid, (list, tuple)):
        if len(ep_grid) != 3:
            raise ValueError('Anisotropic epsilon input must contain exactly three components')
        out = []
        for component in ep_grid:
            tensor = torch.as_tensor(component, dtype=dtype_f, device=device)
            if tensor.numel() != Nx * Ny:
                raise ValueError('Each epsilon component must have Nx*Ny entries')
            out.append(tensor.reshape(Nx, Ny))
        return out

    tensor = torch.as_tensor(ep_grid, dtype=dtype_f, device=device)
    if tensor.numel() != Nx * Ny:
        raise ValueError('Epsilon grid must have Nx*Ny entries')
    return tensor.reshape(Nx, Ny)


def field_coefficients_batched(obj, which_layer, z_offset):
    """Return stacked Fourier coefficients for Ex,Ey,Ez,Hx,Hy,Hz."""
    which_layer = obj._validate_layer_index(which_layer)
    z_tensor = torch.as_tensor(z_offset, dtype=obj.dtype_f, device=obj.device)
    scalar_input = z_tensor.ndim == 0
    if scalar_input:
        z_tensor = z_tensor.reshape(1)
    ai0, bi0 = obj.GetAmplitudes_noTranslate(which_layer)

    q = obj.q_list[which_layer]
    phi = obj.phi_list[which_layer]
    kp = obj.kp_list[which_layer]
    thickness = torch.as_tensor(obj.thickness_list[which_layer], dtype=obj.dtype_f, device=obj.device)

    dz = z_tensor.reshape(-1, 1)
    q_row = q.reshape(1, -1)
    ai = ai0.reshape(1, -1) * torch.exp(1j * q_row * dz)
    bi = bi0.reshape(1, -1) * torch.exp(1j * q_row * (thickness - dz))

    fhxy = torch.matmul(ai + bi, torch.transpose(phi, 0, 1))
    fhx = fhxy[:, :obj.nG]
    fhy = fhxy[:, obj.nG:]

    tmp1 = (ai - bi) / (obj.omega * q_row)
    tmp2 = torch.matmul(tmp1, torch.transpose(phi, 0, 1))
    fexy = torch.matmul(tmp2, torch.transpose(kp, 0, 1))
    fey = -fexy[:, :obj.nG]
    fex = fexy[:, obj.nG:]

    fhz = (obj.kx.reshape(1, -1) * fey - obj.ky.reshape(1, -1) * fex) / obj.omega
    fez = (obj.ky.reshape(1, -1) * fhx - obj.kx.reshape(1, -1) * fhy) / obj.omega
    if obj.id_list[which_layer][0] == 0:
        fez = fez / obj.Uniform_ep_list[obj.id_list[which_layer][2]]
    else:
        epinv = obj.Patterned_epinv_list[obj.id_list[which_layer][2]]
        fez = torch.matmul(fez, torch.transpose(epinv, 0, 1))

    coeffs = torch.stack((fex, fey, fez, fhx, fhy, fhz), dim=1)
    return coeffs, scalar_input


def get_line_coordinates(obj, axis, coords, fixed_coord, which_layer=None):
    """
    Return physical coordinates and reduced coordinates for line cuts.

    If ``which_layer`` is given, the default coordinate count follows that
    layer's grid shape. Otherwise a structure-wide default count is used.
    """
    length_x = torch.linalg.norm(obj.L1)
    length_y = torch.linalg.norm(obj.L2)

    if coords is None:
        if not obj._layer_is_axis_aligned():
            raise ValueError('Auto-generated x/y coordinates are only available for axis-aligned orthogonal lattices')

        if which_layer is not None:
            Nx, Ny = obj._default_grid_shape(which_layer)
            count = Nx if axis == 'x' else Ny
        elif axis == 'x':
            count = max(shape[0] for shape in obj.GridLayer_Nxy_list) if obj.GridLayer_Nxy_list else max(obj.nG, 64)
        else:
            count = max(shape[1] for shape in obj.GridLayer_Nxy_list) if obj.GridLayer_Nxy_list else max(obj.nG, 64)

        if axis == 'x':
            coords = torch.arange(count, dtype=obj.dtype_f, device=obj.device) / count * obj.L1[0]
        else:
            coords = torch.arange(count, dtype=obj.dtype_f, device=obj.device) / count * obj.L2[1]

    coords = torch.as_tensor(coords, dtype=obj.dtype_f, device=obj.device)
    fixed_coord = torch.as_tensor(fixed_coord, dtype=obj.dtype_f, device=obj.device)
    if axis == 'x':
        return coords, coords / length_x, fixed_coord / length_y
    return coords, coords / length_y, fixed_coord / length_x


def build_layer_z_points(obj, thickness, count=None, z_step=None):
    thickness = torch.as_tensor(thickness, dtype=obj.dtype_f, device=obj.device)

    if z_step is None:
        return torch.linspace(0.0, thickness, count, dtype=obj.dtype_f, device=obj.device)

    if thickness.item() == 0.0:
        return torch.zeros((count,), dtype=obj.dtype_f, device=obj.device)

    if thickness.item() <= z_step * (count - 1):
        return torch.linspace(0.0, thickness, count, dtype=obj.dtype_f, device=obj.device)

    local_z = torch.arange(0.0, float(thickness.item()), z_step, dtype=obj.dtype_f, device=obj.device)
    if local_z.numel() == 0 or not torch.isclose(local_z[0], thickness.new_zeros(())):
        local_z = thickness.new_zeros((1,))

    if not torch.isclose(local_z[-1], thickness):
        local_z = torch.concatenate((local_z, thickness.reshape(1)))

    if local_z.numel() < count:
        return torch.linspace(0.0, thickness, count, dtype=obj.dtype_f, device=obj.device)

    return local_z


def build_structure_z_segments(obj, znum=32, z_step=None, duplicate_interfaces=True):
    if torch.is_tensor(znum):
        znum = znum.tolist()

    if isinstance(znum, numbers.Integral):
        min_znum = max(int(znum), 2)
        if z_step is None:
            positive_thicknesses = []
            for thickness in obj.thickness_list:
                thickness_value = float(torch.as_tensor(thickness, dtype=obj.dtype_f, device=obj.device).item())
                if thickness_value > 0.0:
                    positive_thicknesses.append(thickness_value)
            if positive_thicknesses and min_znum > 1:
                mesh_z_step = min(positive_thicknesses) / (min_znum - 1)
            else:
                mesh_z_step = None
            z_counts = None
        else:
            mesh_z_step = float(torch.as_tensor(z_step, dtype=obj.dtype_f, device=obj.device).item())
            if mesh_z_step <= 0.0:
                raise ValueError('z_step must be positive')
            z_counts = None
    else:
        if z_step is not None:
            raise ValueError('z_step cannot be combined with a per-layer znum sequence')
        z_counts = [max(int(value), 2) for value in znum]
        if len(z_counts) != obj.Layer_N:
            raise ValueError('znum sequence must have one entry per layer')
        min_znum = None
        mesh_z_step = None

    z_pieces = []
    segments = []
    layer_ranges = []
    layer_edges = [torch.zeros((), dtype=obj.dtype_f, device=obj.device)]
    z_offset = torch.zeros((), dtype=obj.dtype_f, device=obj.device)
    cursor = 0

    for layer_index, thickness in enumerate(obj.thickness_list):
        thickness = torch.as_tensor(thickness, dtype=obj.dtype_f, device=obj.device)
        if z_counts is None:
            local_z = build_layer_z_points(obj, thickness, count=min_znum, z_step=mesh_z_step)
        else:
            local_z = build_layer_z_points(obj, thickness, count=z_counts[layer_index], z_step=None)

        global_z = z_offset + local_z
        if layer_index > 0 and not duplicate_interfaces:
            local_z = local_z[1:]
            global_z = global_z[1:]

        start = cursor
        stop = cursor + local_z.numel()
        layer_ranges.append((start, stop))
        segments.append((layer_index, local_z))
        z_pieces.append(global_z)
        cursor = stop
        z_offset = z_offset + thickness
        layer_edges.append(z_offset)

    return torch.concatenate(z_pieces), segments, layer_ranges, torch.stack(layer_edges), mesh_z_step


# S-matrix assembly helpers
def identity_smatrix(n, dtype_c, device):
    S11 = torch.eye(n, dtype=dtype_c, device=device)
    S12 = torch.zeros((n, n), dtype=dtype_c, device=device)
    S21 = torch.zeros((n, n), dtype=dtype_c, device=device)
    S22 = torch.eye(n, dtype=dtype_c, device=device)
    return S11, S12, S21, S22


def compose_smatrix(left_smat, right_smat):
    left11, left12, left21, left22 = left_smat
    right11, right12, right21, right22 = right_smat

    n = left11.shape[0]
    eye = torch.eye(n, dtype=left11.dtype, device=left11.device)
    left12_right21 = torch.matmul(left12, right21)
    left12_right22 = torch.matmul(left12, right22)
    lu, pivots = torch.linalg.lu_factor(eye - left12_right21)
    solved11 = torch.linalg.lu_solve(lu, pivots, left11)
    solved12 = torch.linalg.lu_solve(lu, pivots, left12_right22)

    right11_solved11 = torch.matmul(right11, solved11)
    right11_solved12 = torch.matmul(right11, solved12)
    left22_right21 = torch.matmul(left22, right21)
    left22_right22 = torch.matmul(left22, right22)
    left22_right21_solved11 = torch.matmul(left22_right21, solved11)
    left22_right21_solved12 = torch.matmul(left22_right21, solved12)

    new11 = right11_solved11
    new12 = right12 + right11_solved12
    new21 = left21 + left22_right21_solved11
    new22 = left22_right22 + left22_right21_solved12
    return new11, new12, new21, new22


def make_step_smatrix(l, q_list, phi_list, kp_list, thickness_list, dtype_c=torch.complex128):
    lp1 = l + 1

    phi_l = phi_list[l]
    phi_lp1 = phi_list[lp1]
    kp_l = kp_list[l]
    kp_lp1 = kp_list[lp1]
    q_l = q_list[l]
    q_lp1 = q_list[lp1]

    Q = torch.linalg.solve(phi_l, phi_lp1)
    rhs = torch.matmul(kp_lp1, phi_lp1 * (1.0 / q_lp1).reshape(1, -1))
    P = q_l.reshape(-1, 1) * torch.linalg.solve(torch.matmul(kp_l, phi_l), rhs)

    T11 = 0.5 * (Q + P)
    T12 = 0.5 * (Q - P)

    phase1 = torch.exp(1j * q_l * thickness_list[l])
    phase2 = torch.exp(1j * q_lp1 * thickness_list[lp1])
    d1 = torch.diag(phase1)
    T12_d2 = T12 * phase2.reshape(1, -1)
    lu, pivots = torch.linalg.lu_factor(T11)
    step11 = torch.linalg.lu_solve(lu, pivots, d1)
    step12 = torch.linalg.lu_solve(lu, pivots, -T12_d2)
    step21 = torch.matmul(T12, step11)
    step22 = T11 * phase2.reshape(1, -1) + torch.matmul(T12, step12)
    return step11.to(dtype=dtype_c), step12.to(dtype=dtype_c), step21.to(dtype=dtype_c), step22.to(dtype=dtype_c)


def GetSMatrix(indi, indj, q_list, phi_list, kp_list, thickness_list, dtype_c=torch.complex128):
    nG2 = q_list[0].shape[0]
    device = q_list[0].device
    if indi == indj:
        return identity_smatrix(nG2, dtype_c, device)
    if indi > indj:
        raise Exception('indi must be < indj')

    smat = identity_smatrix(nG2, dtype_c, device)
    for l in range(indi, indj):
        smat = compose_smatrix(
            smat,
            make_step_smatrix(l, q_list, phi_list, kp_list, thickness_list, dtype_c=dtype_c),
        )
    return smat


def SolveExterior(a0, bN, q_list, phi_list, kp_list, thickness_list, dtype_c=torch.complex128):
    Nlayer = len(thickness_list)
    S11, S12, S21, S22 = GetSMatrix(0, Nlayer - 1, q_list, phi_list, kp_list, thickness_list, dtype_c=dtype_c)
    aN = torch.matmul(S11, a0) + torch.matmul(S12, bN)
    b0 = torch.matmul(S21, a0) + torch.matmul(S22, bN)
    return aN, b0


def SolveInterior(which_layer, a0, bN, q_list, phi_list, kp_list, thickness_list, dtype_c=torch.complex128):
    Nlayer = len(thickness_list)
    nG2 = q_list[0].shape[0]
    device = q_list[0].device

    S11, S12, _, _ = GetSMatrix(0, which_layer, q_list, phi_list, kp_list, thickness_list, dtype_c=dtype_c)
    _, _, pS21, pS22 = GetSMatrix(which_layer, Nlayer - 1, q_list, phi_list, kp_list, thickness_list, dtype_c=dtype_c)

    rhs = torch.matmul(S11, a0) + torch.matmul(S12, torch.matmul(pS22, bN))
    ai = torch.linalg.solve(torch.eye(nG2, dtype=dtype_c, device=device) - torch.matmul(S12, pS21), rhs)
    bi = torch.matmul(pS21, ai) + torch.matmul(pS22, bN)
    return ai, bi


def SolveInteriorCached(which_layer, a0, bN, prefix_smat, suffix_smat, dtype_c=torch.complex128):
    nG2 = prefix_smat[which_layer][0].shape[0]
    device = prefix_smat[which_layer][0].device

    S11, S12, _, _ = prefix_smat[which_layer]
    _, _, pS21, pS22 = suffix_smat[which_layer]

    rhs = torch.matmul(S11, a0) + torch.matmul(S12, torch.matmul(pS22, bN))
    ai = torch.linalg.solve(torch.eye(nG2, dtype=dtype_c, device=device) - torch.matmul(S12, pS21), rhs)
    bi = torch.matmul(pS21, ai) + torch.matmul(pS22, bN)
    return ai, bi


class obj:
    def __init__(self, nG, L1, L2, freq, theta, phi, verbose=1, device=None, dtype_f=torch.float64, dtype_c=torch.complex128):
        """The time harmonic convention is exp(-i omega t), speed of light = 1."""

        if device is None and torch.is_tensor(freq):
            device = freq.device
        self.device = torch.device(device) if device is not None else torch.device('cpu')
        self.dtype_f = dtype_f
        self.dtype_c = dtype_c

        freq_tensor = torch.as_tensor(freq, device=self.device)
        self.freq = freq_tensor.to(dtype=self.dtype_c if torch.is_complex(freq_tensor) else self.dtype_f)
        self.omega = 2 * torch.pi * torch.as_tensor(freq, dtype=self.dtype_c, device=self.device)
        self.L1 = torch.as_tensor(L1, dtype=self.dtype_f, device=self.device)
        self.L2 = torch.as_tensor(L2, dtype=self.dtype_f, device=self.device)
        self.phi = torch.as_tensor(phi, dtype=self.dtype_f, device=self.device)
        self.theta = torch.as_tensor(theta, dtype=self.dtype_f, device=self.device)
        self.nG_requested = int(nG)
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
        self.GridLayer_epgrid_list = []

        self.FourierLayer_N = 0
        self.FourierLayer_params = []

        self.Pscale = torch.as_tensor(1.0, dtype=self.dtype_f, device=self.device)
        self.Gmethod = 0
        self.setup_ready = False
        self.direction = None

        self._a0 = None
        self._bN = None

        self._smat_cache_ready = False
        self._prefix_smat = None
        self._suffix_smat = None
        self._dirty_layers = set()
        self._exterior_cache = None
        self._amplitude_cache_ready = False
        self._layer_amplitudes = None
        self._amplitude_cache_info = None

    @property
    def a0(self):
        return self._a0

    @a0.setter
    def a0(self, value):
        if value is None:
            self._a0 = None
            self._invalidate_excitation_caches()
            return

        tensor = torch.as_tensor(value, dtype=self.dtype_c, device=self.device)
        if self.setup_ready and tensor.numel() != 2 * self.nG:
            raise ValueError('a0 must have length 2*self.nG')
        self._a0 = tensor.reshape(-1)
        self._invalidate_excitation_caches()

    @property
    def bN(self):
        return self._bN

    @bN.setter
    def bN(self, value):
        if value is None:
            self._bN = None
            self._invalidate_excitation_caches()
            return

        tensor = torch.as_tensor(value, dtype=self.dtype_c, device=self.device)
        if self.setup_ready and tensor.numel() != 2 * self.nG:
            raise ValueError('bN must have length 2*self.nG')
        self._bN = tensor.reshape(-1)
        self._invalidate_excitation_caches()

    def _invalidate_excitation_caches(self):
        self._exterior_cache = None
        self._amplitude_cache_ready = False
        self._layer_amplitudes = None
        self._amplitude_cache_info = None

    def _invalidate_structure_caches(self, clear_dirty=False):
        self._smat_cache_ready = False
        self._prefix_smat = None
        self._suffix_smat = None
        if clear_dirty:
            self._dirty_layers = set()
        self._invalidate_excitation_caches()

    def _ensure_setup_ready(self):
        if not self.setup_ready:
            raise RuntimeError('Call Init_Setup() before solving the structure')

    def _ensure_excitation_ready(self):
        self._ensure_setup_ready()
        if self.a0 is None or self.bN is None:
            raise RuntimeError('Define an excitation with MakeExcitationPlanewave() or by setting a0/bN')

    def _validate_layer_index(self, which_layer):
        which_layer = int(which_layer)
        if which_layer < 0 or which_layer >= self.Layer_N:
            raise IndexError('Layer index out of range')
        return which_layer

    def _layer_is_axis_aligned(self):
        atol = 1e-12 if self.dtype_f == torch.float64 else 1e-6
        zero = torch.zeros((), dtype=self.dtype_f, device=self.device)
        return (
            torch.isclose(self.L1[1], zero, atol=atol, rtol=0.0)
            and torch.isclose(self.L2[0], zero, atol=atol, rtol=0.0)
        )

    def _default_grid_shape(self, which_layer):
        which_layer = self._validate_layer_index(which_layer)
        if self.id_list[which_layer][0] != 1:
            raise ValueError('Nxy must be provided for non-grid layers')
        return self.GridLayer_Nxy_list[self.id_list[which_layer][3]]

    def _ensure_smatrix_cache_current(self):
        self._ensure_setup_ready()
        if any(entry is None for entry in self.kp_list) or any(entry is None for entry in self.q_list) or any(entry is None for entry in self.phi_list):
            raise RuntimeError('All layers must have solved eigensystems before building S-matrix caches')

        if not self._smat_cache_ready or self._prefix_smat is None or self._suffix_smat is None:
            self.BuildSMatrixCache()
        elif self._dirty_layers:
            self.UpdateSMatrixCache()

    def _apply_grid_layer_eps(self, which_layer, ep_grid):
        which_layer = self._validate_layer_index(which_layer)
        layer_id = self.id_list[which_layer]
        if layer_id[0] != 1:
            raise ValueError('GridLayer_updateeps only supports grid-patterned layers')

        grid_id = layer_id[3]
        pattern_id = layer_id[2]
        Nx, Ny = self.GridLayer_Nxy_list[grid_id]
        ep_grid = _coerce_grid_eps(ep_grid, Nx, Ny, self.dtype_f, self.device)

        dN = 1.0 / Nx / Ny
        epinv, ep2 = Epsilon_fft(dN, ep_grid, self.G, dtype_f=self.dtype_f, dtype_c=self.dtype_c, device=self.device)

        self.GridLayer_epgrid_list[grid_id] = _clone_nested(ep_grid)
        self.Patterned_epinv_list[pattern_id] = epinv
        self.Patterned_ep2_list[pattern_id] = ep2

        kp = MakeKPMatrix(self.omega, 1, epinv, self.kx, self.ky, dtype_c=self.dtype_c)
        q, phi = SolveLayerEigensystem(self.omega, self.kx, self.ky, kp, ep2)

        self.kp_list[which_layer] = kp
        self.q_list[which_layer] = q
        self.phi_list[which_layer] = phi

    # Layer definition and setup

    def Add_LayerUniform(self, thickness, epsilon):
        self.id_list.append([0, self.Layer_N, self.Uniform_N])
        epsilon_tensor = torch.as_tensor(epsilon, device=self.device)
        self.Uniform_ep_list.append(epsilon_tensor.to(dtype=self.dtype_c if torch.is_complex(epsilon_tensor) else self.dtype_f))
        self.thickness_list.append(torch.as_tensor(thickness, dtype=self.dtype_f, device=self.device))

        self.Layer_N += 1
        self.Uniform_N += 1
        self._invalidate_structure_caches(clear_dirty=True)

    def Add_LayerGrid(self, thickness, Nx, Ny):
        self.thickness_list.append(torch.as_tensor(thickness, dtype=self.dtype_f, device=self.device))
        self.GridLayer_Nxy_list.append([Nx, Ny])
        self.GridLayer_epgrid_list.append(None)
        self.id_list.append([1, self.Layer_N, self.Patterned_N, self.GridLayer_N])

        self.Layer_N += 1
        self.GridLayer_N += 1
        self.Patterned_N += 1
        self._invalidate_structure_caches(clear_dirty=True)

    def Add_LayerFourier(self, thickness, params):
        self.thickness_list.append(torch.as_tensor(thickness, dtype=self.dtype_f, device=self.device))
        self.FourierLayer_params.append(params)
        self.id_list.append([2, self.Layer_N, self.Patterned_N, self.FourierLayer_N])

        self.Layer_N += 1
        self.Patterned_N += 1
        self.FourierLayer_N += 1
        self._invalidate_structure_caches(clear_dirty=True)

    def Init_Setup(self, Pscale=1.0, Gmethod=0):
        ep0 = torch.as_tensor(self.Uniform_ep_list[0], dtype=self.dtype_c, device=self.device)
        kx0 = self.omega * torch.sin(self.theta) * torch.cos(self.phi) * torch.sqrt(ep0)
        ky0 = self.omega * torch.sin(self.theta) * torch.sin(self.phi) * torch.sqrt(ep0)

        self.Gmethod = int(Gmethod)
        self.Pscale = torch.as_tensor(Pscale, dtype=self.dtype_f, device=self.device)

        self.Lk1, self.Lk2 = Lattice_Reciprocate(self.L1, self.L2, dtype=self.dtype_f, device=self.device)
        self.G, self.nG = Lattice_getG(self.nG_requested, self.Lk1, self.Lk2, method=self.Gmethod)

        self.Lk1 = self.Lk1 / self.Pscale
        self.Lk2 = self.Lk2 / self.Pscale
        self.kx, self.ky = Lattice_SetKs(self.G, kx0, ky0, self.Lk1, self.Lk2)

        self.normalization = torch.real(torch.sqrt(ep0)).to(dtype=self.dtype_f) / torch.cos(self.theta)
        self.setup_ready = True

        if self.verbose > 0:
            print('Total nG = ', self.nG)

        self.kp_list = []
        self.q_list = []
        self.phi_list = []
        self.Patterned_ep2_list = [None] * self.Patterned_N
        self.Patterned_epinv_list = [None] * self.Patterned_N

        for i in range(self.Layer_N):
            if self.id_list[i][0] == 0:
                ep = self.Uniform_ep_list[self.id_list[i][2]]
                kp = MakeKPMatrix(self.omega, 0, 1.0 / ep, self.kx, self.ky, dtype_c=self.dtype_c)
                q, phi = SolveLayerEigensystem_uniform(self.omega, self.kx, self.ky, ep, dtype_c=self.dtype_c)
                self.kp_list.append(kp)
                self.q_list.append(q)
                self.phi_list.append(phi)
            else:
                self.kp_list.append(None)
                self.q_list.append(None)
                self.phi_list.append(None)

        self._a0 = None
        self._bN = None
        self.direction = None
        self._invalidate_structure_caches(clear_dirty=True)

    def MakeExcitationPlanewave(self, p_amp, p_phase, s_amp, s_phase, order=0, direction='forward'):
        self._ensure_setup_ready()
        self.direction = direction
        theta = self.theta
        phi = self.phi
        p_amp = torch.as_tensor(p_amp, dtype=self.dtype_f, device=self.device)
        p_phase = torch.as_tensor(p_phase, dtype=self.dtype_f, device=self.device)
        s_amp = torch.as_tensor(s_amp, dtype=self.dtype_f, device=self.device)
        s_phase = torch.as_tensor(s_phase, dtype=self.dtype_f, device=self.device)

        a0 = torch.zeros(2 * self.nG, dtype=self.dtype_c, device=self.device)
        bN = torch.zeros(2 * self.nG, dtype=self.dtype_c, device=self.device)
        if direction == 'forward':
            tmp1 = torch.zeros(2 * self.nG, dtype=self.dtype_c, device=self.device)
            tmp1[order] = 1.0
            a0 = a0 + tmp1 * (
                -s_amp * torch.cos(theta) * torch.cos(phi) * torch.exp(1j * s_phase)
                - p_amp * torch.sin(phi) * torch.exp(1j * p_phase)
            )

            tmp2 = torch.zeros(2 * self.nG, dtype=self.dtype_c, device=self.device)
            tmp2[order + self.nG] = 1.0
            a0 = a0 + tmp2 * (
                -s_amp * torch.cos(theta) * torch.sin(phi) * torch.exp(1j * s_phase)
                + p_amp * torch.cos(phi) * torch.exp(1j * p_phase)
            )
        elif direction == 'backward':
            tmp1 = torch.zeros(2 * self.nG, dtype=self.dtype_c, device=self.device)
            tmp1[order] = 1.0
            bN = bN + tmp1 * (
                -s_amp * torch.cos(theta) * torch.cos(phi) * torch.exp(1j * s_phase)
                - p_amp * torch.sin(phi) * torch.exp(1j * p_phase)
            )

            tmp2 = torch.zeros(2 * self.nG, dtype=self.dtype_c, device=self.device)
            tmp2[order + self.nG] = 1.0
            bN = bN + tmp2 * (
                -s_amp * torch.cos(theta) * torch.sin(phi) * torch.exp(1j * s_phase)
                + p_amp * torch.cos(phi) * torch.exp(1j * p_phase)
            )
        else:
            raise ValueError('Unknown excitation direction')

        self.a0 = a0
        self.bN = bN

    def GridLayer_geteps(self, ep_all):
        self._ensure_setup_ready()
        ptri = 0
        ptr = 0

        if isinstance(ep_all, (list, tuple)) and len(ep_all) == 3:
            ep_source = [torch.as_tensor(component, dtype=self.dtype_f, device=self.device) for component in ep_all]
            anisotropic = True
        else:
            ep_source = torch.as_tensor(ep_all, dtype=self.dtype_f, device=self.device)
            anisotropic = False

        for i in range(self.Layer_N):
            if self.id_list[i][0] != 1:
                continue

            Nx, Ny = self.GridLayer_Nxy_list[ptri]
            if anisotropic:
                if self.GridLayer_N == 1 and ep_source[0].ndim == 2:
                    ep_grid = [component.reshape(Nx, Ny) for component in ep_source]
                else:
                    ep_grid = [component[ptr:ptr + Nx * Ny].reshape(Nx, Ny) for component in ep_source]
            else:
                if getattr(ep_source, 'ndim', None) == 2 and self.GridLayer_N == 1:
                    ep_grid = ep_source.reshape(Nx, Ny)
                else:
                    ep_grid = ep_source[ptr:ptr + Nx * Ny].reshape(Nx, Ny)

            self._apply_grid_layer_eps(i, ep_grid)
            ptr += Nx * Ny
            ptri += 1

        self._invalidate_structure_caches(clear_dirty=True)

    def GridLayer_updateeps(self, which_layer, ep_grid, update_cache=True):
        self._ensure_setup_ready()
        which_layer = self._validate_layer_index(which_layer)
        self._apply_grid_layer_eps(which_layer, ep_grid)
        self._dirty_layers.add(which_layer)
        self._invalidate_excitation_caches()

        if update_cache:
            self.UpdateSMatrixCache([which_layer])

    def Return_eps(self, which_layer, Nx, Ny, component='xx'):
        i = self._validate_layer_index(which_layer)
        if self.id_list[i][0] == 0:
            ep = self.Uniform_ep_list[self.id_list[i][2]]
            return torch.ones((Nx, Ny), dtype=ep.dtype, device=self.device) * ep

        if component == 'zz':
            epk = torch.linalg.inv(self.Patterned_epinv_list[self.id_list[i][2]])
        elif component == 'xx':
            epk = self.Patterned_ep2_list[self.id_list[i][2]][:self.nG, :self.nG]
        elif component == 'xy':
            epk = self.Patterned_ep2_list[self.id_list[i][2]][:self.nG, self.nG:]
        elif component == 'yx':
            epk = self.Patterned_ep2_list[self.id_list[i][2]][self.nG:, :self.nG]
        elif component == 'yy':
            epk = self.Patterned_ep2_list[self.id_list[i][2]][self.nG:, self.nG:]
        else:
            raise ValueError('Unknown epsilon component')

        return get_ifft(Nx, Ny, epk[0, :], self.G, dtype_c=self.dtype_c, device=self.device)

    # Cache management

    def BuildSMatrixCache(self):
        self._ensure_setup_ready()
        nG2 = self.q_list[0].shape[0]

        self._prefix_smat = [None] * self.Layer_N
        self._suffix_smat = [None] * self.Layer_N

        self._prefix_smat[0] = identity_smatrix(nG2, self.dtype_c, self.device)
        for i in range(1, self.Layer_N):
            step_smat = make_step_smatrix(i - 1, self.q_list, self.phi_list, self.kp_list, self.thickness_list, dtype_c=self.dtype_c)
            self._prefix_smat[i] = compose_smatrix(self._prefix_smat[i - 1], step_smat)

        self._suffix_smat[self.Layer_N - 1] = identity_smatrix(nG2, self.dtype_c, self.device)
        for i in range(self.Layer_N - 2, -1, -1):
            step_smat = make_step_smatrix(i, self.q_list, self.phi_list, self.kp_list, self.thickness_list, dtype_c=self.dtype_c)
            self._suffix_smat[i] = compose_smatrix(step_smat, self._suffix_smat[i + 1])

        self._smat_cache_ready = True
        self._dirty_layers = set()

    def UpdateSMatrixCache(self, changed_layers=None):
        self._ensure_setup_ready()
        if changed_layers is None:
            changed_layers = sorted(self._dirty_layers)
        else:
            changed_layers = sorted({self._validate_layer_index(layer) for layer in changed_layers})

        if not changed_layers:
            return

        if not self._smat_cache_ready or self._prefix_smat is None or self._suffix_smat is None:
            self.BuildSMatrixCache()
            self._invalidate_excitation_caches()
            return

        affected_steps = set()
        for layer in changed_layers:
            if layer > 0:
                affected_steps.add(layer - 1)
            if layer < self.Layer_N - 1:
                affected_steps.add(layer)

        if not affected_steps:
            self._dirty_layers.difference_update(changed_layers)
            self._invalidate_excitation_caches()
            return

        prefix_start = min(affected_steps) + 1
        for i in range(prefix_start, self.Layer_N):
            step_smat = make_step_smatrix(i - 1, self.q_list, self.phi_list, self.kp_list, self.thickness_list, dtype_c=self.dtype_c)
            self._prefix_smat[i] = compose_smatrix(self._prefix_smat[i - 1], step_smat)

        suffix_start = max(affected_steps)
        for i in range(suffix_start, -1, -1):
            step_smat = make_step_smatrix(i, self.q_list, self.phi_list, self.kp_list, self.thickness_list, dtype_c=self.dtype_c)
            self._suffix_smat[i] = compose_smatrix(step_smat, self._suffix_smat[i + 1])

        self._dirty_layers.difference_update(changed_layers)
        self._smat_cache_ready = True
        self._invalidate_excitation_caches()

    def GetExteriorAmplitudesCached(self):
        self._ensure_excitation_ready()
        if self._exterior_cache is None:
            self._ensure_smatrix_cache_current()
            S11, S12, S21, S22 = self._prefix_smat[-1]
            aN = torch.matmul(S11, self.a0) + torch.matmul(S12, self.bN)
            b0 = torch.matmul(S21, self.a0) + torch.matmul(S22, self.bN)
            self._exterior_cache = (aN, b0)
        return self._exterior_cache

    def BuildAmplitudeCache(self):
        self._ensure_excitation_ready()
        self._ensure_smatrix_cache_current()
        self.GetExteriorAmplitudesCached()

        self._layer_amplitudes = [
            SolveInteriorCached(layer_index, self.a0, self.bN, self._prefix_smat, self._suffix_smat, dtype_c=self.dtype_c)
            for layer_index in range(self.Layer_N)
        ]

        self._amplitude_cache_info = {
            'mode': 'exact',
            'aN_abs_error': None,
            'bN_abs_error': None,
            'aN_rel_error': None,
            'bN_rel_error': None,
            'atol': None,
            'rtol': None,
            'layer_count': self.Layer_N,
            'nG': self.nG,
        }
        self._amplitude_cache_ready = True

    def ClearSMatrixCache(self):
        self._smat_cache_ready = False
        self._prefix_smat = None
        self._suffix_smat = None
        self._dirty_layers = set()

    # Core solves and amplitudes

    def RT_Solve(self, normalize=0, byorder=0):
        aN, b0 = self.GetExteriorAmplitudesCached()
        _, bi = GetZPoyntingFlux(self.a0, b0, self.omega, self.kp_list[0], self.phi_list[0], self.q_list[0], byorder=byorder)
        fe, _ = GetZPoyntingFlux(aN, self.bN, self.omega, self.kp_list[-1], self.phi_list[-1], self.q_list[-1], byorder=byorder)

        if self.direction == 'forward':
            R = torch.real(-bi)
            T = torch.real(fe)
        elif self.direction == 'backward':
            R = torch.real(fe)
            T = torch.real(-bi)
        else:
            raise ValueError('Unknown excitation direction')

        if normalize == 1:
            R = R * self.normalization
            T = T * self.normalization
        return R, T

    def GetAmplitudes_noTranslate(self, which_layer):
        which_layer = self._validate_layer_index(which_layer)
        self._ensure_excitation_ready()

        if self._amplitude_cache_ready:
            return self._layer_amplitudes[which_layer]
        self._ensure_smatrix_cache_current()
        return SolveInteriorCached(which_layer, self.a0, self.bN, self._prefix_smat, self._suffix_smat, dtype_c=self.dtype_c)

    def GetAmplitudes(self, which_layer, z_offset):
        ai, bi = self.GetAmplitudes_noTranslate(which_layer)
        ai, bi = TranslateAmplitudes(self.q_list[which_layer], self.thickness_list[which_layer], z_offset, ai, bi, dtype_f=self.dtype_f)
        return ai, bi

    # Field reconstruction

    def Solve_FieldFourier(self, which_layer, z_offset):
        coeffs, _ = field_coefficients_batched(self, which_layer, z_offset)
        out = []
        for z_index in range(coeffs.shape[0]):
            layer_coeffs = coeffs[z_index]
            out.append([
                [layer_coeffs[0], layer_coeffs[1], layer_coeffs[2]],
                [layer_coeffs[3], layer_coeffs[4], layer_coeffs[5]],
            ])
        return out

    def Solve_FieldOnGrid(self, which_layer, z_offset, Nxy=None, components=None):
        if components is None:
            components = ('Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz')

        all_components = ('Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz')
        requested_core = []
        for name in components:
            if name not in all_components:
                raise ValueError(f'Unknown field component: {name}')
            if name not in requested_core:
                requested_core.append(name)

        if Nxy is None:
            Nxy = self._default_grid_shape(which_layer)
        Nx, Ny = Nxy[0], Nxy[1]

        coeffs, scalar_input = field_coefficients_batched(self, which_layer, z_offset)
        indices = [all_components.index(name) for name in requested_core]

        spatial = get_ifft_batch(Nx, Ny, coeffs[:, indices, :], self.G, dtype_c=self.dtype_c, device=self.device)
        core_fields = {name: spatial[:, idx] for idx, name in enumerate(requested_core)}

        legacy = []
        for z_index in range(coeffs.shape[0]):
            Ex = core_fields['Ex'][z_index] if 'Ex' in core_fields else None
            Ey = core_fields['Ey'][z_index] if 'Ey' in core_fields else None
            Ez = core_fields['Ez'][z_index] if 'Ez' in core_fields else None
            Hx = core_fields['Hx'][z_index] if 'Hx' in core_fields else None
            Hy = core_fields['Hy'][z_index] if 'Hy' in core_fields else None
            Hz = core_fields['Hz'][z_index] if 'Hz' in core_fields else None
            E = [Ex, Ey, Ez]
            H = [Hx, Hy, Hz]
            if Ex is None and Ey is None and Ez is None:
                E = None
            if Hx is None and Hy is None and Hz is None:
                H = None
            legacy.append([E, H])

        if scalar_input:
            return legacy[0]
        return legacy

    def Solve_FieldXY(self, which_layer, z_offset, Nxy=None, components=('Ex', 'Ey', 'Ez')):
        all_components = ('Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz')
        requested_core = []
        for name in components:
            if name not in all_components:
                raise ValueError(f'Unknown field component: {name}')
            if name not in requested_core:
                requested_core.append(name)

        if Nxy is None:
            Nxy = self._default_grid_shape(which_layer)
        Nx, Ny = Nxy[0], Nxy[1]

        coeffs, scalar_input = field_coefficients_batched(self, which_layer, z_offset)
        indices = [all_components.index(name) for name in requested_core]

        spatial = get_ifft_batch(Nx, Ny, coeffs[:, indices, :], self.G, dtype_c=self.dtype_c, device=self.device)
        core_fields = {name: spatial[:, idx] for idx, name in enumerate(requested_core)}

        Ex = core_fields['Ex'][0] if scalar_input and 'Ex' in core_fields else core_fields.get('Ex')
        Ey = core_fields['Ey'][0] if scalar_input and 'Ey' in core_fields else core_fields.get('Ey')
        Ez = core_fields['Ez'][0] if scalar_input and 'Ez' in core_fields else core_fields.get('Ez')
        Hx = core_fields['Hx'][0] if scalar_input and 'Hx' in core_fields else core_fields.get('Hx')
        Hy = core_fields['Hy'][0] if scalar_input and 'Hy' in core_fields else core_fields.get('Hy')
        Hz = core_fields['Hz'][0] if scalar_input and 'Hz' in core_fields else core_fields.get('Hz')

        E = [Ex, Ey, Ez]
        H = [Hx, Hy, Hz]
        if Ex is None and Ey is None and Ez is None:
            E = None
        if Hx is None and Hy is None and Hz is None:
            H = None
        return E, H

    def Solve_FieldXZLayer(self, which_layer, z_list, y0=0.0, x_coords=None, components=('Ex',)):
        all_components = ('Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz')
        requested_core = []
        for name in components:
            if name not in all_components:
                raise ValueError(f'Unknown field component: {name}')
            if name not in requested_core:
                requested_core.append(name)

        coeffs, scalar_input = field_coefficients_batched(self, which_layer, z_list)
        x_coords, x_red, y0_red = get_line_coordinates(self, 'x', x_coords, y0, which_layer=which_layer)
        indices = [all_components.index(name) for name in requested_core]

        spatial = get_ifft_xline_batch(x_red, y0_red, coeffs[:, indices, :], self.G, dtype_c=self.dtype_c, device=self.device)
        core_fields = {name: spatial[:, idx] for idx, name in enumerate(requested_core)}

        Ex = core_fields['Ex'][0] if scalar_input and 'Ex' in core_fields else core_fields.get('Ex')
        Ey = core_fields['Ey'][0] if scalar_input and 'Ey' in core_fields else core_fields.get('Ey')
        Ez = core_fields['Ez'][0] if scalar_input and 'Ez' in core_fields else core_fields.get('Ez')
        Hx = core_fields['Hx'][0] if scalar_input and 'Hx' in core_fields else core_fields.get('Hx')
        Hy = core_fields['Hy'][0] if scalar_input and 'Hy' in core_fields else core_fields.get('Hy')
        Hz = core_fields['Hz'][0] if scalar_input and 'Hz' in core_fields else core_fields.get('Hz')

        E = [Ex, Ey, Ez]
        H = [Hx, Hy, Hz]
        if Ex is None and Ey is None and Ez is None:
            E = None
        if Hx is None and Hy is None and Hz is None:
            H = None
        z_coords = torch.as_tensor(z_list, dtype=self.dtype_f, device=self.device)
        if scalar_input:
            z_coords = z_coords.reshape(())
        return E, H, x_coords, z_coords

    def Solve_FieldYZLayer(self, which_layer, z_list, x0=0.0, y_coords=None, components=('Ex',)):
        all_components = ('Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz')
        requested_core = []
        for name in components:
            if name not in all_components:
                raise ValueError(f'Unknown field component: {name}')
            if name not in requested_core:
                requested_core.append(name)

        coeffs, scalar_input = field_coefficients_batched(self, which_layer, z_list)
        y_coords, y_red, x0_red = get_line_coordinates(self, 'y', y_coords, x0, which_layer=which_layer)
        indices = [all_components.index(name) for name in requested_core]

        spatial = get_ifft_yline_batch(x0_red, y_red, coeffs[:, indices, :], self.G, dtype_c=self.dtype_c, device=self.device)
        core_fields = {name: spatial[:, idx] for idx, name in enumerate(requested_core)}

        Ex = core_fields['Ex'][0] if scalar_input and 'Ex' in core_fields else core_fields.get('Ex')
        Ey = core_fields['Ey'][0] if scalar_input and 'Ey' in core_fields else core_fields.get('Ey')
        Ez = core_fields['Ez'][0] if scalar_input and 'Ez' in core_fields else core_fields.get('Ez')
        Hx = core_fields['Hx'][0] if scalar_input and 'Hx' in core_fields else core_fields.get('Hx')
        Hy = core_fields['Hy'][0] if scalar_input and 'Hy' in core_fields else core_fields.get('Hy')
        Hz = core_fields['Hz'][0] if scalar_input and 'Hz' in core_fields else core_fields.get('Hz')

        E = [Ex, Ey, Ez]
        H = [Hx, Hy, Hz]
        if Ex is None and Ey is None and Ez is None:
            E = None
        if Hx is None and Hy is None and Hz is None:
            H = None
        z_coords = torch.as_tensor(z_list, dtype=self.dtype_f, device=self.device)
        if scalar_input:
            z_coords = z_coords.reshape(())
        return E, H, y_coords, z_coords

    def Solve_FieldXZ(self, y0=0.0, x_coords=None, znum=32, z_step=None, components=('Ex',), duplicate_interfaces=True):
        all_components = ('Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz')
        requested_core = []
        for name in components:
            if name not in all_components:
                raise ValueError(f'Unknown field component: {name}')
            if name not in requested_core:
                requested_core.append(name)
        x_coords, x_red, y0_red = get_line_coordinates(self, 'x', x_coords, y0)
        y0 = torch.as_tensor(y0, dtype=self.dtype_f, device=self.device)
        z_coords, segments, layer_ranges, layer_edges, mesh_z_step = build_structure_z_segments(
            self,
            znum=znum,
            z_step=z_step,
            duplicate_interfaces=duplicate_interfaces,
        )

        core_fields = {name: [] for name in requested_core}
        indices = [all_components.index(name) for name in requested_core]
        for layer_index, local_z in segments:
            coeffs, _ = field_coefficients_batched(self, layer_index, local_z)
            spatial = get_ifft_xline_batch(x_red, y0_red, coeffs[:, indices, :], self.G, dtype_c=self.dtype_c, device=self.device)
            layer_fields = {name: spatial[:, idx] for idx, name in enumerate(requested_core)}
            for name in requested_core:
                core_fields[name].append(layer_fields[name])

        for name in requested_core:
            core_fields[name] = torch.concatenate(core_fields[name], dim=0)

        Ex = core_fields.get('Ex')
        Ey = core_fields.get('Ey')
        Ez = core_fields.get('Ez')
        Hx = core_fields.get('Hx')
        Hy = core_fields.get('Hy')
        Hz = core_fields.get('Hz')
        E = [Ex, Ey, Ez]
        H = [Hx, Hy, Hz]
        if Ex is None and Ey is None and Ez is None:
            E = None
        if Hx is None and Hy is None and Hz is None:
            H = None
        return E, H, x_coords, z_coords, layer_ranges, layer_edges, mesh_z_step

    def Solve_FieldYZ(self, x0=0.0, y_coords=None, znum=32, z_step=None, components=('Ex',), duplicate_interfaces=True):
        all_components = ('Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz')
        requested_core = []
        for name in components:
            if name not in all_components:
                raise ValueError(f'Unknown field component: {name}')
            if name not in requested_core:
                requested_core.append(name)
        y_coords, y_red, x0_red = get_line_coordinates(self, 'y', y_coords, x0)
        x0 = torch.as_tensor(x0, dtype=self.dtype_f, device=self.device)
        z_coords, segments, layer_ranges, layer_edges, mesh_z_step = build_structure_z_segments(
            self,
            znum=znum,
            z_step=z_step,
            duplicate_interfaces=duplicate_interfaces,
        )

        core_fields = {name: [] for name in requested_core}
        indices = [all_components.index(name) for name in requested_core]
        for layer_index, local_z in segments:
            coeffs, _ = field_coefficients_batched(self, layer_index, local_z)
            spatial = get_ifft_yline_batch(x0_red, y_red, coeffs[:, indices, :], self.G, dtype_c=self.dtype_c, device=self.device)
            layer_fields = {name: spatial[:, idx] for idx, name in enumerate(requested_core)}
            for name in requested_core:
                core_fields[name].append(layer_fields[name])

        for name in requested_core:
            core_fields[name] = torch.concatenate(core_fields[name], dim=0)

        Ex = core_fields.get('Ex')
        Ey = core_fields.get('Ey')
        Ez = core_fields.get('Ez')
        Hx = core_fields.get('Hx')
        Hy = core_fields.get('Hy')
        Hz = core_fields.get('Hz')
        E = [Ex, Ey, Ez]
        H = [Hx, Hy, Hz]
        if Ex is None and Ey is None and Ez is None:
            E = None
        if Hx is None and Hy is None and Hz is None:
            H = None
        return E, H, y_coords, z_coords, layer_ranges, layer_edges, mesh_z_step

    # State serialization

    def _topology_signature(self):
        signature = []
        for layer_id in self.id_list:
            if layer_id[0] == 0:
                signature.append(('uniform',))
            elif layer_id[0] == 1:
                Nx, Ny = self.GridLayer_Nxy_list[layer_id[3]]
                signature.append(('grid', int(Nx), int(Ny)))
            else:
                signature.append(('fourier',))
        return tuple(signature)

    @staticmethod
    def _infer_state_device(state):
        if torch.is_tensor(state):
            return state.device
        if isinstance(state, dict):
            for value in state.values():
                device = obj._infer_state_device(value)
                if device is not None:
                    return device
        if isinstance(state, (list, tuple)):
            for value in state:
                device = obj._infer_state_device(value)
                if device is not None:
                    return device
        return None

    @classmethod
    def _from_exported_state(cls, state, restore_caches):
        if state.get('state_version') != STATE_VERSION:
            raise ValueError('Unsupported state version')

        meta = state['meta']
        runtime = state['runtime']
        layer_defs = state['layer_defs']

        inferred_device = cls._infer_state_device(runtime)
        device = inferred_device if inferred_device is not None else torch.device(meta.get('device', 'cpu'))

        restored = cls(
            meta['nG_requested'],
            meta['L1'],
            meta['L2'],
            meta['freq'],
            meta['theta'],
            meta['phi'],
            verbose=meta['verbose'],
            device=device,
            dtype_f=meta['dtype_f'],
            dtype_c=meta['dtype_c'],
        )

        for layer_def in layer_defs:
            if layer_def['kind'] == 'uniform':
                restored.Add_LayerUniform(layer_def['thickness'], layer_def['epsilon'])
            elif layer_def['kind'] == 'grid':
                restored.Add_LayerGrid(layer_def['thickness'], layer_def['Nx'], layer_def['Ny'])
            elif layer_def['kind'] == 'fourier':
                restored.Add_LayerFourier(layer_def['thickness'], copy.deepcopy(layer_def['params']))
            else:
                raise ValueError(f"Unknown layer type in saved state: {layer_def['kind']}")

        restored.Pscale = torch.as_tensor(runtime['Pscale'], dtype=restored.dtype_f, device=restored.device)
        restored.Gmethod = runtime['Gmethod']
        restored.setup_ready = runtime['setup_ready']
        restored.nG = int(runtime['nG'])
        restored.G = runtime['G']
        restored.Lk1 = runtime['Lk1']
        restored.Lk2 = runtime['Lk2']
        restored.kx = runtime['kx']
        restored.ky = runtime['ky']
        restored.normalization = runtime['normalization']
        restored.kp_list = runtime['kp_list']
        restored.q_list = runtime['q_list']
        restored.phi_list = runtime['phi_list']
        restored.Patterned_epinv_list = runtime['Patterned_epinv_list']
        restored.Patterned_ep2_list = runtime['Patterned_ep2_list']
        restored.GridLayer_epgrid_list = runtime['GridLayer_epgrid_list']
        restored.direction = runtime['direction']

        restored._a0 = None
        restored._bN = None
        if runtime['a0'] is not None:
            restored.a0 = runtime['a0']
        if runtime['bN'] is not None:
            restored.bN = runtime['bN']

        if restore_caches and 'caches' in state:
            caches = state['caches']
            restored._prefix_smat = caches['prefix_smat']
            restored._suffix_smat = caches['suffix_smat']
            restored._smat_cache_ready = caches['smat_cache_ready']
            restored._exterior_cache = caches['exterior_cache']
            restored._amplitude_cache_ready = caches['amplitude_cache_ready']
            restored._layer_amplitudes = caches['layer_amplitudes']
            restored._amplitude_cache_info = caches.get('amplitude_cache_info')
            restored._dirty_layers = set(caches['dirty_layers'])
        else:
            restored._dirty_layers = set()
            restored._invalidate_structure_caches(clear_dirty=True)

        return restored

    def ExportState(self, include_caches=False):
        layer_defs = []
        for layer_id in self.id_list:
            layer_index = layer_id[1]
            thickness = _clone_nested(self.thickness_list[layer_index])
            if layer_id[0] == 0:
                layer_defs.append({
                    'kind': 'uniform',
                    'thickness': thickness,
                    'epsilon': _clone_nested(self.Uniform_ep_list[layer_id[2]]),
                })
            elif layer_id[0] == 1:
                Nx, Ny = self.GridLayer_Nxy_list[layer_id[3]]
                layer_defs.append({
                    'kind': 'grid',
                    'thickness': thickness,
                    'Nx': int(Nx),
                    'Ny': int(Ny),
                })
            else:
                layer_defs.append({
                    'kind': 'fourier',
                    'thickness': thickness,
                    'params': copy.deepcopy(self.FourierLayer_params[layer_id[3]]),
                })

        state = {
            'state_version': STATE_VERSION,
            'topology_signature': self._topology_signature(),
            'meta': {
                'nG_requested': self.nG_requested,
                'L1': _clone_nested(self.L1),
                'L2': _clone_nested(self.L2),
                'freq': _clone_nested(self.freq),
                'theta': _clone_nested(self.theta),
                'phi': _clone_nested(self.phi),
                'verbose': self.verbose,
                'dtype_f': self.dtype_f,
                'dtype_c': self.dtype_c,
                'device': str(self.device),
            },
            'layer_defs': layer_defs,
            'runtime': {
                'Pscale': _clone_nested(self.Pscale),
                'Gmethod': self.Gmethod,
                'setup_ready': self.setup_ready,
                'nG': self.nG,
                'G': _clone_nested(getattr(self, 'G', None)),
                'Lk1': _clone_nested(getattr(self, 'Lk1', None)),
                'Lk2': _clone_nested(getattr(self, 'Lk2', None)),
                'kx': _clone_nested(getattr(self, 'kx', None)),
                'ky': _clone_nested(getattr(self, 'ky', None)),
                'normalization': _clone_nested(getattr(self, 'normalization', None)),
                'kp_list': _clone_nested(self.kp_list),
                'q_list': _clone_nested(self.q_list),
                'phi_list': _clone_nested(self.phi_list),
                'Patterned_epinv_list': _clone_nested(self.Patterned_epinv_list),
                'Patterned_ep2_list': _clone_nested(self.Patterned_ep2_list),
                'GridLayer_epgrid_list': _clone_nested(self.GridLayer_epgrid_list),
                'direction': self.direction,
                'a0': _clone_nested(self.a0),
                'bN': _clone_nested(self.bN),
            },
        }

        if include_caches:
            state['caches'] = {
                'smat_cache_ready': self._smat_cache_ready,
                'prefix_smat': _clone_nested(self._prefix_smat),
                'suffix_smat': _clone_nested(self._suffix_smat),
                'dirty_layers': sorted(self._dirty_layers),
                'exterior_cache': _clone_nested(self._exterior_cache),
                'amplitude_cache_ready': self._amplitude_cache_ready,
                'layer_amplitudes': _clone_nested(self._layer_amplitudes),
                'amplitude_cache_info': copy.deepcopy(self._amplitude_cache_info),
            }

        return state

    def RestoreState(self, state, restore_caches=False):
        if state.get('topology_signature') != self._topology_signature():
            raise ValueError('RestoreState requires an object with matching layer topology')
        restored = self.__class__._from_exported_state(state, restore_caches=restore_caches)
        self.__dict__.clear()
        self.__dict__.update(restored.__dict__)

    def SaveState(self, path, include_caches=False):
        torch.save(self.ExportState(include_caches=include_caches), path)

    @classmethod
    def LoadState(cls, path, map_location=None):
        state = torch.load(path, map_location=map_location, weights_only=False)
        return cls._from_exported_state(state, restore_caches='caches' in state)

    # Derived post-processing

    def _solve_absorption_density(self, which_layer, eps_imag, min_znum=2, z_step=None, Nxy=None):
        which_layer = self._validate_layer_index(which_layer)
        if Nxy is not None:
            Nx, Ny = int(Nxy[0]), int(Nxy[1])
        elif self.id_list[which_layer][0] == 1:
            Nx, Ny = self._default_grid_shape(which_layer)
            Nx = int(Nx)
            Ny = int(Ny)
        else:
            sample = eps_imag
            if isinstance(sample, dict):
                sample = next(iter(sample.values()))
            if isinstance(sample, (list, tuple)) and len(sample) == 3:
                sample = sample[0]

            sample = torch.as_tensor(sample, device=self.device)
            if sample.ndim == 2:
                Nx, Ny = int(sample.shape[0]), int(sample.shape[1])
            elif self.GridLayer_Nxy_list:
                Nx = max(shape[0] for shape in self.GridLayer_Nxy_list)
                Ny = max(shape[1] for shape in self.GridLayer_Nxy_list)
            else:
                Nx = max(self.nG, 64)
                Ny = max(self.nG, 64)

        def to_grid(component):
            tensor = torch.as_tensor(component, device=self.device)
            if torch.is_complex(tensor):
                tensor = torch.imag(tensor)
            tensor = tensor.to(dtype=self.dtype_f)
            if tensor.ndim == 0:
                return torch.ones((Nx, Ny), dtype=self.dtype_f, device=self.device) * tensor
            if tensor.shape != (Nx, Ny):
                raise ValueError(f'Absorption pattern must have shape ({Nx}, {Ny})')
            return tensor

        patterns = {}
        if isinstance(eps_imag, dict):
            for name, pattern in eps_imag.items():
                if isinstance(pattern, (list, tuple)):
                    if len(pattern) != 3:
                        raise ValueError('Absorption pattern must be isotropic or a length-3 diagonal tuple/list')
                    patterns[str(name)] = tuple(to_grid(component) for component in pattern)
                else:
                    grid = to_grid(pattern)
                    patterns[str(name)] = (grid, grid, grid)
        elif isinstance(eps_imag, (list, tuple)):
            if len(eps_imag) != 3:
                raise ValueError('Absorption pattern must be isotropic or a length-3 diagonal tuple/list')
            patterns['default'] = tuple(to_grid(component) for component in eps_imag)
        else:
            grid = to_grid(eps_imag)
            patterns['default'] = (grid, grid, grid)

        z_count = max(int(min_znum), 2)
        z_coords = build_layer_z_points(self, self.thickness_list[which_layer], count=z_count, z_step=z_step)
        E, _ = self.Solve_FieldXY(which_layer, z_coords, Nxy=(Nx, Ny), components=('Ex', 'Ey', 'Ez'))
        ex, ey, ez = E
        omega_real = torch.real(self.omega).to(dtype=self.dtype_f)
        densities = {}
        total_density = torch.zeros((z_coords.numel(), Nx, Ny), dtype=self.dtype_f, device=self.device)
        for name, (loss_x, loss_y, loss_z) in patterns.items():
            density = 0.5 * omega_real * (
                loss_x.unsqueeze(0) * torch.abs(ex) ** 2
                + loss_y.unsqueeze(0) * torch.abs(ey) ** 2
                + loss_z.unsqueeze(0) * torch.abs(ez) ** 2
            )
            density = torch.real(density)
            densities[name] = density
            total_density = total_density + density

        cell_area = torch.abs(self.L1[0] * self.L2[1] - self.L1[1] * self.L2[0]) * self.Pscale ** 2
        dA = cell_area / (Nx * Ny)
        return z_coords, densities, total_density, dA, Nx, Ny

    def Solve_AbsorptionLayer(self, which_layer, eps_imag, min_znum=2, z_step=None, Nxy=None, normalize=0):
        return self.Solve_AbsorptionLayerZ(
            which_layer,
            eps_imag,
            min_znum=min_znum,
            z_step=z_step,
            Nxy=Nxy,
            normalize=normalize,
        )['total']

    def Solve_AbsorptionLayerZ(self, which_layer, eps_imag, min_znum=2, z_step=None, Nxy=None, normalize=0):
        z_coords, densities, total_density, dA, Nx, Ny = self._solve_absorption_density(
            which_layer,
            eps_imag,
            min_znum=min_znum,
            z_step=z_step,
            Nxy=Nxy,
        )

        absorption_z = dA * torch.sum(total_density, dim=(-2, -1))
        total = torch.trapz(absorption_z, z_coords)

        per_pattern = {}
        for name, density in densities.items():
            pattern_z = dA * torch.sum(density, dim=(-2, -1))
            pattern_total = torch.trapz(pattern_z, z_coords)
            if normalize == 1:
                pattern_z = pattern_z * self.normalization
                pattern_total = pattern_total * self.normalization
            per_pattern[name] = {
                'total': torch.real(pattern_total),
                'absorption_z': torch.real(pattern_z),
            }

        if normalize == 1:
            absorption_z = absorption_z * self.normalization
            total = total * self.normalization

        return {
            'total': torch.real(total),
            'z_coords': z_coords,
            'absorption_z': torch.real(absorption_z),
            'per_pattern': per_pattern,
            'Nx': Nx,
            'Ny': Ny,
        }

    def Solve_AbsorptionLayerXY(self, which_layer, eps_imag, min_znum=2, z_step=None, Nxy=None, normalize=0):
        z_coords, densities, total_density, dA, Nx, Ny = self._solve_absorption_density(
            which_layer,
            eps_imag,
            min_znum=min_znum,
            z_step=z_step,
            Nxy=Nxy,
        )

        absorption_xy = torch.trapz(total_density, z_coords, dim=0)
        total = dA * torch.sum(absorption_xy)

        per_pattern = {}
        for name, density in densities.items():
            pattern_xy = torch.trapz(density, z_coords, dim=0)
            pattern_total = dA * torch.sum(pattern_xy)
            if normalize == 1:
                pattern_xy = pattern_xy * self.normalization
                pattern_total = pattern_total * self.normalization
            per_pattern[name] = {
                'total': torch.real(pattern_total),
                'absorption_xy': torch.real(pattern_xy),
            }

        if normalize == 1:
            absorption_xy = absorption_xy * self.normalization
            total = total * self.normalization

        return {
            'total': torch.real(total),
            'absorption_xy': torch.real(absorption_xy),
            'per_pattern': per_pattern,
            'Nx': Nx,
            'Ny': Ny,
        }

    def Solve_Absorption(self, eps_imag_layers, min_znum=2, z_step=None, Nxy=None, normalize=0):
        if isinstance(eps_imag_layers, dict):
            items = sorted(eps_imag_layers.items())
        else:
            if len(eps_imag_layers) != self.Layer_N:
                raise ValueError('Absorption layer sequence must have one entry per layer')
            items = [(layer_index, eps_imag) for layer_index, eps_imag in enumerate(eps_imag_layers)]

        per_layer = {}
        total = torch.zeros((), dtype=self.dtype_f, device=self.device)
        for layer_index, eps_imag in items:
            if eps_imag is None:
                continue
            layer_value = self.Solve_AbsorptionLayer(
                layer_index,
                eps_imag,
                min_znum=min_znum,
                z_step=z_step,
                Nxy=Nxy,
                normalize=normalize,
            )
            per_layer[int(layer_index)] = layer_value
            total = total + layer_value

        return {
            'total': total,
            'per_layer': per_layer,
        }

    def Volume_integral(self, which_layer, Mx, My, Mz, normalize=0):
        which_layer = self._validate_layer_index(which_layer)
        kp = self.kp_list[which_layer]
        q = self.q_list[which_layer]
        phi = self.phi_list[which_layer]

        if self.id_list[which_layer][0] == 0:
            epinv = 1.0 / self.Uniform_ep_list[self.id_list[which_layer][2]]
        else:
            epinv = self.Patterned_epinv_list[self.id_list[which_layer][2]]

        ai, bi = self.GetAmplitudes_noTranslate(which_layer)
        ab = torch.hstack((ai, bi))
        abMatrix = torch.outer(ab, torch.conj(ab))

        Mt = Matrix_zintegral(q, self.thickness_list[which_layer])
        abM = abMatrix * Mt

        Faxy = torch.matmul(torch.matmul(kp, phi), torch.diag(1.0 / self.omega / q))
        Faz1 = 1.0 / self.omega * torch.matmul(epinv, torch.diag(self.ky))
        Faz2 = -1.0 / self.omega * torch.matmul(epinv, torch.diag(self.kx))
        Faz = torch.matmul(torch.hstack((Faz1, Faz2)), phi)

        tmp1 = torch.vstack((Faxy, Faz))
        tmp2 = torch.vstack((-Faxy, Faz))
        F = torch.hstack((tmp1, tmp2))

        Mx = torch.as_tensor(Mx, dtype=F.dtype, device=self.device)
        My = torch.as_tensor(My, dtype=F.dtype, device=self.device)
        Mz = torch.as_tensor(Mz, dtype=F.dtype, device=self.device)
        Mzeros = torch.zeros_like(Mx)
        Mtotal = torch.vstack((
            torch.hstack((Mx, Mzeros, Mzeros)),
            torch.hstack((Mzeros, My, Mzeros)),
            torch.hstack((Mzeros, Mzeros, Mz)),
        ))

        tmp = torch.matmul(torch.matmul(torch.conj(torch.transpose(F, 0, 1)), Mtotal), F)
        val = torch.trace(torch.matmul(abM, tmp))

        if normalize == 1:
            val = val * self.normalization
        return val

    def Solve_ZStressTensorIntegral(self, which_layer):
        which_layer = self._validate_layer_index(which_layer)
        eh = self.Solve_FieldFourier(which_layer, 0.0)
        e = eh[0][0]
        h = eh[0][1]
        ex = e[0]
        ey = e[1]
        ez = e[2]

        hx = h[0]
        hy = h[1]
        hz = h[2]

        dz = (self.ky * hx - self.kx * hy) / self.omega

        if self.id_list[which_layer][0] == 0:
            dx = ex * self.Uniform_ep_list[self.id_list[which_layer][2]]
            dy = ey * self.Uniform_ep_list[self.id_list[which_layer][2]]
        else:
            exy = torch.hstack((-ey, ex))
            dxy = torch.matmul(self.Patterned_ep2_list[self.id_list[which_layer][2]], exy)
            dx = dxy[self.nG:]
            dy = -dxy[:self.nG]

        Tx = torch.sum(ex * torch.conj(dz) + hx * torch.conj(hz))
        Ty = torch.sum(ey * torch.conj(dz) + hy * torch.conj(hz))
        Tz = 0.5 * torch.sum(
            ez * torch.conj(dz)
            + hz * torch.conj(hz)
            - ey * torch.conj(dy)
            - ex * torch.conj(dx)
            - torch.abs(hx) ** 2
            - torch.abs(hy) ** 2
        )
        return torch.real(Tx), torch.real(Ty), torch.real(Tz)


def MakeKPMatrix(omega, layer_type, epinv, kx, ky, dtype_c=torch.complex128):
    nG = len(kx)

    Jk = torch.vstack((torch.diag(-ky), torch.diag(kx)))
    eye = torch.eye(2 * nG, dtype=dtype_c, device=kx.device)
    if layer_type == 0:
        JkkJT = torch.matmul(Jk, torch.transpose(Jk, 0, 1))
        kp = omega ** 2 * eye - epinv * JkkJT
    else:
        tmp = torch.matmul(Jk, epinv)
        kp = omega ** 2 * eye - torch.matmul(tmp, torch.transpose(Jk, 0, 1))
    return kp


def SolveLayerEigensystem_uniform(omega, kx, ky, epsilon, dtype_c=torch.complex128):
    nG = len(kx)
    q = torch.sqrt(torch.as_tensor(epsilon, dtype=dtype_c, device=kx.device) * omega ** 2 - kx ** 2 - ky ** 2)
    q = torch.where(torch.imag(q) < 0.0, -q, q)
    q = torch.concatenate((q, q))
    phi = torch.eye(2 * nG, dtype=dtype_c, device=kx.device)
    return q, phi


def SolveLayerEigensystem(omega, kx, ky, kp, ep2):
    k = torch.vstack((torch.diag(kx), torch.diag(ky)))
    kkT = torch.matmul(k, torch.transpose(k, 0, 1))
    M = torch.matmul(ep2, kp) - kkT

    q, phi = torch.linalg.eig(M)
    q = torch.sqrt(q)
    q = torch.where(torch.imag(q) < 0.0, -q, q)
    return q, phi


def TranslateAmplitudes(q, thickness, dz, ai, bi, dtype_f=torch.float64):
    thickness = torch.as_tensor(thickness, dtype=dtype_f, device=q.device)
    dz = torch.as_tensor(dz, dtype=dtype_f, device=q.device)
    aim = ai * torch.exp(1j * q * dz)
    bim = bi * torch.exp(1j * q * (thickness - dz))
    return aim, bim


def GetZPoyntingFlux(ai, bi, omega, kp, phi, q, byorder=0):
    n = ai.shape[0] // 2
    A = torch.matmul(torch.matmul(kp, phi), torch.diag(1.0 / omega / q))

    pa = torch.matmul(phi, ai)
    pb = torch.matmul(phi, bi)
    Aa = torch.matmul(A, ai)
    Ab = torch.matmul(A, bi)

    diff = 0.5 * (torch.conj(pb) * Aa - torch.conj(Ab) * pa)
    forward_xy = torch.real(torch.conj(Aa) * pa) + diff
    backward_xy = -torch.real(torch.conj(Ab) * pb) + torch.conj(diff)

    forward = forward_xy[:n] + forward_xy[n:]
    backward = backward_xy[:n] + backward_xy[n:]
    if byorder == 0:
        forward = torch.sum(forward)
        backward = torch.sum(backward)

    return forward, backward


def Matrix_zintegral(q, thickness, shift=1e-12):
    nG2 = q.shape[0]
    qi, qj = Gmeshgrid(q)

    qij = qj - torch.conj(qi) + torch.eye(nG2, dtype=q.dtype, device=q.device) * shift
    Maa = (torch.exp(1j * qij * thickness) - 1) / 1j / qij

    qij2 = qj + torch.conj(qi)
    Mab = (torch.exp(1j * qj * thickness) - torch.exp(-1j * torch.conj(qi) * thickness)) / 1j / qij2

    tmp1 = torch.vstack((Maa, Mab))
    tmp2 = torch.vstack((Mab, Maa))
    Mt = torch.hstack((tmp1, tmp2))
    return Mt


def Gmeshgrid(x):
    N = x.shape[0]
    qj = x.reshape(1, N).repeat(N, 1)
    qi = x.reshape(N, 1).repeat(1, N)
    return qi, qj
