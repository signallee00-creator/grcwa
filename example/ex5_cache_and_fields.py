"""Cache-aware field reconstruction, line cuts, and state reuse."""

import os
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import grcwa


L1 = [0.2, 0.0]
L2 = [0.0, 0.2]
nG = 61
freq = 1.0
theta = torch.pi / 12
phi = 0.0
Nx = 80
Ny = 80

obj = grcwa.obj(nG, L1, L2, freq, theta, phi, verbose=1)
obj.Add_LayerUniform(0.12, 1.0)
obj.Add_LayerGrid(0.18, Nx, Ny)
obj.Add_LayerUniform(0.15, 2.0)
obj.Init_Setup()

x0 = torch.linspace(0.0, 1.0, Nx, dtype=torch.float64)
y0 = torch.linspace(0.0, 1.0, Ny, dtype=torch.float64)
x, y = torch.meshgrid(x0, y0, indexing='ij')

epgrid = torch.ones((Nx, Ny), dtype=torch.float64) * 4.0
hole = (x - 0.5) ** 2 + (y - 0.5) ** 2 < 0.2 ** 2
epgrid[hole] = 1.0

obj.GridLayer_geteps(epgrid)
obj.MakeExcitationPlanewave(1.0, 0.0, 0.0, 0.0, order=0, direction='forward')

obj.BuildSMatrixCache()
obj.BuildAmplitudeCache()
obj.ClearSMatrixCache()

R, T = obj.RT_Solve(normalize=1)
print('R =', R.item(), 'T =', T.item(), 'R + T =', (R + T).item())

z_list = torch.linspace(0.0, 0.18, 9, dtype=torch.float64)
field_xy_e, field_xy_h = obj.Solve_FieldXY(1, z_list, components=('Ex',))
field_xz_e, field_xz_h, x_coords, z_coords, _, _, _ = obj.Solve_FieldXZ(y0=0.0, znum=3, components=('Ex', 'Hy'))
field_xz_layer_e, field_xz_layer_h, _, _ = obj.Solve_FieldXZLayer(1, z_list, y0=0.0, components=('Ex', 'Hy'))

print('XY Ex shape =', tuple(field_xy_e[0].shape))
print('XY H is None =', field_xy_h is None)
print('XZ Ex shape =', tuple(field_xz_e[0].shape))
print('XZ Hy shape =', tuple(field_xz_h[1].shape))
print('XZ layer Ex shape =', tuple(field_xz_layer_e[0].shape))
print('x_coords shape =', tuple(x_coords.shape), 'z_coords shape =', tuple(z_coords.shape))

with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as handle:
    path = handle.name

obj.SaveState(path, include_caches=True)
restored = grcwa.obj.LoadState(path)
restored_field = restored.Solve_FieldXZ(y0=0.0, znum=3, components=('Ex',))
print('Restored XZ Ex shape =', tuple(restored_field[0][0].shape))
os.remove(path)
