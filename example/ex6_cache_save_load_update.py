"""Save/load and partial grid-layer updates with cache reuse."""

import os
import sys
import tempfile
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import grcwa


def make_circle_grid(nx, ny, eps_hi, eps_lo, radius, x_shift=0.0):
    x_axis = torch.linspace(0.0, 1.0, nx, dtype=torch.float64)
    y_axis = torch.linspace(0.0, 1.0, ny, dtype=torch.float64)
    xx, yy = torch.meshgrid(x_axis, y_axis, indexing='ij')
    grid = torch.ones((nx, ny), dtype=torch.float64) * eps_hi
    mask = (xx - (0.5 + x_shift)) ** 2 + (yy - 0.5) ** 2 < radius ** 2
    grid[mask] = eps_lo
    return grid


def build_deck(grid0, grid1):
    obj = grcwa.obj(81, [0.22, 0.0], [0.0, 0.24], 1.0, torch.pi / 14, torch.pi / 19, verbose=1)
    obj.Add_LayerUniform(0.10, 1.0)
    obj.Add_LayerGrid(0.14, grid0.shape[0], grid0.shape[1])
    obj.Add_LayerUniform(0.08, 2.1)
    obj.Add_LayerGrid(0.18, grid1.shape[0], grid1.shape[1])
    obj.Add_LayerUniform(0.11, 1.4)
    obj.Init_Setup(Pscale=1.0, Gmethod=0)
    obj.MakeExcitationPlanewave(0.8, 0.0, 0.3, 0.2, order=0, direction='forward')
    obj.GridLayer_geteps(torch.concatenate((grid0.flatten(), grid1.flatten())))
    return obj


def timed(label, fn):
    start = time.perf_counter()
    out = fn()
    elapsed = time.perf_counter() - start
    print(f'{label:<32} {elapsed:.4f}s')
    return out, elapsed


def print_rt(label, rt_pair):
    R, T = rt_pair
    print(f'{label:<32} R={R.item():.12f} T={T.item():.12f} R+T={(R + T).item():.12f}')


def main():
    grid0 = make_circle_grid(72, 72, eps_hi=4.0, eps_lo=1.2, radius=0.18)
    grid1 = make_circle_grid(72, 72, eps_hi=3.4, eps_lo=1.0, radius=0.15, x_shift=0.08)
    changed_grid1 = torch.roll(grid1, shifts=5, dims=0)

    obj = build_deck(grid0, grid1)
    _, _ = timed('Build S-matrix cache', obj.BuildSMatrixCache)
    _, _ = timed('Build amplitude cache', obj.BuildAmplitudeCache)
    baseline_rt, _ = timed('Baseline cached RT', lambda: obj.RT_Solve(normalize=1))
    print_rt('Baseline cached RT', baseline_rt)

    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as handle:
        state_path = handle.name

    try:
        _, _ = timed('Save cached state', lambda: obj.SaveState(state_path, include_caches=True))

        restored, _ = timed('Load cached state', lambda: grcwa.obj.LoadState(state_path, map_location='cpu'))
        restored_rt, _ = timed('Loaded cached RT', lambda: restored.RT_Solve(normalize=1))
        print_rt('Loaded cached RT', restored_rt)
        print('Loaded vs baseline dR =', abs((restored_rt[0] - baseline_rt[0]).item()))
        print('Loaded vs baseline dT =', abs((restored_rt[1] - baseline_rt[1]).item()))

        same_eps_obj, _ = timed('Reload cached state', lambda: grcwa.obj.LoadState(state_path, map_location='cpu'))
        _, _ = timed('Same-eps partial update', lambda: same_eps_obj.GridLayer_updateeps(3, grid1, update_cache=True))
        same_rt, _ = timed('Same-eps cached RT', lambda: same_eps_obj.RT_Solve(normalize=1))
        print_rt('Same-eps cached RT', same_rt)
        print('Same-eps dR =', abs((same_rt[0] - baseline_rt[0]).item()))
        print('Same-eps dT =', abs((same_rt[1] - baseline_rt[1]).item()))

        changed_obj, _ = timed('Reload cached state', lambda: grcwa.obj.LoadState(state_path, map_location='cpu'))
        _, _ = timed('Changed-eps partial update', lambda: changed_obj.GridLayer_updateeps(3, changed_grid1, update_cache=True))
        changed_rt, _ = timed('Changed-eps cached RT', lambda: changed_obj.RT_Solve(normalize=1))
        print_rt('Changed-eps cached RT', changed_rt)
        print('Changed-vs-baseline dR =', abs((changed_rt[0] - baseline_rt[0]).item()))
        print('Changed-vs-baseline dT =', abs((changed_rt[1] - baseline_rt[1]).item()))

        rebuilt_obj, _ = timed('Full rebuild changed deck', lambda: build_deck(grid0, changed_grid1))
        rebuilt_obj.BuildSMatrixCache()
        rebuilt_obj.BuildAmplitudeCache()
        rebuilt_rt, _ = timed('Changed-eps full RT', lambda: rebuilt_obj.RT_Solve(normalize=1))
        print_rt('Changed-eps full RT', rebuilt_rt)
        print('Partial-vs-full changed dR =', abs((changed_rt[0] - rebuilt_rt[0]).item()))
        print('Partial-vs-full changed dT =', abs((changed_rt[1] - rebuilt_rt[1]).item()))
    finally:
        os.remove(state_path)


if __name__ == '__main__':
    main()
