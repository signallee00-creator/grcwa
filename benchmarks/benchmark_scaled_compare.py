"""Compare the current worktree against the tagged torch baseline and validate large field cuts."""

import importlib.util
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import grcwa as grcwa_current

_TEMP_DIRS = []


def load_git_package(ref, module_name):
    temp_dir = tempfile.TemporaryDirectory()
    _TEMP_DIRS.append(temp_dir)
    root = Path(temp_dir.name)

    listing = subprocess.check_output(
        ['git', 'ls-tree', '-r', '--name-only', ref, 'grcwa'],
        cwd=REPO_ROOT,
        text=True,
    ).splitlines()

    for relpath in listing:
        content = subprocess.check_output(
            ['git', 'show', f'{ref}:{relpath}'],
            cwd=REPO_ROOT,
        )
        dst = root / relpath
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(content)

    init_path = root / 'grcwa' / '__init__.py'
    spec = importlib.util.spec_from_file_location(
        module_name,
        init_path,
        submodule_search_locations=[str(init_path.parent)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def make_grids(nx, ny, npattern):
    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 1.0, ny)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    grids = []
    for index in range(npattern):
        base = 2.0 + 0.45 * index
        grid = np.ones((nx, ny), dtype=np.float64) * base
        if index % 3 == 0:
            mask = (xx - (0.32 + 0.07 * (index % 4))) ** 2 + (yy - 0.52) ** 2 < (0.13 + 0.01 * index) ** 2
            grid[mask] = 1.1 + 0.15 * index
        elif index % 3 == 1:
            mask = np.logical_and(xx > 0.18 + 0.04 * index, xx < 0.72)
            grid[mask] = 1.35 + 0.08 * index
        else:
            mask = np.logical_and(np.abs(xx - yy) < 0.10 + 0.01 * index, yy > 0.15)
            grid[mask] = 1.25 + 0.05 * index
        grids.append(grid)
    return grids


def build_obj(lib, nG, npattern, nx, ny):
    obj = lib.obj(nG, [0.22, 0.0], [0.0, 0.27], 1.05, np.pi / 13, np.pi / 17, verbose=0)
    obj.Add_LayerUniform(0.11, 1.0)
    grids = make_grids(nx, ny, npattern)
    for index, grid in enumerate(grids):
        obj.Add_LayerGrid(0.10 + 0.015 * index, nx, ny)
        obj.Add_LayerUniform(0.07 + 0.01 * (index % 3), 1.7 + 0.2 * index)
    obj.Init_Setup(Pscale=1.0, Gmethod=0)
    obj.MakeExcitationPlanewave(0.75, 0.0, 0.35, 0.2, order=0, direction='forward')
    obj.GridLayer_geteps(np.concatenate([grid.ravel() for grid in grids]))
    return obj


def bench(fn, repeat=3):
    times = []
    result = None
    for _ in range(repeat):
        start = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - start)
    return statistics.mean(times), result


def tensor_max_error(lhs, rhs):
    lhs = torch.as_tensor(np.asarray(lhs))
    rhs = torch.as_tensor(np.asarray(rhs))
    return torch.max(torch.abs(lhs - rhs)).item()


def tensor_tree_nbytes(value):
    if value is None:
        return 0
    if torch.is_tensor(value):
        return value.nelement() * value.element_size()
    if isinstance(value, (list, tuple)):
        return sum(tensor_tree_nbytes(item) for item in value)
    if isinstance(value, dict):
        return sum(tensor_tree_nbytes(item) for item in value.values())
    return 0


def format_rel_error(value):
    if value is None:
        return 'n/a'
    return f'{value:.3e}'


def format_abs_error(value):
    if value is None:
        return 'n/a'
    return f'{value:.3e}'


def compare_shared_apis_against_tag():
    tagged = load_git_package('v0.2.0', 'grcwa_v020')
    case = dict(nG=81, npattern=4, nx=48, ny=48)
    current = build_obj(grcwa_current, **case)
    baseline = build_obj(tagged, **case)
    layer_index = 3
    z_list = [0.0, float(current.thickness_list[layer_index]) * 0.5]

    rt_current, (R_current, T_current) = bench(lambda: current.RT_Solve(normalize=1))
    rt_baseline, (R_baseline, T_baseline) = bench(lambda: baseline.RT_Solve(normalize=1))

    amp_current, current_amp = bench(lambda: [current.GetAmplitudes(layer, 0.0) for layer in range(current.Layer_N)])
    amp_baseline, baseline_amp = bench(lambda: [baseline.GetAmplitudes(layer, 0.0) for layer in range(baseline.Layer_N)])

    field_current, current_field = bench(lambda: current.Solve_FieldOnGrid(layer_index, z_list), repeat=1)
    field_baseline, baseline_field = bench(lambda: baseline.Solve_FieldOnGrid(layer_index, z_list), repeat=1)

    print('Case A: current worktree vs git tag v0.2.0')
    print(f"  layers = {current.Layer_N}, nG = {current.nG}, grid = {case['nx']}x{case['ny']}")
    print(f"  RT current  = {rt_current:.4f}s")
    print(f"  RT v0.2.0   = {rt_baseline:.4f}s")
    print(f"  RT speedup  = {rt_baseline / rt_current:.1f}x")
    print(f"  |R diff|    = {abs(float(R_current) - float(R_baseline)):.3e}")
    print(f"  |T diff|    = {abs(float(T_current) - float(T_baseline)):.3e}")
    print(f"  Amp current = {amp_current:.4f}s")
    print(f"  Amp v0.2.0  = {amp_baseline:.4f}s")
    print(f"  Amp speedup = {amp_baseline / amp_current:.1f}x")
    print(f"  max |a0 diff| = {tensor_max_error(current_amp[0][0], baseline_amp[0][0]):.3e}")
    print(f"  max |b0 diff| = {tensor_max_error(current_amp[0][1], baseline_amp[0][1]):.3e}")
    print(f"  Field current = {field_current:.4f}s")
    print(f"  Field v0.2.0  = {field_baseline:.4f}s")
    print(f"  Field speedup = {field_baseline / field_current:.1f}x")
    print(f"  max |Ex diff| = {tensor_max_error(current_field[0][0][0], baseline_field[0][0][0]):.3e}")
    print(f"  max |Hy diff| = {tensor_max_error(current_field[0][1][1], baseline_field[0][1][1]):.3e}")


def benchmark_current_scaling(label, nG, npattern, nx, ny, znum=3):
    obj = build_obj(grcwa_current, nG=nG, npattern=npattern, nx=nx, ny=ny)
    grid_layers = [index for index, layer_id in enumerate(obj.id_list) if layer_id[0] == 1]
    layer_index = grid_layers[len(grid_layers) // 2]
    z_list = torch.linspace(0.0, float(obj.thickness_list[layer_index]), 5, dtype=torch.float64)

    rt_uncached, _ = bench(lambda: obj.RT_Solve(normalize=1), repeat=3)
    smat_build, _ = bench(obj.BuildSMatrixCache, repeat=1)
    smat_cache_mb = (tensor_tree_nbytes(obj._prefix_smat) + tensor_tree_nbytes(obj._suffix_smat)) / (1024 ** 2)
    rt_cached, _ = bench(lambda: obj.RT_Solve(normalize=1), repeat=3)

    amp_uncached, _ = bench(lambda: [obj.GetAmplitudes_noTranslate(i) for i in range(obj.Layer_N)], repeat=3)
    amp_build, _ = bench(obj.BuildAmplitudeCache, repeat=1)
    amp_cache_mb = (tensor_tree_nbytes(obj._layer_amplitudes) + tensor_tree_nbytes(obj._exterior_cache)) / (1024 ** 2)
    amp_cached, _ = bench(lambda: [obj.GetAmplitudes_noTranslate(i) for i in range(obj.Layer_N)], repeat=3)
    amp_info = obj._amplitude_cache_info

    xy_time, xy = bench(lambda: obj.Solve_FieldXY(layer_index, z_list, components=('Ex', 'Hy'), derived=('Pz', 'E2norm')), repeat=3)

    Ny = obj.GridLayer_Nxy_list[obj.id_list[layer_index][3]][1]
    iy = Ny // 3
    y0 = float(obj.L2[1]) * iy / Ny
    xz_time, xz = bench(lambda: obj.Solve_FieldXZ(y0=y0, znum=znum, components=('Ex', 'Hy'), derived=('Pz', 'E2norm')), repeat=3)
    start, stop = xz['layer_ranges'][layer_index]
    local_z = xz['z_coords'][start:stop] - xz['layer_edges'][layer_index]
    xy_slice = obj.Solve_FieldXY(layer_index, local_z, components=('Ex', 'Hy'), derived=('Pz', 'E2norm'))

    print()
    print(f"{label}: current-only scaling, layers={obj.Layer_N}, nG={obj.nG}, grid={nx}x{ny}")
    print(f"  RT uncached = {rt_uncached:.4f}s")
    print(f"  RT cached   = {rt_cached:.4f}s")
    print(f"  RT speedup  = {rt_uncached / rt_cached:.1f}x")
    print(f"  BuildSMatrixCache = {smat_build:.4f}s")
    print(f"  S-matrix cache = {smat_cache_mb:.2f} MB")
    print(f"  Amp uncached = {amp_uncached:.4f}s")
    print(f"  Amp cached   = {amp_cached:.4f}s")
    print(f"  Amp speedup  = {amp_uncached / amp_cached:.1f}x")
    print(f"  BuildAmplitudeCache = {amp_build:.4f}s")
    print(f"  amplitude+exterior cache = {amp_cache_mb:.2f} MB")
    print(f"  Amp cache mode = {amp_info['mode']}")
    print(f"  Amp aN abs/rel = {format_abs_error(amp_info['aN_abs_error'])} / {format_rel_error(amp_info['aN_rel_error'])}")
    print(f"  Amp bN abs/rel = {format_abs_error(amp_info['bN_abs_error'])} / {format_rel_error(amp_info['bN_rel_error'])}")
    print(f"  Field XY = {xy_time:.4f}s")
    print(f"  Field XZ = {xz_time:.4f}s")
    print(f"  XZ layer z-counts = {[stop_i - start_i for start_i, stop_i in xz['layer_ranges']]}")
    print(f"  XZ/XY max |Ex diff| = {torch.max(torch.abs(xz['Ex'][start:stop] - xy_slice['Ex'][:, :, iy])).item():.3e}")
    print(f"  XZ/XY max |Hy diff| = {torch.max(torch.abs(xz['Hy'][start:stop] - xy_slice['Hy'][:, :, iy])).item():.3e}")
    print(f"  XZ/XY max |Pz diff| = {torch.max(torch.abs(xz['Pz'][start:stop] - xy_slice['Pz'][:, :, iy])).item():.3e}")
    print(f"  XZ/XY max |E2norm diff| = {torch.max(torch.abs(xz['E2norm'][start:stop] - xy_slice['E2norm'][:, :, iy])).item():.3e}")


def main():
    compare_shared_apis_against_tag()
    benchmark_current_scaling('Case B', nG=121, npattern=6, nx=64, ny=64, znum=3)
    benchmark_current_scaling('Case C', nG=161, npattern=8, nx=72, ny=72, znum=3)


if __name__ == '__main__':
    main()
