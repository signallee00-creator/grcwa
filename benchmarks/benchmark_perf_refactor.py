"""Small benchmark for cache reuse, selective reconstruction, and partial updates."""

import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import grcwa


def build_obj(grid0_override=None, grid1_override=None):
    obj = grcwa.obj(41, [0.2, 0.0], [0.0, 0.25], 1.1, torch.pi / 14, torch.pi / 18, verbose=0)
    obj.Add_LayerUniform(0.12, 1.0)
    obj.Add_LayerGrid(0.16, 32, 28)
    obj.Add_LayerUniform(0.08, 2.4)
    obj.Add_LayerGrid(0.2, 32, 28)
    obj.Add_LayerUniform(0.14, 1.3)
    obj.Init_Setup()
    obj.MakeExcitationPlanewave(0.8, 0.0, 0.3, 0.1, order=0, direction='forward')

    x_axis = torch.linspace(0.0, 1.0, 32, dtype=torch.float64)
    y_axis = torch.linspace(0.0, 1.0, 28, dtype=torch.float64)
    xx, yy = torch.meshgrid(x_axis, y_axis, indexing='ij')

    grid0 = torch.ones((32, 28), dtype=torch.float64) * 3.2
    grid0[(xx - 0.45) ** 2 + (yy - 0.52) ** 2 < 0.17 ** 2] = 1.1

    grid1 = torch.ones((32, 28), dtype=torch.float64) * 4.0
    grid1[torch.logical_and(xx > 0.2, xx < 0.7)] = 1.6

    if grid0_override is not None:
        grid0 = grid0_override
    if grid1_override is not None:
        grid1 = grid1_override

    obj.GridLayer_geteps(torch.concatenate((grid0.flatten(), grid1.flatten())))
    return obj, grid0, grid1


def bench(label, fn, repeat=5):
    values = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        values.append(time.perf_counter() - start)
    print(f"{label:<30} mean={statistics.mean(values):.4f}s min={min(values):.4f}s")


def main():
    obj, grid0, grid1 = build_obj()
    z_list = torch.linspace(0.0, 0.2, 11, dtype=torch.float64)

    bench("RT uncached", lambda: obj.RT_Solve(normalize=1))
    obj.BuildSMatrixCache()
    bench("RT cached", lambda: obj.RT_Solve(normalize=1))

    bench("Amplitude per-layer", lambda: [obj.GetAmplitudes_noTranslate(i) for i in range(obj.Layer_N)])
    obj.BuildAmplitudeCache()
    bench("Amplitude cached", lambda: [obj.GetAmplitudes_noTranslate(i) for i in range(obj.Layer_N)])

    bench("Field full grid", lambda: obj.Solve_FieldOnGrid(1, z_list))
    bench("Field selective XY", lambda: obj.Solve_FieldXY(1, z_list, components=('Ex',)))
    bench("Field XZ line", lambda: obj.Solve_FieldXZ(y0=0.0, znum=3, components=('Ex',)))

    new_grid = torch.roll(grid0, shifts=3, dims=0)
    baseline = obj.ExportState(include_caches=True)
    reusable, _, _ = build_obj()

    def full_rebuild():
        obj_local, _, _ = build_obj(grid0_override=new_grid, grid1_override=grid1)
        obj_local.RT_Solve(normalize=1)

    def partial_update():
        reusable.RestoreState(baseline, restore_caches=True)
        reusable.GridLayer_updateeps(1, new_grid, update_cache=False)
        reusable.UpdateSMatrixCache()
        reusable.RT_Solve(normalize=1)

    bench("Full rebuild", full_rebuild, repeat=3)
    bench("Single-layer update", partial_update, repeat=3)


if __name__ == '__main__':
    main()
