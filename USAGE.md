# grcwa Usage

## Quick Start

```python
import torch
import grcwa

obj = grcwa.obj(
    101,
    [0.2, 0.0],
    [0.0, 0.2],
    1.0,
    torch.pi / 10,
    0.0,
    verbose=0,
    dtype_f=torch.float64,
    dtype_c=torch.complex128,
)

obj.Add_LayerUniform(0.1, 1.0)
obj.Add_LayerGrid(0.2, 100, 100)
obj.Add_LayerUniform(0.1, 1.5)

obj.Init_Setup()
obj.MakeExcitationPlanewave(1.0, 0.0, 0.0, 0.0, order=0, direction='forward')
obj.GridLayer_geteps(ep_grid)
```

## Core Solves

```python
R, T = obj.RT_Solve(normalize=1)
ai, bi = obj.GetAmplitudes(which_layer, z_offset)
E, H = obj.Solve_FieldOnGrid(which_layer, z_offset)
```

## Cache Workflow

```python
obj.BuildSMatrixCache()
obj.BuildAmplitudeCache()
```

If repeated field queries matter more than keeping the full S-matrix sweep in memory:

```python
obj.ClearSMatrixCache()
```

After that:

- `RT_Solve()` still uses the cached exterior amplitudes
- `GetAmplitudes*()` and field reconstruction still use the cached layer amplitudes
- `GridLayer_updateeps(..., update_cache=True)` will rebuild the S-matrix cache when needed

## Field APIs

Legacy API:

```python
E, H = obj.Solve_FieldOnGrid(which_layer, z_offset, components=('Ex', 'Hy'))
```

Dict-based APIs:

```python
xy = obj.Solve_FieldXY(which_layer, z_list, components=('Ex', 'Hy'), derived=('Pz', 'E2norm'))
xz = obj.Solve_FieldXZ(y0=0.0, znum=3, components=('Ex', 'Hy'))
yz = obj.Solve_FieldYZ(x0=0.0, znum=3, components=('Ex',))
```

Notes:

- `Solve_FieldXZ()` and `Solve_FieldYZ()` return whole-structure cuts
- `Solve_FieldXZLayer()` and `Solve_FieldYZLayer()` return single-layer line cuts
- integer `znum` means the minimum samples per layer
- thicker layers follow the same approximate `z_step`
- pass `z_step=...` to control structure-wide spacing directly

Derived keys:

- `Px`, `Py`, `Pz`
- `E2norm = |Ex|^2 + |Ey|^2 + |Ez|^2`
- `Pnorm = sqrt(Px^2 + Py^2 + Pz^2)`

## Partial Layer Updates

```python
obj.GridLayer_updateeps(which_layer, new_ep_grid, update_cache=False)
obj.UpdateSMatrixCache()
```

This updates only the affected cache ranges instead of rebuilding every patterned layer from scratch.

## Save and Load

Lean baseline:

```python
obj.SaveState('baseline.pt', include_caches=False)
restored = grcwa.obj.LoadState('baseline.pt')
```

Lightweight cached state:

```python
obj.BuildSMatrixCache()
obj.BuildAmplitudeCache()
obj.ClearSMatrixCache()
obj.SaveState('baseline_light.pt', include_caches=True)
```

This keeps the small amplitude/exterior caches without saving the heavy prefix/suffix S-matrix sweep.

## Benchmarks and Examples

- [example/ex5_cache_and_fields.py](C:/Users/LSH/Desktop/GRCWA/example/ex5_cache_and_fields.py)
- [example/ex6_cache_save_load_update.py](C:/Users/LSH/Desktop/GRCWA/example/ex6_cache_save_load_update.py)
- [benchmarks/benchmark_scaled_compare.py](C:/Users/LSH/Desktop/GRCWA/benchmarks/benchmark_scaled_compare.py)
- [update.md](C:/Users/LSH/Desktop/GRCWA/update.md)
