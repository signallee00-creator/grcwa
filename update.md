# Update Log

Date: 2026-03-29

## 1. Torch-only migration baseline

This repo was first migrated from the original numpy/backend/autograd stack to a torch-only implementation.

Main points:

- Removed backend switching and the old backend abstraction.
- Moved the numerical core to direct PyTorch tensor operations.
- Added explicit `device`, `dtype_f`, and `dtype_c` handling on `grcwa.obj(...)`.
- Kept CPU `float64` / `complex128` as the safe default path.
- Verified numerical agreement against the original implementation on the original examples and tests.

This baseline was tagged as `v0.2.0`.

## 2. Post-v0.2.0 performance and workflow updates

After the torch-only migration, the following changes were added on top of `v0.2.0`.

### 2.1 S-matrix cache

- Added `BuildSMatrixCache()`
- Added `UpdateSMatrixCache(changed_layers=None)`
- Added cached exterior amplitude reuse through `GetExteriorAmplitudesCached()`
- `RT_Solve()` now uses the cached exterior path when the cache is available

The cache stores:

- prefix sweep `S(0, i)`
- suffix sweep `S(i, N-1)`

The per-interface step matrices are now recomputed on demand during cache
build/update instead of being stored persistently. This reduces cache
memory and also keeps `include_caches=True` state files smaller.

### 2.2 Partial patterned-layer updates

- Added `GridLayer_updateeps(which_layer, ep_grid, update_cache=True)`
- Added dirty-layer tracking
- Added partial cache rebuild of affected step/prefix/suffix ranges

This avoids full structure rebuilds when only a subset of grid layers changes.

### 2.3 State save / restore

- Added `ExportState(include_caches=False)`
- Added `RestoreState(state, restore_caches=False)`
- Added `SaveState(path, include_caches=False)`
- Added `LoadState(path, map_location=None)`

This supports:

- baseline object save / restore
- repeated parameter sweeps with a stable starting state
- reusing caches across runs

Recommended usage for large structures:

- use `include_caches=False` for the leanest saved baseline
- if cached field solves matter more than cached S-matrix reuse, build the
  amplitude cache, call `ClearSMatrixCache()`, and then save with
  `include_caches=True`

### 2.4 Field reconstruction API expansion

- Kept the legacy `Solve_FieldOnGrid(...)` API
- Added selective component reconstruction via `components=(...)`
- Simplified field APIs:
  - `Solve_FieldXY(...)`
  - `Solve_FieldXZLayer(...)`
  - `Solve_FieldYZLayer(...)`
  - `Solve_FieldXZ(...)`
  - `Solve_FieldYZ(...)`

Current field-return convention:

- `Solve_FieldXY(...) -> E, H`
- `Solve_FieldXZLayer(...) -> E, H, x_coords, z_coords`
- `Solve_FieldYZLayer(...) -> E, H, y_coords, z_coords`
- `Solve_FieldXZ(...) -> E, H, x_coords, z_coords, layer_ranges, layer_edges, z_step`
- `Solve_FieldYZ(...) -> E, H, y_coords, z_coords, layer_ranges, layer_edges, z_step`

where:

- `E = [Ex, Ey, Ez]`
- `H = [Hx, Hy, Hz]`
- unrequested components are `None`
- if only electric components are requested, `H` is returned as `None`

### 2.5 Structure-wide XZ / YZ sampling

`Solve_FieldXZ(...)` and `Solve_FieldYZ(...)` return whole-structure cuts, not just one-layer line cuts.

Sampling rules:

- `znum` is interpreted as `min_znum`
- thicker layers are sampled on an approximately uniform `z_step` mesh
- `z_step` can be passed explicitly
- returned metadata includes:
  - `z_coords`
  - `layer_ranges`
  - `layer_edges`
  - `z_step`

### 2.6 Exact-only amplitude cache

`BuildAmplitudeCache()` was simplified back to an exact cached solve path.

Current behavior:

- no step-based amplitude sweep is used
- amplitudes are built from cached prefix/suffix S-matrices through exact interior solves
- this is slower to build than a successful sweep would be, but much more stable

This change was made because large structures frequently pushed the sweep path into huge boundary mismatch and then fell back anyway.

After `BuildAmplitudeCache()`, the object can optionally release the heavy
S-matrix sweep via `ClearSMatrixCache()` while still keeping:

- cached exterior amplitudes for `RT_Solve()`
- cached per-layer amplitudes for field reconstruction

### 2.7 Absorption utilities

Real-space absorption helpers were added and intentionally left simple so they are easy to replace later if needed.

Added:

- `Solve_AbsorptionLayer(...)`
- `Solve_AbsorptionLayerZ(...)`
- `Solve_AbsorptionLayerXY(...)`
- `Solve_Absorption(...)`

Supported inputs:

- one isotropic `Nx x Ny` loss map
- diagonal anisotropic `(loss_x, loss_y, loss_z)`
- multiple named patterns in one layer via dict
- multiple layers via dict or layer-aligned sequence

The current implementation evaluates absorption from real-space `FieldXY(...)` samples.

### 2.8 Readability cleanup

The solver flow was cleaned up again to stay closer to the older `rcwa.py`
style and make the code easier to read directly.

Main points:

- removed the temporary field helper files and moved the field logic back into
  `grcwa/rcwa.py`
- kept the public field APIs in legacy-style `E, H` form
- reduced several tiny helper functions by merging line-coordinate handling and
  simplifying the absorption input path
- inlined S-matrix step construction where it is used so cache build/update is
  easier to follow when reading the source

## 3. Examples and tests added

### Examples

- `example/ex5_cache_and_fields.py`
  - cache-aware field reconstruction
  - XY / XZ usage
  - state save / load

- `example/ex6_cache_save_load_update.py`
  - build deck
  - build caches
  - cached `RT_Solve()`
  - save and load state
  - same-epsilon partial update
  - changed-epsilon partial update
  - compare partial update against full rebuild

### Benchmarks

- `benchmarks/benchmark_perf_refactor.py`
- `benchmarks/benchmark_scaled_compare.py`

### Tests

Tests were added for:

- S-matrix cache correctness
- partial patterned-layer updates
- state save / restore
- selective field reconstruction
- structure-wide XZ / YZ consistency
- mesh-like `z` sampling
- absorption layer and multi-layer accumulation

Current local test result:

- `python -m pytest -q`
- `24 passed`

## 4. Current benchmark summary

### 4.1 Current worktree vs `v0.2.0`

Representative case:

- layers: `9`
- actual `nG`: `79`
- grid: `48 x 48`

Measured on the current workstation:

- `RT_Solve()`:
  - current: about `0.14s`
  - `v0.2.0`: about `0.17s`
  - about `1.2x` faster

- all-layer `GetAmplitudes(...)`:
  - current: about `0.022s`
  - `v0.2.0`: about `1.58s`
  - about `73.4x` faster

- `Solve_FieldOnGrid(...)`:
  - current: about `0.0039s`
  - `v0.2.0`: about `0.235s`
  - about `59.8x` faster

Numerical agreement stayed very tight:

- `|R diff|`, `|T diff|` around `1e-15`
- field differences around `1e-14`

### 4.2 Larger current-only scaling cases

Case B:

- layers: `13`
- actual `nG`: `119`
- grid: `64 x 64`

Measured:

- `BuildSMatrixCache`: about `1.47s`
- `RT uncached -> cached`: about `37.1x`
- `BuildAmplitudeCache`: about `0.057s`
- `Amplitude uncached -> cached lookup`: about `3215.8x`
- `FieldXY`: about `0.0058s`
- `FieldXZ`: about `0.0300s`
- `S-matrix cache`: about `89.9 MB`
- `amplitude + exterior cache`: about `0.10 MB`

Case C:

- layers: `17`
- actual `nG`: `159`
- grid: `72 x 72`

Measured:

- `BuildSMatrixCache`: about `3.57s`
- `RT uncached -> cached`: about `53.1x`
- `BuildAmplitudeCache`: about `0.146s`
- `Amplitude uncached -> cached lookup`: about `8393.6x`
- `FieldXY`: about `0.0073s`
- `FieldXZ`: about `0.0471s`
- `S-matrix cache`: about `209.9 MB`
- `amplitude + exterior cache`: about `0.17 MB`

Field consistency checks for the new XZ mesh path remained near machine precision:

- `XZ` vs `XY` slice differences around `1e-15`

## 5. Current remaining bottlenecks

Representative Python profiler run on a larger case:

- layers: `17`
- actual `nG`: `159`
- grid: `72 x 72`

Main observation:

- the dominant remaining cost is `BuildSMatrixCache()`
- for large `nG` and large layer counts, the dominant remaining memory cost is also the prefix/suffix S-matrix cache itself

Profile summary:

- `BuildSMatrixCache()` total: about `4.18s`
- most of that time is in:
  - `compose_smatrix(...)`
  - `make_step_smatrix(...)`
  - `torch.matmul`
  - `torch.linalg.solve`

By comparison:

- `FieldXY(...)`: about `0.01s`
- `FieldXZ(...)`: about `0.056s`

Interpretation:

- once the cache exists, field solves are relatively cheap
- the real remaining hotspot is building the structure cache itself
- this is especially important for large `nG` and large layer counts
- a practical memory-aware workflow is:
  - `BuildSMatrixCache()`
  - `BuildAmplitudeCache()`
  - `ClearSMatrixCache()`
  - keep only the lightweight amplitude/exterior cache for repeated field queries or save/load

## 6. Recommendation on the next git push

Current judgment:

- this is a good checkpoint
- but it is not yet the final “performance cleanup is done” point

Reason:

- there is still a clear remaining bottleneck in `BuildSMatrixCache()`
- the codebase is already substantially improved and test-covered
- the save/load/update workflow is in place and working

Suggested push strategy:

- If the goal is a safe checkpoint of all current functionality, it is reasonable to push now.
- If the goal is a cleaner performance milestone, do one more patch first focused on `BuildSMatrixCache()` before pushing.

Recommended next patch target:

- optimize or restructure `BuildSMatrixCache()`
- only after that, cut the next performance-focused commit or release
