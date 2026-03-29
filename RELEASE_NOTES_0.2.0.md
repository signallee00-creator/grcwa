# grcwa 0.2.0

## Highlights

- Migrated the solver from the old numpy/backend/autograd stack to a torch-only implementation.
- Added explicit `device`, `dtype_f`, and `dtype_c` controls on `grcwa.obj`.
- Updated FFT, lattice, and RCWA kernels to use direct `torch.as_tensor(..., dtype=..., device=...)` handling.
- Ported examples and tests to PyTorch and refreshed the user-facing documentation.

## Breaking Changes

- `backend.py` and `primitives.py` were removed.
- Backend switching is no longer supported.
- PyTorch is now a required dependency.

## Recommended Defaults

- CPU stability-first:
  `device='cpu', dtype_f=torch.float64, dtype_c=torch.complex128`
- GPU speed-first:
  `device='cuda', dtype_f=torch.float32, dtype_c=torch.complex64`

## Verification

- `python -m pytest -q`
  Passed: `10 passed`

- Snapshot numerical checks
  - `example/ex1.py`: matches the original implementation numerically
  - `example/ex2.py`: matches the original implementation numerically

## Benchmark Notes

Benchmarks were run on the current workstation after the torch migration.

- CPU `float64/complex128`
  - Forward: about `0.177s`
  - Thickness gradient: about `0.181s`
  - Epsilon-grid gradient: about `0.231s`

- CPU `float32/complex64`
  - Faster forward and scalar-gradient performance than CPU `complex128`
  - Small numerical drift relative to CPU `complex128`

- GPU `float32/complex64`
  - Fastest measured configuration on the local RTX 3060 Ti
  - Not the default recommendation for CPU-only server deployment

## Migration Example

```python
obj = grcwa.obj(
    nG, L1, L2, freq, theta, phi,
    device='cpu',
    dtype_f=torch.float64,
    dtype_c=torch.complex128,
)
```
