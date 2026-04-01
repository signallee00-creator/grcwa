`materials/nk/` stores tabulated optical-constant decks.

Recommended structure:

- `materials/nk/nk_xx/`
- `materials/nk/nk_xy/`
- `materials/nk/nk_yy/`
- `materials/nk/nk_zz/`

Recommended file format:

```text
WL n k
112 1.4 0
113 1.401 0
...
```

Rules used by `materials.material`:

- `WL` is interpreted in `nm`
- query wavelengths are interpreted in `m`
- data are linearly interpolated in `n` and `k`
- outside the tabulated range, the first/last segment is linearly extrapolated
- if both `SiO2_index.txt` and dated files such as `SiO2_index_260401.txt`
  exist, the latest dated file is selected automatically
