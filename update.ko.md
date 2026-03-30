# 업데이트 로그

날짜: 2026-03-29

## 1. torch-only 전환

기존 `numpy/backend/autograd` 구조에서 `torch-only` 구조로 옮겼습니다.

핵심:

- backend switching 제거
- 수치 코어를 직접 PyTorch tensor 연산으로 정리
- `device`, `dtype_f`, `dtype_c`를 `grcwa.obj(...)`에서 직접 제어
- CPU `float64` / `complex128`를 기본 안정 경로로 사용

이 기준점은 `v0.2.0` 태그로 남아 있습니다.

## 2. 그 이후 추가된 것

### 2.1 S-matrix cache

- `BuildSMatrixCache()`
- `UpdateSMatrixCache(changed_layers=None)`
- `GetExteriorAmplitudesCached()`

현재 cache는 다음을 저장합니다.

- prefix sweep `S(0, i)`
- suffix sweep `S(i, N-1)`

이전처럼 per-step S-matrix를 계속 저장하지 않고, 필요한 step은 build/update 중에 다시 계산합니다. 그래서 cache 메모리와 `include_caches=True` 저장 크기를 줄였습니다.

### 2.2 amplitude cache

- `BuildAmplitudeCache()`

현재는 exact-only입니다.

- step amplitude sweep는 쓰지 않음
- cached prefix/suffix를 이용한 exact interior solve로 각 layer amplitude를 구성

이유:

- 큰 구조에서 amplitude sweep가 자주 drift를 만들었고
- 결국 exact fallback으로 돌아가는 경우가 많았기 때문입니다.

### 2.3 메모리용 cache 정리

- `ClearSMatrixCache()`

이 함수는 무거운 prefix/suffix S-matrix만 지우고:

- cached exterior amplitude
- cached layer amplitudes

는 남겨둡니다.

그래서 보통 큰 구조에서는:

```python
obj.BuildSMatrixCache()
obj.BuildAmplitudeCache()
obj.ClearSMatrixCache()
```

이 흐름이 실용적입니다.

### 2.4 부분 레이어 업데이트

- `GridLayer_updateeps(which_layer, ep_grid, update_cache=True)`

바뀐 grid layer만 갱신하고, 영향을 받는 cache 구간만 다시 계산합니다.

### 2.5 field API 확장

- 기존 `Solve_FieldOnGrid(...)` 유지
- `components=(...)`로 필요한 성분만 복원 가능
- 단순화된 field API:
  - `Solve_FieldXY(...)`
  - `Solve_FieldXZLayer(...)`
  - `Solve_FieldYZLayer(...)`
  - `Solve_FieldXZ(...)`
  - `Solve_FieldYZ(...)`

현재 반환 형식:

- `Solve_FieldXY(...) -> E, H`
- `Solve_FieldXZLayer(...) -> E, H, x_coords, z_coords`
- `Solve_FieldYZLayer(...) -> E, H, y_coords, z_coords`
- `Solve_FieldXZ(...) -> E, H, x_coords, z_coords, layer_ranges, layer_edges, z_step`
- `Solve_FieldYZ(...) -> E, H, y_coords, z_coords, layer_ranges, layer_edges, z_step`

여기서:

- `E = [Ex, Ey, Ez]`
- `H = [Hx, Hy, Hz]`
- 요청하지 않은 성분은 `None`
- 전기장만 요청하면 `H`는 `None`

### 2.6 save / load

- `ExportState(include_caches=False)`
- `RestoreState(state, restore_caches=False)`
- `SaveState(path, include_caches=False)`
- `LoadState(path, map_location=None)`

권장:

- 가장 가벼운 baseline 저장: `include_caches=False`
- field 계산용 가벼운 cached state 저장:
  `BuildAmplitudeCache()` 후 `ClearSMatrixCache()` 하고 `include_caches=True`

### 2.7 가독성 정리

코드를 예전 `rcwa.py`를 읽던 흐름에 더 가깝게 다시 정리했습니다.

핵심:

- 임시 field helper 파일들을 없애고 field 계산을 다시 `grcwa/rcwa.py` 안으로 정리
- public field API는 legacy 스타일 `E, H` 반환 형태 유지
- line 좌표 처리와 absorption 입력 정리 쪽의 작은 helper 몇 개를 합쳐서
  함수 수를 줄임
- cache build/update에서 step S-matrix 생성 위치를 바로 보이게 해서
  소스만 읽어도 흐름을 따라가기 쉽게 정리

## 3. 벤치 요약

### current vs `v0.2.0`

대표 케이스:

- layers: `9`
- actual `nG`: `79`
- grid: `48 x 48`

대략:

- `RT_Solve()`: `1.2x` faster
- all-layer amplitudes: `73.4x` faster
- `Solve_FieldOnGrid()`: `59.8x` faster

### larger current-only cases

Case B:

- layers: `13`
- actual `nG`: `119`
- grid: `64 x 64`
- `BuildSMatrixCache`: about `1.47s`
- `S-matrix cache`: about `89.9 MB`
- `BuildAmplitudeCache`: about `0.057s`
- `amplitude + exterior cache`: about `0.10 MB`

Case C:

- layers: `17`
- actual `nG`: `159`
- grid: `72 x 72`
- `BuildSMatrixCache`: about `3.57s`
- `S-matrix cache`: about `209.9 MB`
- `BuildAmplitudeCache`: about `0.146s`
- `amplitude + exterior cache`: about `0.17 MB`

핵심 해석:

- field 쪽은 이미 꽤 가벼움
- 지금 남은 큰 병목은 `BuildSMatrixCache()`
- 메모리도 역시 S-matrix cache가 본체

## 4. 테스트 상태

- `python -m pytest -q`
- 현재 `24 passed`

## 5. 관련 파일

- [README.rst](C:/Users/LSH/Desktop/GRCWA/README.rst)
- [USAGE.md](C:/Users/LSH/Desktop/GRCWA/USAGE.md)
- [사용법.md](C:/Users/LSH/Desktop/GRCWA/사용법.md)
- [example/ex5_cache_and_fields.py](C:/Users/LSH/Desktop/GRCWA/example/ex5_cache_and_fields.py)
- [example/ex6_cache_save_load_update.py](C:/Users/LSH/Desktop/GRCWA/example/ex6_cache_save_load_update.py)
- [benchmarks/benchmark_scaled_compare.py](C:/Users/LSH/Desktop/GRCWA/benchmarks/benchmark_scaled_compare.py)
