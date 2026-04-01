# grcwa 한글 개요

`grcwa`는 주기 구조 광학 문제를 풀기 위한 RCWA 코드입니다. 현재 이 포크는 기존 `numpy/backend/autograd` 구조에서 `PyTorch` 중심 구조로 정리되어 있습니다.

## 현재 방향

- torch-only 구현
- `device`, `dtype_f`, `dtype_c`를 명시적으로 제어
- S-matrix cache, amplitude cache, partial grid-layer update 지원
- `Solve_FieldXY`, `Solve_FieldXZ`, `Solve_FieldYZ` 같은 legacy-style `E, H` field API 지원
- save/load state와 baseline 재사용 지원
- 루트 helper 패키지 `patterns/`, `materials/` 지원

## 추천 기본 설정

CPU 위주라면:

```python
obj = grcwa.obj(
    nG, L1, L2, freq, theta, phi,
    device='cpu',
    dtype_f=torch.float64,
    dtype_c=torch.complex128,
)
```

## 메모리 관점 핵심

큰 구조에서는 무거운 건 amplitude cache가 아니라 S-matrix cache입니다.

- `BuildSMatrixCache()`는 prefix/suffix S-matrix를 들고 있어서 메모리를 많이 씁니다.
- `BuildAmplitudeCache()`는 layer별 amplitude와 exterior amplitude만 저장하므로 훨씬 가볍습니다.

그래서 반복 field 조회가 목적이면 보통 이렇게 쓰는 게 좋습니다.

```python
obj.BuildSMatrixCache()
obj.BuildAmplitudeCache()
obj.ClearSMatrixCache()
```

이후에도:

- `RT_Solve()`는 exterior cache를 그대로 사용
- `GetAmplitudes*()`와 field reconstruction은 amplitude cache를 그대로 사용

## 문서

- 영어 빠른 사용법: [USAGE.md](C:/Users/LSH/Desktop/GRCWA/USAGE.md)
- 한글 사용법: [사용법.md](C:/Users/LSH/Desktop/GRCWA/사용법.md)
- 변경 로그: [update.md](C:/Users/LSH/Desktop/GRCWA/update.md), [update.ko.md](C:/Users/LSH/Desktop/GRCWA/update.ko.md)

## 예제

- [example/ex5_cache_and_fields.py](C:/Users/LSH/Desktop/GRCWA/example/ex5_cache_and_fields.py)
- [example/ex6_cache_save_load_update.py](C:/Users/LSH/Desktop/GRCWA/example/ex6_cache_save_load_update.py)

## patterns / materials

`grcwa/` 바깥에 구조 생성용 `patterns/`, 광상수 데이터용 `materials/`를 같이 둡니다.

`patterns/` 예:

```python
from patterns import pattern_xx, nm
import torch

pat = pattern_xx(
    size_x=400 * nm,
    size_y=400 * nm,
    Nx=128,
    Ny=128,
    lamb0=1550 * nm,
    CRA=0.0,
    azimuth=0.0,
    eps_bg=1.0,
    eps_obj=3.48**2,
    radius=90 * nm,
    dtype_f=torch.float64,
    dtype_c=torch.complex128,
    device='cpu',
)

obj.GridLayer_geteps(pat.flatten())
```

여러 patterned layer면:

```python
obj.GridLayer_geteps(torch.cat([pat1.flatten(), pat2.flatten()]))
```

`materials/` 예:

```python
from materials import material
import torch

mat = material(
    1550e-9,
    dtype_f=torch.float64,
    dtype_c=torch.complex128,
    device='cpu',
)

eps_sio2 = mat.eps_SiO2_xx()
eps_old = mat.eps_SiO2_xx(filename='SiO2_index.txt')
```

기본 규칙:

- 파일 형식은 `WL n k`
- `WL`은 `nm`
- 코드에서 넣는 파장은 `m`
- `n`, `k`는 선형 보간 + 끝점 외삽
- `eps = (n + i k)^2`
- `SiO2_index.txt`와 `SiO2_index_260401.txt`가 같이 있으면 최신 dated 파일을 기본 사용

## 현재 상태 한 줄 요약

field 쪽은 꽤 빨라졌고, 지금 남은 큰 병목은 `BuildSMatrixCache()`입니다.
