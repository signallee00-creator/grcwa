# grcwa 한글 개요

`grcwa`는 주기 구조 광학 문제를 풀기 위한 RCWA 코드입니다. 현재 이 포크는 기존 `numpy/backend/autograd` 구조에서 `PyTorch` 중심 구조로 정리되어 있습니다.

## 현재 방향

- torch-only 구현
- `device`, `dtype_f`, `dtype_c`를 명시적으로 제어
- S-matrix cache, amplitude cache, partial grid-layer update 지원
- `FieldXY`, `FieldXZ`, `FieldYZ` 같은 dict 기반 field API 추가
- save/load state와 baseline 재사용 지원

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

## 현재 상태 한 줄 요약

field 쪽은 꽤 빨라졌고, 지금 남은 큰 병목은 `BuildSMatrixCache()`입니다.
