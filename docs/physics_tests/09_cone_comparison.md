# Elliptic vs Pyramidal Cone 비교 결과

## 목적
MuJoCo의 두 가지 friction cone 모델의 행동 차이 비교.

## 배경
- **Elliptic cone**: 연속적인 원뿔 마찰 모델 (더 정확, 계산 비용 높음)
- **Pyramidal cone**: 다면체 근사 (빠르지만 방향 의존성 있음)

## 테스트 조건
- 구(R=0.3, density=10, mass=1.131kg) → 10N y방향 힘 → 박스 충돌
- mu_sphere=0.3, mu_ground=0.5 → geomean=0.387
- iterations=50, ls_iterations=10, dt=0.002

## 결과

| 항목 | Elliptic | Pyramidal |
|---|---|---|
| 구-박스 max침투 | 9.51mm | 9.15mm |
| 구-박스 max법선력 | **235.5N** | **62.5N** |
| 구-바닥 법선력 | **11.09N** (=mg) | **2.38N** (!!) |
| 구 최종 위치 (y) | -0.794 | -0.794 |
| 구 최종 속도 | ~0 | ~0 |
| 박스 최종 위치 (y) | 0.006 | 0.006 |

## 분석

### 주요 차이: 정적 법선력
- **Elliptic**: 구-바닥 법선력 = 11.09N (**정확히 mg**)
- **Pyramidal**: 구-바닥 법선력 = 2.38N (**mg의 21%밖에 안 됨**)

### Pyramidal의 법선력 문제
- Pyramidal cone은 마찰력을 다면체로 근사하면서 법선력 보고가 부정확
- 실제 contact force는 정상 작동 (위치/속도 동일)하지만 **efc_force 분해가 다름**
- Pyramidal에서는 마찰 성분이 법선 efc_address에 합산되는 방식이 달라서 단순 efc_force[0]이 법선력을 정확히 반영하지 않음

### 충돌 법선력
- Elliptic: 235.5N (충돌 순간 스파이크)
- Pyramidal: 62.5N (훨씬 낮음) — 같은 이유로 분해 방식 차이

### 최종 행동
- **위치와 속도는 동일** — 물리적 행동 자체는 차이 없음
- 차이는 **force 보고 방식**에 있음

## 권장
- **Elliptic cone 사용 권장** — 법선력 보고가 정확하고, 마찰 모델이 더 정밀
- Pyramidal은 성능이 중요한 대규모 시뮬레이션에서만 고려
- 법선력 측정이 필요한 경우 반드시 elliptic 사용

## 스크립트 (archived)
`jk_solver_examples/contacts/verify_cone.py` — 삭제됨, 결과만 보존
