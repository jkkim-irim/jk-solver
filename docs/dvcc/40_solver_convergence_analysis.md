# Solver ���렴성 분석

## 테스트 조건

- 구(R=0.3, density=10) → 10N 힘으로 박스(half=0.5)에 충돌
- ke=10000, kd=100, cone=elliptic, impratio=1.0, dt=0.002

## iterations sweep (ls_iterations=10 고정)

| iterations | max침투(mm) | max법선력(N) | ground NF(N) | z진동(mm) |
|---|---|---|---|---|
| 1 | 9.30 | 245.0 | 11.09 | 0.004 |
| 2 | 9.51 | 235.5 | 11.09 | 0.009 |
| **5** | **9.51** | **235.5** | **11.09** | 0.016 |
| 50 | 9.51 | 235.5 | 11.09 | 0.016 |
| 200 | 9.51 | 235.5 | 11.09 | 0.016 |

## ls_iterations sweep (iterations=50 고정)

| ls_iterations | max침투(mm) | max법선력(N) | ground NF(N) |
|---|---|---|---|
| 1 | 9.38 | 260.1 | 11.09 |
| **5** | **9.51** | **235.5** | **11.09** |
| 10 | 9.51 | 235.5 | 11.09 |
| 30 | 9.51 | 235.5 | 11.09 |

## 결론

- **iterations ≥ 2이면 수렴** (단순 접촉 시나리오)
- **ls_iterations ≥ 5이면 수렴**
- 다관절 로봇 등 복잡한 시스템에서는 더 많은 iterations 필요할 수 있음
- 현재 설정: iterations=5, ls_iterations=5 (단순 접촉 테스트에 충분)
- Elliptic cone 사용 권장 (pyramidal은 레거시, constraint 수 2배)

## solver conv 모니터링

tkinter StatusWindow 우측 패널에서 실시간 확인:
- `niter`: 실제 수행된 iteration 수
- `nacon`: active contact 수
- `nefc`: constraint 수 (≈ nacon × 3)
- `niter >= max_iter`이면 빨간색 표시 (수렴 실패 경고)
