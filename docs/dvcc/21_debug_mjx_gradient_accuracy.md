# MJX Gradient 정확도 문제

> 날짜: 2026-03-31

## 문제

MJX `jax.grad`로 ALLEX 60-DOF kp/kd 최적화 시도 → gradient 부정확.

## 검증 결과 (jax.grad vs finite difference)

| 규모             | 결과                               |
| -------------- | -------------------------------- |
| 1-DOF pendulum | ratio 0.99~1.00 정확               |
| ALLEX 1관절      | ratio 0.87~1.15 정확               |
| ALLEX 7관절 동시   | **BAD** (ratio 0.00~0.05, 부호 반대) |

float64 사용해도 개선 안 됨 — 수치 정밀도 문제 아님.

## 근본 원인

MJX 제약: `iterations=1` 필수 (iterations>1이면 while_loop → reverse-mode AD 불가).
`iterations=1` implicit solver가 60-DOF 규모에서 부정확 → forward pass 자체가 틀리면 미분도 틀림.

**딜레마**: iterations>1이면 JAX AD 불가, iterations=1이면 solver 부정확.

## 결론

- ALLEX 규모에서 MJX diffsim gradient-based 최적화 **사용 불가**
- 대안: grid search / Coordinate Descent / CMA-ES 등 derivative-free 방법
- Newton SolverMuJoCo (iterations=50+)로 정확한 forward sim + derivative-free 최적화 조합이 현실적

## 참고

- `mjx.put_data`에 `MjModel`(not `MjxModel`) 전달 필수 (MJX 3.6.0 `mj_isSparse` 버그)
- NeuralSim도 동일한 이유로 gradient 부정확 → 효과 없었음
