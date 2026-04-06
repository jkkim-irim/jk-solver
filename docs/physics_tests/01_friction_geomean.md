# Friction Geometric Mean Override 검증 결과

## 목적

MuJoCo 기본 마찰 조합(max)을 geometric mean(sqrt(mu1*mu2))으로 오버라이드한 효과 검증.

## 배경

- MuJoCo: `max(mu1, mu2)` — 한쪽이 높으면 무조건 높음 (비현실적)
- Newton semi-implicit: `(mu1 + mu2) / 2` — 산술평균
- 현실: geometric mean `sqrt(mu1 * mu2)` — 한쪽이 0이면 결과도 0 (ODE, Bullet, PhysX 방식)

## 구현

- `jk_solver_kernels.py`에 `override_contact_friction_geomean` Warp 커널 추가
- `SolverMuJoCoGeomean` 클래스 (SolverMuJoCo 상속)
- `use_mujoco_contacts=False` + Newton CollisionPipeline 사용
- `_convert_contacts_to_mjwarp()` 후, `_mujoco_warp_step()` 전에 friction 덮어쓰기

## 테스트 조건

- SPHERE_MU=0.3, GROUND_MU=0.8
- 구: R=0.3m, density=10, mass=1.131kg
- mg=11.09N, 가한 힘=10N (y방향)

## 결과

### Contact friction 값 (solver 내부)

| 방식                 | mu 계산         | contact.friction[0] |
| ------------------ | ------------- | ------------------- |
| Original (max)     | max(0.3, 0.8) | **0.8000**          |
| Override (geomean) | sqrt(0.3*0.8) | **0.4899**          |

### 운동 검증 (1초 후 y방향 속도)

| 방식      | mu   | 마찰력(N) | net(N) | 이론 vel(m/s) | 실측 vel(m/s) | 오차   |
| ------- | ---- | ------ | ------ | ----------- | ----------- | ---- |
| max     | 0.80 | 8.88   | 1.12   | 0.99        | **1.005**   | 1.2% |
| geomean | 0.49 | 5.44   | 4.56   | 4.04        | **3.966**   | 1.7% |

## 분석

- 오버라이드 정상 작동: contact.friction 값이 정확히 geometric mean으로 변경됨
- 이론 계산(F_net = F_applied - mu*mg)과 실측이 1-2% 이내 일치
- geomean 적용 시 구가 약 4배 빠르게 이동 (mu 0.8 → 0.49)
- 한쪽 mu가 낮으면 실제로 마찰이 낮아지는 현실적 행동

## 스크립트 (archived)

`jk_solver_examples/contacts/verify_friction.py` — 삭제됨, 결과만 보존
