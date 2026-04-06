# Restitution (반발 계수) 검증 결과

## 목적

반발 계수(restitution)가 충돌 후 바운스 높이에 미치는 영향 확인.

## 결과: MuJoCo Solver에서 미지원

**ShapeConfig.restitution은 XPBD solver 전용이며, MuJoCo solver에서는 무시됩니다.**

테스트 결과 restitution=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0] 모두 동일한 결과:

- 바운스 높이 = 0.038m (거의 바운스 없음)
- restitution 값과 무관

## MuJoCo에서 반발 효과를 내는 방법

MuJoCo는 반발을 `solref`(time constant, damping ratio)로 제어:

- **damping ratio < 1.0** → underdamped → 바운스 발생
- **damping ratio = 1.0** → critically damped → 바운스 없음 (기본값)
- **damping ratio > 1.0** → overdamped → 느리게 접근

ke/kd로 간접 제어 가능:

- `dampratio = kd/2 * sqrt(d_r / ke)`
- ke를 높이고 kd를 낮추면 dampratio < 1 → 바운스

예: ke=10000, kd=10 → dampratio ≈ 0.05 → 진동/바운스 발생 (00_contact_ke_kd.md 참조)

## 결론

- MuJoCo solver 사용 시 restitution 파라미터는 효과 없음
- 반발 효과가 필요하면 ke/kd 조합으로 solref를 조절하거나, MuJoCo의 pair-level solref를 직접 설정해야 함
- 또는 XPBD solver로 전환 필요

## 스크립트 (archived)

`jk_solver_examples/contacts/verify_restitution.py` — 삭제됨, 결과만 보존
