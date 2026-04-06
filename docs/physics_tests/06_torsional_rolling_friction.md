# Torsional/Rolling Friction 검증 결과

## 목적
접촉점에서의 비틀림(torsional) 마찰과 구름(rolling) 마찰이 물체 회전에 미치는 영향 확인.

## 결과: Newton MuJoCo solver 경로에서 효과 없음

### Torsional friction (z축 토크 1Nm, 1초 후 각속도)

| mu_torsional | omega_z (rad/s) |
|---|---|
| 0.0 | 2.456 |
| 0.001 | 2.456 |
| 0.01 | 2.456 |
| 0.1 | 2.456 |

**모든 값에서 동일 → torsional friction 미반영**

### Rolling friction (5N y방향 힘, 1초 후)

| mu_rolling | y_pos | vel_y | |omega| |
|---|---|---|---|
| 0.0 | 0.085 | 0.170 | 1.133 |
| 0.001 | 0.085 | 0.170 | 1.133 |
| 0.01 | 0.085 | 0.170 | 1.133 |
| 0.05 | 0.085 | 0.170 | 1.133 |

**모든 값에서 동일 → rolling friction 미반영**

## 원인 분석

MuJoCo의 마찰 모델은 `condim` (contact dimensionality)로 결정:
- **condim=1**: 마찰 없음 (frictionless)
- **condim=3**: sliding friction만 (기본값)
- **condim=4**: sliding + torsional
- **condim=6**: sliding + torsional + rolling (elliptic cone 필요)

Newton → MuJoCo 변환 시 `geom_condim`이 기본값 3으로 설정되어 torsional/rolling 성분이 무시됨.
`geom_friction` vec3에 [slide, torsional, rolling] 값은 저장되지만, condim=3이면 [0]만 사용.

## 해결 방법
- `geom_condim`을 6으로 설정하는 커스텀 속성 또는 solver 초기화 시 강제 설정 필요
- 또는 MuJoCo MJCF에서 직접 `condim="6"` 지정

## 결론
- 현재 Newton MuJoCo solver에서는 torsional/rolling friction이 **사실상 비활성화** 상태
- manipulation에서 in-hand rotation 저항이 필요하면 condim 설정 오버라이드가 필요
- sliding friction(mu)만 유효

## 스크립트 (archived)
`jk_solver_examples/contacts/verify_torsional_rolling.py` — 삭제됨, 결과만 보존
