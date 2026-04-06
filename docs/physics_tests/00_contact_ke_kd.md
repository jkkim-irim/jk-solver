# Contact Stiffness/Damping (ke/kd) 검증 결과

## 목적

접촉 강성(ke)과 감쇠(kd)가 정적 접촉에서 법선력과 침투 깊이에 미치는 영향 확인.

## 테스트 조건

- 구: R=0.3m, density=10 kg/m^3, mass=1.131kg
- mg = 11.09N
- 바닥에 정적 접촉 상태에서 1초 안정화 후 측정
- solver: iterations=50, ls_iterations=10, cone=elliptic, impratio=1.0

## 결과

| ke         | kd      | 법선력(N)    | 침투(mm)   | solref       | 비고               |
| ---------- | ------- | --------- | -------- | ------------ | ---------------- |
| 100        | 10      | 11.09     | 3.29     | (0.20, 0.50) | 매우 물렁, 많이 파고듦    |
| 1,000      | 100     | 11.10     | 0.51     | (0.02, 1.58) | 약간 물렁            |
| **10,000** | **100** | **11.09** | **0.07** | (0.02, 0.50) | **기본값, 균형**      |
| 100,000    | 100     | 10.55     | -0.01    | (0.02, 0.16) | 너무 딱딱, 진동        |
| 10,000     | 10      | 12.44     | 0.21     | (0.20, 0.05) | 감쇠 부족, 진동        |
| 10,000     | 1,000   | 11.10     | 0.27     | (0.002, 5.0) | 과감쇠, 안정적이나 침투 큼  |
| 0          | 0       | 11.09     | 0.27     | (0.02, 1.0)  | MuJoCo 기본 solref |

## 분석

- 법선력은 ke/kd와 무관하게 정적 상태에서 mg(11.09N)로 수렴
- ke 증가 → 침투 감소 (더 딱딱한 접촉), 너무 높으면 진동 발생
- kd 증가 → 진동 억제 (감쇠), 대신 침투가 약간 증가
- ke=0, kd=0 → MuJoCo 기본값(solref=0.02,1.0) 사용, 합리적 결과
- **권장: ke=10,000, kd=100 (기본값)** — 침투 최소 + 안정적

## MuJoCo 접촉 모델 심화: ke/kd → solref → solimp

### MuJoCo는 스프링이 아니라 constraint다

MuJoCo는 접촉을 `F = ke*x + kd*v` 같은 스프링으로 풀지 않는다.
"물체가 뚫고 들어가면 안 된다"는 부등식 제약을 **최적화(constraint optimization)**로 푼다.

스프링 방식은 dt가 크면 힘이 폭발하고 불안정하지만,
constraint 방식은 "침투=0" 조건을 최적화로 풀기 때문에 dt에 덜 민감하고 안정적이다.
이게 MuJoCo가 큰 dt에서도 비교적 안정적인 이유다.

### solref = [timeconst, dampratio] — "접촉 응답 속도"

접촉이 발생하면 MuJoCo는 침투를 0으로 되돌리려 한다. 이때:

- **timeconst** (시간 상수): 침투를 얼마나 빨리 복원하나. 작을수록 빨리 밀어냄 (딱딱)
- **dampratio** (감쇠 비): 복원 과정에서 진동을 얼마나 억제하나
  - `< 1.0`: underdamped → 바운스/진동
  - `= 1.0`: critically damped → 진동 없이 복원 (기본값)
  - `> 1.0`: overdamped → 느리게 복원

물리 비유: 문을 닫는 도어클로저.
timeconst = 문이 닫히는 속도, dampratio = 쾅 닫히냐 부드럽게 닫히냐.

### ke/kd → solref 변환 공식

Newton의 ke/kd는 solref로 변환된다:

```
timeconst = 2 / (kd * d_width)
dampratio = (kd/2) * sqrt(d_r / ke)
```

- `d_width`: solimp에서 결정되는 impedance 전이 폭 (~0.001)
- `d_r`: impedance 스케일링 상수
- ke=0 or kd=0 → 기본값 solref=(0.02, 1.0) 사용

직관:

- **ke 높이면** → dampratio 낮아짐 → 더 바운스 (강한 스프링)
- **kd 높이면** → timeconst 짧아지고 + dampratio 높아짐 → 빨리 + 부드럽게 복원
- **ke=10000, kd=100** → solref=(0.02, 0.50) → 약간 바운스하면서 빠르게 복원

결과 표의 solref 열과 대조:
| ke | kd | solref | 해석 |
|---|---|---|---|
| 100, 10 | (0.20, 0.50) | 느리게 복원(0.20), 약간 바운스(0.50) |
| 10000, 100 | (0.02, 0.50) | 빠르게 복원(0.02), 약간 바운스(0.50) |
| 10000, 10 | (0.20, 0.05) | 느리게 복원(0.20), 심하게 바운스(0.05) |
| 10000, 1000 | (0.002, 5.0) | 매우 빠르게(0.002), 과감쇠(5.0) |

### solimp = [dmin, dmax, width, midpoint, power] — "침투 허용 범위"

solref가 "얼마나 빨리 밀어내나"라면, solimp는 **"얼마나 깊이 파고드는 걸 허용하나"**다.

MuJoCo의 impedance 함수:

```
d(침투거리) = dmin + (dmax - dmin) * sigmoid(침투거리, width, midpoint, power)
```

- **dmin** (0~1): 침투=0일 때 impedance. 높을수록 표면에서부터 강하게 저항
- **dmax** (0~1): 깊이 침투 시 impedance. 보통 0.99~1.0
- **width**: 전이 폭. impedance가 dmin→dmax로 바뀌는 거리
- **midpoint, power**: sigmoid 형태 제어

기본값 solimp = (0.9, 0.95, 0.001, 0.5, 2):

- 표면에서 impedance 0.9 (90% 저항)
- 깊이 들어가면 0.95 (95% 저항)
- 0.001m(1mm) 범위에서 전이

### 전체 처리 흐름

```
접촉 감지 → 침투 거리 d 측정
    ↓
solimp(d) → impedance 계산 (0~1, 얼마나 강하게 밀어낼지)
    ↓
solref → 복원 dynamics 결정 (얼마나 빨리, 진동 유무)
    ↓
constraint force 계산 → 물체에 적용
```

ke/kd는 사용자 친화적 인터페이스이고,
MuJoCo 내부에서는 solref/solimp로 변환되어 constraint optimization으로 푼다.

## 스크립트 (archived)

`jk_solver_examples/contacts/verify_ke_kd.py` — 삭제됨, 결과만 보존
