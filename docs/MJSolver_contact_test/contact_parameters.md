# MuJoCo Solver 접촉 파라미터 가이드

## 접촉 처리 흐름

```
1. Broadphase          AABB 겹침 체크 (빠름)
   ↓ gap으로 AABB 확장
2. Narrowphase         정밀 거리 계산 → contact.dist
   ↓ dist ≤ margin 이면 active contact (nacon)
3. Constraint 생성     solref → aref, solimp → D(r) 계산
   ↓ 접촉점당 법선1 + 마찰2 = nefc (elliptic cone, condim=3)
4. Solver Iteration    D, aref로 constraint force 계산
```

---

## 1. solref — 접촉 응답 특성

접촉 시 "얼마나 빨리, 얼마나 진동 없이" 반발하는지를 결정.

| 파라미터               | 의미                      | 범위          |
| ------------------ | ----------------------- | ----------- |
| **tc** (timeconst) | 접촉 응답 시간 [s]. 작을수록 딱딱함  | 0.001 ~ 0.1 |
| **dr** (dampratio) | 감쇠비. 1.0 = 임계감쇠 (진동 없음) | 0.5 ~ 2.0   |

### ke/kd → solref 변환

Newton에서 ke/kd를 설정하면 내부적으로 solref로 자동 변환됨 (`d_width=1.0, d_r=1.0`):

```
tc = 2 / kd
dr = kd / (2 * sqrt(ke))
```

ke=0, kd=0이면 MuJoCo 기본값 적용: `tc=0.02, dr=1.0`

### 직접 설정

solref를 직접 제어하려면 ke=0, kd=0으로 두고 `mjw_model.geom_solref`에 직접 값을 쓰면 됨.

---

## 2. solimp — impedance 함수 (침투깊이에 따른 constraint 강도)

침투가 깊어질수록 constraint가 얼마나 강하게 작용하는지를 sigmoid 함수로 제어.

| 파라미터         | 기본값    | 의미                       |
| ------------ | ------ | ------------------------ |
| **dmin**     | 0.9    | 얕은 침투에서의 impedance (0~1) |
| **dmax**     | 0.95   | 깊은 침투에서의 impedance (0~1) |
| **width**    | 0.001m | dmin→dmax 전이 구간 폭 [m]    |
| **midpoint** | 0.5    | 전이 시작점 = `mid * width`   |
| **power**    | 2.0    | sigmoid 경사도. 높을수록 급격한 전이 |

### impedance 함수 D(r)

```
s = clamp((r - mid * width) / ((1 - mid) * width), 0, 1)
D(r) = dmin + (dmax - dmin) * s^power
```

![](/home/jkkim/.config/marktext/images/2026-04-06-20-28-11-image.png)

- `r < mid * width`: D = dmin (최소 impedance)
- `mid * width < r < width`: dmin → dmax로 전이 (sigmoid)
- `r > width`: D = dmax (최대 impedance)

### 

### 주요 설정 패턴

| 의도           | dmin | dmax | width |
| ------------ | ---- | ---- | ----- |
| 딱딱한 접촉 (금속)  | 1.0  | 1.0  | 0.001 |
| 부드러운 접촉 (고무) | 0.3  | 0.9  | 0.01  |
| 기본값          | 0.9  | 0.95 | 0.001 |

---

## 3. solmix — 두 geom의 파라미터 블렌딩

서로 다른 solimp를 가진 두 geom이 접촉할 때, 최종 solimp는 **가중평균**으로 결정.

```
mix = solmix_A / (solmix_A + solmix_B)
최종_solimp = mix * solimp_A + (1 - mix) * solimp_B
```

| solmix_A | solmix_B | mix  | 결과            |
| -------- | -------- | ---- | ------------- |
| 1.0      | 1.0      | 0.5  | 50:50 평균 (기본) |
| 0.0      | 1.0      | 0.0  | B의 solimp만 사용 |
| 2.0      | 1.0      | 0.67 | A 쪽에 가중       |

solref도 동일한 방식으로 블렌딩됨.

---

## 4. rigid_gap — broadphase AABB 확장

| 파라미터       | 기본값  | 의미                                          |
| ---------- | ---- | ------------------------------------------- |
| **gap**    | 0.1m | AABB를 `margin + gap`만큼 확장                   |
| **margin** | 0.0m | geom 표면 확장. `dist ≤ margin`이면 constraint 생성 |

### gap vs margin

```
  ┌──────────────────────────┐
  │    AABB (margin + gap)   │  ← gap: 후보 등록만. 힘 없음.
  │  ┌──────────────────┐   │
  │  │  margin 확장      │   │  ← margin: 이 안에 들어오면 힘 발생
  │  │ ┌──────────────┐ │   │
  │  │ │ 실제 표면      │ │   │
  │  │ └──────────────┘ │   │
  │  └──────────────────┘   │
  └──────────────────────────┘
```

- **gap이 크면**: broadphase 후보가 늘어남 → narrowphase 연산 증가. 하지만 solver 연산에는 영향 없음.
- **gap이 너무 작으면**: 빠른 물체가 한 timestep에 AABB를 뚫고 접촉을 놓칠 수 있음.
- **권장**: `gap ≥ max_velocity * dt`. (예: 5 m/s, dt=0.002s → gap ≥ 0.01m)

접촉 쌍의 유효 gap: `effective_gap = gap_A + gap_B`

---

## 5. impratio — friction constraint impedance 배율

법선 constraint 대비 마찰 constraint의 impedance 비율.

```
friction_impedance = normal_impedance * impratio
```

| impratio | 효과                                   |
| -------- | ------------------------------------ |
| 1.0      | 법선과 마찰이 동일 강도 (기본)                   |
| 1000.0   | 마찰 constraint가 1000배 강함 → 미끄러짐 거의 불가 |
| 0.1      | 마찰 약함 → 잘 미끄러짐                       |

**주의**: solver 생성 시 `spec.option.impratio`에 고정. 런타임 변경 불가.

---

## 6. friction override — max → geometric mean

### MuJoCo 기본 방식

```
mu = max(mu_A, mu_B)
```

한쪽이 마찰이 크면 무조건 큰 값 사용 → **비현실적**.

### jk-solver 방식 (geometric mean)

```
mu = sqrt(mu_A * mu_B)
```

두 재질의 기하평균 → **물리적으로 합리적**.

### friction = 0 문제

한쪽 `mu = 0`이면 `sqrt(0 * x) = 0` → MuJoCo 내부에서 friction이 **impedance multiplier**로도 사용되기 때문에 `invweight = 0` → **법선 constraint까지 비활성화** → 충돌 자체가 사라짐.

**해결**: `wp.max(sqrt(mu1 * mu2), 1e-6)` 하한 clamp.

```python
# jk_kernels.py
gm_slide = wp.max(wp.sqrt(f1[0] * f2[0]), 1.0e-6)
```

---

## 파라미터 요약

| 파라미터                                   | 범주              | 런타임 변경            | geom별 설정          |
| -------------------------------------- | --------------- | ----------------- | ----------------- |
| tc, dr (solref)                        | 접촉 응답           | Reset 필요          | O                 |
| dmin, dmax, width, mid, power (solimp) | impedance       | Reset 필요          | O                 |
| solmix                                 | 블렌딩 비율          | Reset 필요          | O                 |
| mu (friction)                          | 마찰계수            | Reset 필요          | O                 |
| rigid_gap                              | broadphase 확장   | Reset 필요          | O (gap per shape) |
| impratio                               | 마찰 impedance 배율 | **X** (solver 고정) | X (전역)            |
| iterations, ls_iterations              | solver 반복       | **X** (solver 고정) | X (전역)            |
| cone                                   | 마찰 cone 모델      | **X** (solver 고정) | X (전역)            |
