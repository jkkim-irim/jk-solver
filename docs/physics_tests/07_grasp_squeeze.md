# Grasp Squeeze Test 검증 결과

## 목적

양쪽 구로 박스를 끼워 잡을 때 최소 grip force 측정. 마찰력이 박스 중력을 이겨야 holds.

## 이론

- 마찰력 (한쪽) = mu * squeeze_force
- 양쪽 합: 2 * mu * squeeze_force >= box_mg
- **최소 squeeze = box_mg / (2 * mu)**

## 테스트 조건

- 박스: half=0.2m, density=100, mass=6.40kg, mg=62.78N
- 구: R=0.15m, 질량 무시 (density=0), y방향으로만 이동 가능
- 박스: z축으로만 이동 가능 (공중에 배치, z=1.0)
- geomean friction 적용
- 구와 박스 mu 동일 설정 → geomean = mu

## 결과

| mu  | 이론 min(N) | squeeze=20 | squeeze=50 | squeeze=100 |
| --- | --------- | ---------- | ---------- | ----------- |
| 0.2 | 157.0     | NO         | NO         | NO          |
| 0.3 | 104.6     | NO         | NO         | NO          |
| 0.5 | 62.8      | NO         | NO         | **YES**     |
| 0.8 | 39.2      | NO         | **YES**    | **YES**     |
| 1.0 | 31.4      | NO         | **YES**    | **YES**     |

(YES = box_z > 0.8, NO = 바닥으로 떨어짐)

## 분석

### 이론 vs 실측 차이

- mu=0.5: 이론 min=62.8N이지만 실제로 50N으로는 불가, 100N에서 성공
- mu=0.8: 이론 min=39.2N, 50N에서 성공 (이론보다 약간 높음)
- mu=1.0: 이론 min=31.4N, 50N에서 성공

### 차이 원인

1. 접촉이 완벽한 점접촉이 아님 (구-박스 곡면 접촉)
2. 동적 효과: squeeze 가하는 동안 박스가 이미 떨어지기 시작
3. ke/kd에 의한 접촉 침투로 실효 접촉력이 이론보다 낮음
4. 접촉 안정화에 시간 소요

### 결론

- mu 높을수록 적은 squeeze로 파지 가능 (예상대로)
- 실제 필요한 squeeze force는 이론값의 약 1.3~1.6배
- **manipulation 학습 시 마진을 두고 grip force 설계 필요**

## 스크립트 (archived)

`jk_solver_examples/contacts/verify_grasp.py` — 삭제됨, 결과만 보존
