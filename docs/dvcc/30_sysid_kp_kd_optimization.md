# System ID: kp/kd 미분 최적화

> 대상: ALLEX 48 actuator, CES GT 궤적 기반

## 최종 결과

- Overall RMS: 5.22° (손가락 제외 시 ~1.7°)
- Waist/Neck: 0.85°, Shoulder/Elbow/Wrist: 1.7~2.3°, Hand: 6.2~6.7°

## 설정

- MJX implicit solver + jax.grad (1-DOF 단위에서는 정확)
- Sliding window: segment=500 steps (2.5초)
- Adam optimizer, lr auto schedule (reduce_on_plateau)
- 토크 제한(forcerange) 수동 적용 필수

## 핵심 교훈

1. **토크 포화 시 gradient=0**: kp가 크면 토크 clamp에 걸려 gradient가 죽음. 초기 kp를 토크 제한의 50%/range로 설정
2. **전체 궤적 rollout 불가**: 2000+ steps에서 gradient vanishing/exploding. segment=500 steps가 안정 한계
3. **CSV-GT 시간 정렬 필수**: cross-correlation으로 offset 탐색 (GT offset=719 steps)
4. **kp/kd 분리 최적화**: kd 고정 → kp 최적화 → kd 최적화 순서가 안정적
5. **관절군별 분리 최적화**: 손가락(15개)만 따로 최적화하면 gradient가 깨끗
6. **Adam 필수**: SGD는 segment별 gradient 크기 변동에 대응 불가

## 한계

- kp/kd만으로는 sim-real gap 한계 (손가락 PIP 6~12°)
- 토크 포화 + 마찰/백래시 미모델링이 원인
- 다음 단계: armature/damping/frictionloss 추가 최적화 또는 MLP 잔차 학습
