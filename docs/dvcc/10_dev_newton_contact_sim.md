# 접촉 시뮬레이션 개발

> 대상: `jk_solver_examples/contacts/`

## 개요

Newton + MuJoCo Warp solver 기반 접촉력(법선/마찰) 측정 및 검증 환경.
박스-구-바닥 3체 접촉 시나리오로 solver 파라미터 튜닝.

```bash
conda activate newton && python jk-solver/jk_solver_examples/contacts/test_contact_force.py
```

## 접촉력 추출 방식

MuJoCo Warp의 EFC(equality/friction constraint) force에서 직접 읽음:

```python
d = solver.mjw_data
efc_force = d.efc.force.numpy()           # (nworld, nefc_max)
contact_efc = d.contact.efc_address.numpy()  # (nacon, dim)

# 법선력: efc_address[c, 0]
nf = abs(efc_force[world, contact_efc[c, 0]])

# 마찰력: efc_address[c, 1:dim] (elliptic) 또는 [c, 1:2*(dim-1)] (pyramidal)
```

## Geometric Mean 마찰 오버라이드

MuJoCo 기본: `mu = max(mu1, mu2)` → 비현실적.
jk-solver: `mu = sqrt(mu1 * mu2)` (geometric mean) 커널로 오버라이드.

**주의**: `mu=0`이면 MuJoCo 내부에서 impedance=0 → constraint 비활성화 → 충돌 소실.
해결: `wp.max(sqrt(mu1*mu2), 1e-6)` 하한 clamp.

## GUI 구성

- **imgui Param Tuner**: solver iter, mu, ke/kd, mass 등 실시간 조절 → Reset으로 적용
- **tkinter StatusWindow**: 좌(Sim Status: body별 pos/vel/force), 우(Solver Conv: niter/nacon/nefc)

## 검증 항목

- 정적 접촉 법선력 ≈ mg
- 접촉 쌍별(pair) 법선력/���찰력 분리 표시
- solver 수렴 iteration 모니터링 (max iter 도달 시 빨간색 경고)
