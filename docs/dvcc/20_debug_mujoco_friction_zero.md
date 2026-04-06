# MuJoCo friction=0 충돌 소실 문제

> 날짜: 2026-04-06

## 문제

마찰계수를 0으로 설정하면 마찰력뿐 아니라 **법선력(충돌)까지 사라짐**.

## 근본 원인

`mujoco_warp/_src/constraint.py`의 `_contact_pyramidal` 커널:

```python
invweight = invweight + fri0 * fri0 * invweight      # (1)
invweight = invweight * 2.0 * fri0 * fri0 * ...       # (2)
```

`fri0 = friction[0] = 0`이면 **(2)에서 `invweight = 0`**.

이후 `_efc_row()`에서:
```python
D = 1.0 / max(invweight * (1.0 - imp) / imp, MJ_MINVAL)
```
`invweight=0` → `D = 1/MJ_MINVAL` → 무한대 impedance → constraint 비활성화.

**핵심**: MuJoCo에서 friction 값은 단순 마찰계수가 아니라 **impedance multiplier** 역할도 함.
friction=0이면 법선/접선 모든 방향의 constraint가 통째로 죽음.

## 해결

`override_contact_friction_geomean` 커널에서 `1e-6` 하한 clamp:

```python
gm_slide = wp.max(wp.sqrt(f1[0] * f2[0]), 1.0e-6)
```

## 영향

- elliptic / pyramidal 양쪽 동일 현상
- Newton의 `contacts.force`도 같은 이유로 0이 됨 (constraint 자체가 비활성화이므로)
