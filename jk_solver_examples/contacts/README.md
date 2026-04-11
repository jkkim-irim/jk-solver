# Contact Examples

Newton + MuJoCo Warp solver 기반 접촉력/관절 시뮬레이션 및 실시간 디버깅 환경.

## 예제 목록

모든 명령은 `newton/jk-solver/` 디렉토리에서 실행:

| 예제 | 설명 | 실행 |
|------|------|------|
| `test_contact_ball.py` | Box(550g) + Sphere(5g)×2 접촉력 테스트 | `uv run jk_solver_examples/contacts/test_contact_ball.py` |
| `test_contact_hammer.py` | USD 망치 + STL finger pad 메시 접촉력 테스트 | `uv run jk_solver_examples/contacts/test_contact_hammer.py` |
| `test_finger_pinch_env.py` | ALLEX Hand articulation pinch 환경 | `uv run jk_solver_examples/contacts/test_finger_pinch_env.py` |

---

## test_contact_ball / test_contact_hammer

### 시나리오

- **ball**: 박스(550g) + 구(5g)×2, 구에 y방향 힘을 가해 박스로 밀어붙임
- **hammer**: USD 망치 + STL finger/thumb pad, coacd/convex hull 메시 충돌

공통:
- 법선력/마찰력을 **vector sum**으로 측정 (scalar sum 아닌 방향 고려)
- Geometric mean 마찰 오버라이드 (`mu = sqrt(mu1 * mu2)`)
- per-geom solref/solimp 직접 제어 + impedance plot

### 데모

https://github.com/user-attachments/assets/3c104e63-1c3a-4873-a209-bd2592fc6341

---

## test_finger_pinch_env

### 시나리오

ALLEX_Hand.usd (27 joints, 22 links)를 Newton에 로드하여 pinch 동작 시뮬레이션.

- 손 자세 제어: MJCF 기반 per-joint kp/kd, effort limit, position limit 적용
- DIP-PIP quartic polynomial mimic constraint (equality constraint)
- Pinch 토글: trajectory interpolation으로 부드러운 자세 전환 (configurable duration)
- Box(550g) 파지 대상

### 주요 설정

| 항목 | 설명 |
|------|------|
| `HAND_POS` / `HAND_ROT_DEG` | 손 위치 (m) / 오일러 각도 (deg) |
| `INIT_POSE_DEG` | 18개 active joint 초기 자세 (deg) |
| `PINCH_POSE_DEG` | pinch 시 변경할 joint 목표 자세 (deg) |
| `PINCH_DURATION` | pinch/open 전환 시간 (s) |
| `HAND_JOINT_GAINS` | per-joint kp/kd (MJCF actuator spec 기준) |
| `HAND_EFFORT_LIMIT` | per-joint torque limit (N·m) |
| `JOINT_LIMITS_RAD` | per-joint position limit (rad, MJCF ctrlrange 기준) |
| `EQ_SOLREF` / `EQ_SOLIMP` | equality constraint stiffness/impedance |
| `MIMIC_CONSTRAINTS` | DIP-PIP quartic polynomial 계수 |

### Joint 구조 (18 active DOF)

| 부위 | joints | DOF | 비고 |
|------|--------|-----|------|
| 손목 | Yaw, Roll, Pitch | 3 | 높은 kp/kd |
| 엄지 | Yaw, CMC, MCP | 3 | IP → MCP mimic |
| 검지 | ABAD, MCP, PIP | 3 | DIP → PIP mimic |
| 중지 | ABAD, MCP, PIP | 3 | DIP → PIP mimic |
| 약지 | ABAD, MCP, PIP | 3 | DIP → PIP mimic |
| 소지 | ABAD, MCP, PIP | 3 | DIP → PIP mimic |

---

## 디버깅 UI (공통)

실행하면 Newton 뷰어와 Contact Debug Monitor 두 창이 열립니다.

### Contact Debug Monitor (StatusWindow)

tkinter 기반 좌우 분할 디버깅 창. `jk_solver_examples/debug_monitor/status_window.py`에서 재사용 가능.

**좌측 탭:**

| 탭 | 내용 |
|----|------|
| Solver | solver 설정 요약, niter/nacon/nefc |
| Objects | body/joint 실시간 상태 |
| Contact | per-geom solref/solimp, impedance plot, pair forces |
| Joint | Index DIP efc_force / efc_pos / PIP torque 시계열 그래프 |

**우측 탭:**

| 탭 | 내용 |
|----|------|
| Solver | substep별 solver 수렴 로그 (`(sub)time`, niter, nacon, nefc) |
| Joint Constraint | substep별 Index DIP constraint force/error + PIP torque 로그 |

### Solver Conv 로그

| 항목 | 설명 |
|------|------|
| `niter` | solver iteration 수 / 최대값. 최대 도달 = 수렴 실패 (빨간색) |
| `nacon` | active contact 수. 0이면 접촉 없음 (회색) |
| `nefc` | 총 constraint 수 (contact + equality + limit) |

### imgui Param Tuner

Newton 뷰어 좌측 패널 → **Param Tuner**에서 실시간 조절:

- **Pinch 버튼** (pinch env): 토글로 pinch/open 전환
- Solver: iterations, ls_iterations, impratio
- Joint Drive: wrist kp/kd
- Physics: gravity
- Viewer Controls: pick stiffness/damping (hammer)

**Reset** 버튼으로 변경사항 일괄 적용.
