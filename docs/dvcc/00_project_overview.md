# jk-solver — 프로젝트 개요

ALLEX 60-DOF 휴머노이드 로봇의 미분 최적 제어 프레임워크.
Newton 물리 엔진 기반, MuJoCo Warp solver 사용, 접촉 시뮬레이션 + 파라미터 최적화.

---

## 프로젝트 구조

```
jk-solver/
├── allex_description/         # ALLEX MJCF/USD 로봇 모델
├── jk_solver_examples/        # 시뮬레이션 예제
│   ├── contacts/              # 접촉력 테스트 (법선/마찰)
│   ├── diffsim/               # differentiable simulation 예제
│   ├── jk_init.py             # viewer 초기화 유틸
│   ├── jk_solver.py           # jk-solver 메인 solver
│   └── jk_solver_kernels.py   # Warp 커널 (sparse H, friction override 등)
├── viewer/                    # JkSolverViewerGL (Newton ViewerGL 확장)
├── trajectory_generator/      # cubic Hermite spline 궤적 생성
├── tutorials/                 # 학습/가이드 문서
└── docs/                      # PRD 및 물리 테스트 결과
```

## 아키텍처

| 역할     | 구성요소                                                      |
| ------ | --------------------------------------------------------- |
| 물리 시뮬  | Newton + MuJoCo Warp solver (SolverMuJoCo)                |
| 미분 최적화 | MJX (JAX backend) + SHAC 알고리즘                             |
| 뷰어/GUI | Newton ViewerGL + JkSolverViewerGL (imgui) + tkinter 상태 창 |
| 모델     | ALLEX MJCF/USD (60-DOF, equality constraint)              |

## 핵심 기능

- **접촉 시뮬레이션**: 법선력/마찰력 측정, geometric mean 마찰 오버라이드
- **Differentiable Simulation**: wp.Tape backward, sliding window diffsim
- **파라미터 최적화**: kp/kd 미분 최적화, System ID (GT 궤적 기반)
- **Equality Constraint**: mimic joint 지원 (다항식 매핑)
- **실시간 GUI**: imgui Param Tuner + tkinter 상태/solver conv 모니터

## 환경

```
conda env: newton
python: 3.11
GPU: CUDA (warp)
newton: latest (upstream 수정 금지)
```
