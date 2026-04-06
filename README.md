# jk-solver

ALLEX 휴머노이드 로봇의 접촉 시뮬레이션 및 미분 최적 제어 프레임워크.
Newton 물리 엔진 기반, MuJoCo Warp solver 사용.

## 의존성

| 패키지         | 버전     | 비고                                    |
| ----------- | ------ | ------------------------------------- |
| Newton      | 1.0.0  | `release-1.0` 브랜치 (commit `d6046f18`) |
| Warp        | 1.12.0 | `warp-lang==1.12.0`                   |
| MuJoCo      | 3.6.0  |                                       |
| MuJoCo Warp | 3.6.0  |                                       |
| Python      | 3.11+  |                                       |
| CUDA        | 12.x   | RTX 30/40/50 시리즈                      |

## 설치

### 1. conda 가상환경 생성

```bash
conda create -n newton python=3.11 -y
conda activate newton
```

### 2. Newton 설치

```bash
git clone git@github.com:newton-physics/newton.git
cd newton
git checkout release-1.0  # v1.0.0 태그 (commit d6046f18)
pip install -e .
```

### 3. 의존성 설치

```bash
pip install warp-lang==1.12.0
pip install mujoco==3.6.0 mujoco-warp==3.6.0
```

### 4. jk-solver 클론

Newton 루트 디렉토리 안에 클론:

```bash
cd newton
git clone git@github.com:jkkim-irim/jk-solver.git
```

최종 구조:

```
newton/                    # Newton 물리 엔진 (upstream)
├── newton/                # Newton 패키지 소스
└── jk-solver/             # 이 저장소
```

## 실행

```bash
conda activate newton
cd newton/jk-solver
python jk_solver_examples/contacts/test_contact_force.py
```

## 프로젝트 구조

```
jk-solver/
├── jk_solver_examples/              # 시뮬레이션 코드
│   ├── __init__.py                  # 패키지 진입점 (viewer 초기화 + run 패치)
│   ├── jk_solver.py                 # SolverJK — SolverMuJoCo 상속, 흐름 제어 (CPU)
│   ├── jk_kernels.py                # GPU Warp 커널 (물리 연산)
│   └── contacts/
│       └── test_contact_force.py    # 접촉력 테스트 + StatusWindow 디버깅 UI
│
├── viewer/
│   └── jk_viewer.py                 # JkViewerGL — Newton ViewerGL 확장 (imgui 패널)
│
├── trajectory_generator/            # ALLEX 궤적 생성 (cubic Hermite spline)
├── allex_description/               # ALLEX 로봇 모델 (MJCF/URDF/mesh)
│
└── docs/
    ├── dvcc/                        # 프로젝트 PRD 문서
    └── physics_tests/               # 물리 테스트 결과 (접촉, 마찰, 수렴성 등)
```

## Solver vs Kernel

| 구분     | 파일              | 실행 위치 | 역할                         |
| ------ | --------------- | ----- | -------------------------- |
| Solver | `jk_solver.py`  | CPU   | step 흐름 제어, `wp.launch` 호출 |
| Kernel | `jk_kernels.py` | GPU   | 병렬 물리 연산 (`wp.tid()` 기반)   |

커스텀 물리 연산 추가: `jk_kernels.py`에 커널 작성 → `jk_solver.py`에서 `wp.launch` 호출.

## 주요 기능

- **접촉 시뮬레이션**: 법선력/마찰력 측정, geometric mean 마찰 오버라이드
- **실시간 디버깅 UI**: tkinter 탭 기반 모니터 (Objects/Contact/Solver) + Solver Conv 로그
- **GPU 최적화**: 불변 배열 캐싱, `atomic_max` 커널로 침투깊이 추적
- **imgui Param Tuner**: solver/contact/geometry 파라미터 실시간 조절

## 규칙

- `newton/` upstream 코드는 **절대 수정 금지**
- 상세 가이드: `docs/dvcc/01_refactoring_guide.md`
