# jk-solver

ALLEX 휴머노이드 로봇의 접촉 시뮬레이션 및 미분 최적 제어 프레임워크.
Newton 물리 엔진 기반, MuJoCo Warp solver 사용.

## 의존성

| 패키지         | 버전       | 비고                                |
| ----------- | -------- | --------------------------------- |
| Newton      | 1.0.0    | `release-1.0` 브랜치                 |
| Warp        | >=1.12.0 | Newton pyproject.toml에서 관리        |
| MuJoCo Warp | 3.5.0.2  | Newton `[sim]` optional dependency |
| Python      | 3.12     | `.python-version` 참조              |
| CUDA        | 12.x+    | RTX 30/40/50 시리즈                  |
| uv          | >=0.9    | 패키지/가상환경 관리                       |

## 설치

### 1. uv 설치

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Newton 클론 및 환경 설정

```bash
git clone git@github.com:newton-physics/newton.git
cd newton
git checkout release-1.0

# uv가 .python-version(3.12) + pyproject.toml 기반으로 가상환경 자동 생성
uv sync --extra sim --extra importers
```

### 3. jk-solver 클론

Newton 루트 디렉토리 안에 클론:

```bash
cd newton
git clone git@github.com:jkkim-irim/jk-solver.git
```

### 최종 구조

```
newton/                    # Newton 물리 엔진 (upstream)
├── pyproject.toml         # uv 패키지 관리
├── uv.lock                # 의존성 lock 파일
├── .python-version        # Python 3.12
├── newton/                # Newton 패키지 소스
└── jk-solver/             # 이 저장소
```

## 실행

**모든 명령은 `newton/jk-solver/` 디렉토리에서 실행:**

```bash
cd newton/jk-solver

# 접촉력 테스트 (ball + box)
uv run jk_solver_examples/contacts/test_contact_ball.py

# 접촉력 테스트 (mesh hammer + finger pads)
uv run jk_solver_examples/contacts/test_contact_hammer.py

# ALLEX Hand pinch 환경
uv run jk_solver_examples/contacts/test_finger_pinch_env.py
```

> `uv run`은 상위 디렉토리의 `pyproject.toml`을 자동 감지하여 올바른 가상환경에서 실행합니다.

## 프로젝트 구조

```
jk-solver/
├── jk_solver_examples/              # 시뮬레이션 코드
│   ├── __init__.py                  # 패키지 진입점 (viewer 초기화 + run 패치)
│   ├── jk_solver.py                 # SolverJK — SolverMuJoCo 상속, 흐름 제어 (CPU)
│   ├── jk_kernels.py                # GPU Warp 커널 (물리 연산)
│   ├── debug_monitor/               # 재사용 가능한 디버깅 UI 모듈
│   │   ├── __init__.py
│   │   └── status_window.py         # StatusWindow — tkinter 탭 기반 디버그 모니터
│   └── contacts/
│       ├── test_contact_ball.py     # 접촉력 테스트 (box + sphere, vector sum force)
│       ├── test_contact_hammer.py   # 접촉력 테스트 (USD mesh hammer + STL finger pads)
│       └── test_finger_pinch_env.py # ALLEX Hand pinch 환경 (USD articulation)
│
├── assets/                          # 시뮬레이션 에셋
│   ├── ALLEX_Hand.usd               # ALLEX 로봇 핸드 (27 joints, 22 links)
│   ├── Hammer.usd                   # 망치 메시
│   ├── Finger_Distal_Pad.stl        # 검지 패드 메시
│   └── R_Thumb_Distal_Pad.stl       # 엄지 패드 메시
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

- **접촉력 시뮬레이션**: 법선력/마찰력 vector sum 측정, geometric mean 마찰 오버라이드
- **ALLEX Hand pinch 환경**: USD articulation 로드, DIP-PIP quartic polynomial mimic constraint, MJCF 기반 joint limit/effort limit/PD gain 설정
- **실시간 디버깅 UI**: tkinter StatusWindow (Solver/Objects/Contact/Joint 탭) + 우측 Solver/Joint Constraint 로그
- **GPU 최적화**: 불변 배열 캐싱, `atomic_max` 커널로 침투깊이 추적
- **imgui Param Tuner**: solver/contact/geometry 파라미터 실시간 조절, pinch 토글 (trajectory interpolation)

## 규칙

- `newton/` upstream 코드는 **절대 수정 금지**
- 상세 가이드: `docs/dvcc/01_refactoring_guide.md`
