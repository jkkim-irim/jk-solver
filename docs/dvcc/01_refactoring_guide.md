# 리팩토링 가이드

## 핵심 원칙

- **YAGNI**: 미래를 대비한 미리 만들기 금지. 지금 필요한 것만 구현.
- **KISS**: 설명이 필요한 구조는 실패한 설계. 직관적으로.
- **선 병합, 후 분리**: 800줄 이상이거나 명확한 재사용 포인트가 보일 때만 분리.
- **Dead Code 즉시 삭제**: 미사용 import, 주석 처리된 코드, 쓰지 않는 변수.
- **불필요한 예외처리 금지**: try/except, fallback, 방어적 validation 남발 금지. 에러가 삼켜지면 디버깅이 불가능해진다. 예외처리는 시스템 경계(사용자 입력, 외부 API)에서만 최소한으로.

## 명명 규칙

- 모호한 이름 금지: `data`, `handle`, `temp` → `efc_normal_force`, `contact_pair_friction`
- 함수/변수명만으로 역할(What)과 이유(Why)가 드러나야 함

## 프로젝트 구조

```
jk-solver/
├── jk_solver_examples/
│   ├── __init__.py          # 패키지 진입점 (init + run 패치)
│   ├── jk_solver.py         # SolverJK (SolverMuJoCo 상속, 흐름 제어)
│   ├── jk_kernels.py        # GPU Warp 커널 (물리 연산)
│   └── contacts/
│       └── test_contact_force.py  # 접촉력 테스트 + StatusWindow UI
├── viewer/
│   └── jk_viewer.py               # JkViewerGL + JkViewerCfg (Newton ViewerGL 확장)
├── trajectory_generator/          # cubic Hermite spline 궤적 생성 (추후 활용)
├── allex_description/             # ALLEX MJCF/URDF/mesh 로봇 모델
└── docs/                          # PRD + 물리 테스트 결과
```

## Solver vs Kernel 구분

| 작성 대상 | 파일 | 역할 |
|-----------|------|------|
| 흐름 제어, 조건 분기, `wp.launch` 호출 | `jk_solver.py` | CPU에서 순차 실행 |
| 병렬 물리 연산, `wp.tid()` 사용 | `jk_kernels.py` | GPU에서 병렬 실행 |

## 리팩토링 프로세스

1. 현상 파악 — 중복/복잡한 조건문/성능 저하 지점 식별
2. 검증 수단 확보 — 리팩토링 전후 동일 결과 증명할 환경 확인
3. 최소 단위 실행 — 함수/클래스 단위로 한 블록씩
4. 중간 검증 — 수정 직후 실행하여 사이드 이펙트 확인
5. 커밋 — `refactor:` 또는 `perf:` 접두사 사용

## 적용 범위

| 대상 경로 | 설명 |
|----------|------|
| `jk_solver_examples/` | solver, kernels, 접촉 테스트 |
| `viewer/` | JkSolverViewerGL |
| `trajectory_generator/` | 궤적 생성 (코드 안정, 정리 완료) |

**주의: `newton/` upstream 코드는 절대 수정 금지.**

## 체크리스트

- [ ] 가독성: 코드만 보고 1분 내 로직 파악 가능한가?
- [ ] 중복: 유사 로직 3곳 이상 반복 없는가?
- [ ] 안정성: 기존 기능 정상 동작하는가?
- [ ] config/상수 적절히 분리되었는가?
- [ ] 미사용 코드 전부 제거되었는가?
- [ ] GPU 캐싱: 불변 배열 `.numpy()` 반복 호출 없는가?
