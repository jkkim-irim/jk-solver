# 커밋 컨벤션

Conventional Commits 기반. 메시지만 보고 변경 종류와 범위를 즉시 파악 가능하게.

## 구조

```
<type>[(scope)]: <description>

[body]
```

## 타입

| 타입         | 의미     | 예시                                                            |
| ---------- | ------ | ------------------------------------------------------------- |
| `feat`     | 기능 추가  | `feat(contact): add geometric mean friction override`         |
| `fix`      | 버그 수정  | `fix(solver): clamp friction to 1e-6 to prevent contact drop` |
| `refactor` | 리팩터링   | `refactor: extract _get_contact_forces from _update_status`   |
| `docs`     | 문서     | `docs: add solver convergence test results`                   |
| `chore`    | 빌드/패키지 | `chore: update pyrightconfig`                                 |
| `test`     | 테스트    | `test: add contact force verification script`                 |
| `style`    | 포맷팅만   | `style: apply black formatter`                                |

## 규칙

- 제목 50자 이내, 첫 글자 소문자, 마침표 없음
- 명령문 사용 (add, fix — added, fixed 아님)
- 본문은 **무엇을/왜** 위주 (어떻게는 코드에)
- scope 예시: `contact`, `solver`, `viewer`, `kernel`, `description`, `trajectory`

## 에이전트 커밋+push 절차

1. `git status` + `git diff --stat`으로 변경 파악
2. 위 규칙에 맞게 메시지 작성
3. 관련 파일만 `git add`
4. `git commit` → `git log -1 --oneline`으로 검증
5. push 요청 시: `git push` (추적 없으면 `git push -u origin <branch>`)
