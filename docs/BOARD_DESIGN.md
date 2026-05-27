# 윷놀이 보드 좌표계 명세

이 문서는 `Board` 구현 기준으로 확정된 보드 좌표계 명세다.
2026-05-27 사용자 확인으로 아래 29칸 physical cell, 자동 지름길, 중앙 진행, `C3` 처리 규칙을 확정했다.

## 1. 설계 목표

- 윷판의 실제 칸은 `physical cell`로 표현한다.
- 이동 경로와 다음 방향은 `logical route position`으로 표현한다.
- 잡기와 업기는 같은 `physical cell`에 도착했는지로 판정한다.
- 지름길 진입은 플레이어 선택 없이 자동 처리한다.
- 뒷도와 후진은 사용하지 않는다.

## 2. Physical Cell

전체 physical cell은 29개로 둔다.

### 외곽 20칸

| ID | 이름 | 설명 |
| ---: | --- | --- |
| 0 | `HOME` | 참먹이. 시작점이자 정확히 도착하면 머무는 칸 |
| 1 | `O1` | HOME에서 1칸 |
| 2 | `O2` | HOME에서 2칸 |
| 3 | `O3` | HOME에서 3칸 |
| 4 | `O4` | HOME에서 4칸 |
| 5 | `C1` | 첫 번째 모서리 분기점 |
| 6 | `O6` | 외곽 6번째 칸 |
| 7 | `O7` | 외곽 7번째 칸 |
| 8 | `O8` | 외곽 8번째 칸 |
| 9 | `O9` | 외곽 9번째 칸 |
| 10 | `C2` | 두 번째 모서리 분기점 |
| 11 | `O11` | 외곽 11번째 칸 |
| 12 | `O12` | 외곽 12번째 칸 |
| 13 | `O13` | 외곽 13번째 칸 |
| 14 | `O14` | 외곽 14번째 칸 |
| 15 | `C3` | 세 번째 모서리 분기점 |
| 16 | `O16` | 외곽 16번째 칸 |
| 17 | `O17` | 외곽 17번째 칸 |
| 18 | `O18` | 외곽 18번째 칸 |
| 19 | `O19` | 외곽 19번째 칸 |

외곽 기본 진행은 다음 순서다.

```text
HOME -> O1 -> O2 -> O3 -> O4 -> C1
     -> O6 -> O7 -> O8 -> O9 -> C2
     -> O11 -> O12 -> O13 -> O14 -> C3
     -> O16 -> O17 -> O18 -> O19 -> HOME
```

### 내부 9칸

| ID | 이름 | 설명 |
| ---: | --- | --- |
| 20 | `A1` | C1 지름길 1번째 내부 칸 |
| 21 | `A2` | C1 지름길 2번째 내부 칸 |
| 22 | `CENTER` | 중앙 공유 칸 |
| 23 | `A3` | CENTER에서 C3 방향 내부 칸 |
| 24 | `A4` | C3 직전 내부 칸 |
| 25 | `B1` | C2 지름길 1번째 내부 칸 |
| 26 | `B2` | C2 지름길 2번째 내부 칸 |
| 27 | `B3` | CENTER에서 HOME 방향 내부 칸 |
| 28 | `B4` | HOME 직전 내부 칸 |

내부 지름길은 다음 두 대각선으로 둔다.

```text
C1 -> A1 -> A2 -> CENTER -> A3 -> A4 -> C3
C2 -> B1 -> B2 -> CENTER -> B3 -> B4 -> HOME
```

## 3. Logical Route

physical cell만으로는 같은 중앙 칸에 있더라도 이후 진행 방향을 알 수 없으므로, 말의 실제 위치는 logical route position으로 저장한다.

확정 route:

```text
OUTER:
HOME, O1, O2, O3, O4, C1, O6, O7, O8, O9, C2,
O11, O12, O13, O14, C3, O16, O17, O18, O19, HOME

C1_DIAGONAL:
C1, A1, A2, CENTER, A3, A4, C3, O16, O17, O18, O19, HOME

C2_DIAGONAL:
C2, B1, B2, CENTER, B3, B4, HOME

CENTER_TO_HOME:
CENTER, B3, B4, HOME
```

## 4. 자동 지름길 규칙

- `OUTER` 경로에서 `C1`에 정확히 멈추면 다음 이동부터 `C1_DIAGONAL`을 따른다.
- `OUTER` 경로에서 `C2`에 정확히 멈추면 다음 이동부터 `C2_DIAGONAL`을 따른다.
- `OUTER` 경로에서 `C3`에 정확히 멈추면 별도 지름길이 없으므로 계속 `OUTER`를 따른다.
- 이동 중 `C1` 또는 `C2`를 지나치기만 하면 `OUTER`를 유지한다.
- `CENTER`에 정확히 멈추면 다음 이동부터 `CENTER_TO_HOME`을 따른다.
- 이동 중 `CENTER`를 지나치기만 하면 현재 route를 유지한다.
- `C1_DIAGONAL`을 따라 `CENTER`를 지나친 말은 `A3 -> A4 -> C3`로 계속 간다.

## 5. 참먹이와 골인

- `WAITING` 말은 `HOME`에서 출발한다고 보고, 사용한 윷 결과만큼 이동한 칸에 놓인다.
- `WAITING + DO`는 `O1`, `WAITING + MO`는 `C1`이다.
- 말이 이동 결과로 `HOME`에 정확히 도착하면 `ON_BOARD` 상태로 `HOME`에 머문다.
- `HOME`에 머문 말이 다음 양수 이동 결과를 사용하면 즉시 `FINISHED`가 된다.
- `HOME`을 초과해 지나가면 즉시 `FINISHED`가 된다.
- `HOME`에 머문 상대 말은 잡을 수 있다.

예시:

```text
O18 + GAE(2) -> HOME, FINISHED 아님
O18 + GEOL(3) -> FINISHED
O19 + DO(1) -> HOME, FINISHED 아님
O19 + GAE(2) -> FINISHED
HOME(ON_BOARD) + DO/GAE/GEOL/YUT/MO -> FINISHED
```

## 6. Board.move 입력/출력

입력:

```python
Board.move(position: Position, steps: int) -> MoveResult
```

`Position`:

```python
status: WAITING | ON_BOARD | FINISHED
route: OUTER | C1_DIAGONAL | C2_DIAGONAL | CENTER_TO_HOME | None
index: int | None
physical_cell: int | None
```

`MoveResult`:

```python
status: WAITING | ON_BOARD | FINISHED
route: str | None
index: int | None
physical_cell: int | None
entered_shortcut: bool
landed_on_home: bool
passed_home: bool
```

## 7. 확정 사항

- 29칸 physical cell 명세를 그대로 사용한다.
- `CENTER`에 정확히 멈춘 말은 항상 `CENTER_TO_HOME`으로 간다.
- `C1_DIAGONAL`에서 `CENTER`를 지나친 말은 `A3 -> A4 -> C3`로 계속 간다.
- `C3`은 외곽 모서리지만 별도 지름길 진입점으로 보지 않는다.
