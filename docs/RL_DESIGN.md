# 윷놀이 RL 1차 설계안

이 문서는 `RULES.md`의 확정 룰을 강화학습 구현 관점에서 옮긴 1차 설계안이다.
목표는 알고리즘을 아직 고정하지 않은 상태에서도 환경, 상태, 행동, 보상, agent 인터페이스를 일관되게 구현할 수 있게 하는 것이다.

## 1. 설계 원칙

- 게임 룰과 학습 알고리즘을 분리한다.
- 환경은 합법 행동, 상태 전이, 승패, 이벤트 로그만 책임진다.
- agent는 관찰값과 action mask를 보고 행동을 선택한다.
- observation은 항상 현재 행동할 플레이어 관점으로 정규화한다.
- 지름길 선택은 action에 포함하지 않는다. 환경이 자동 처리한다.
- 윷 던지기는 agent의 action이 아니다. agent는 pool에 있는 결과를 어떤 말 또는 stack에 적용할지만 결정한다.

## 2. 핵심 용어

- `turn`: 한 플레이어가 윷을 던지고, pool의 결과를 모두 쓰거나 더 이상 움직일 수 없을 때까지의 구간.
- `decision step`: pool에 있는 윷 결과 하나를 선택해 말 또는 stack 하나에 적용하는 한 번의 의사결정.
- `pool`: 현재 turn에서 아직 사용하지 않은 윷 결과들의 multiset.
- `stack`: 같은 칸에 업힌 자기 말들의 묶음. 분리할 수 없고 함께 이동한다.
- `actor`: 현재 decision step에서 행동하는 플레이어.

## 3. Environment 구조

환경은 두 층으로 나눈다.

```text
GameState
  - 순수 게임 상태
  - 보드, 말 위치, stack, pool, 현재 플레이어, 승패 관리

YutnoriEnv
  - RL용 wrapper
  - reset, step, observation, legal action, reward, log 제공
```

권장 API:

```python
class YutnoriEnv:
    def reset(self, seed: int | None = None) -> dict:
        ...

    def step(self, action: int) -> tuple[dict, dict[int, float], bool, dict]:
        ...

    def get_observation(self, player_id: int | None = None) -> dict:
        ...

    def get_legal_actions(self) -> list[int]:
        ...

    def get_action_mask(self) -> np.ndarray:
        ...

    def render(self):
        ...
```

`step`의 reward는 scalar 하나가 아니라 `dict[player_id, reward]` 형태를 권장한다.
이 게임은 self-play 2인 zero-sum 환경이므로, 단순히 다음 observation을 표준 single-agent MDP처럼 처리하면 턴 전환 시 관점이 뒤집히는 문제가 생긴다.

## 4. Turn과 Pool 처리

환경은 항상 agent가 행동해야 하는 decision state만 외부에 노출한다.

턴 시작 절차:

```text
1. 현재 플레이어가 윷을 한 번 던진다.
2. 결과가 윷/모이면 자동 보너스 던지기를 즉시 수행한다.
3. 윷/모가 계속 나오면 도/개/걸이 나올 때까지 반복한다.
4. 나온 모든 결과를 pool에 누적한다.
5. 합법 action이 있으면 agent에게 decision state를 제공한다.
```

decision step 처리:

```text
1. agent가 action = (piece_id, yut_type)을 선택한다.
2. 환경이 action legality를 검증한다.
3. pool에서 해당 yut_type 하나를 제거한다.
4. 선택한 말이 속한 stack 전체를 이동한다.
5. 도착 칸에서 골인, 잡기, 업기, 지름길 진입 이벤트를 판정한다.
6. 실제 잡기가 발생했고 사용한 결과가 도/개/걸이면 보너스 던지기 1회를 수행한다.
7. 이 보너스 던지기에서 윷/모가 나오면 자동 보너스 던지기도 이어서 수행하고, 결과를 pool에 추가한다.
8. 승리 조건을 만족하면 episode를 종료한다.
9. pool이 비었거나 합법 action이 없으면 턴을 종료하고 상대 턴을 시작한다.
10. pool에 합법 action이 남아 있으면 같은 플레이어가 계속 행동한다.
```

중요한 규칙:

- 잡을 수 있는 action이 존재하는 것만으로는 보너스 던지기가 발생하지 않는다.
- 실제로 상대 말을 잡는 이동을 수행한 경우에만 잡기 보너스가 발생한다.
- 윷/모로 상대 말을 잡아도 잡기 보너스는 추가하지 않는다. 윷/모의 자동 보너스는 이미 윷을 던진 시점에 pool에 반영된다.

## 5. Action 설계

뒷도를 제외하므로 윷 결과는 5개다.

```python
YUT_TYPES = ["DO", "GAE", "GEOL", "YUT", "MO"]
YUT_STEPS = [1, 2, 3, 4, 5]
```

고정 action space:

```python
action = piece_id * 5 + yut_type_id
```

- `piece_id`: 0~3
- `yut_type_id`: 0~4
- 총 action 수: `4 * 5 = 20`

디코딩:

```python
piece_id = action // 5
yut_type_id = action % 5
```

합법 action 조건:

- pool에 해당 `yut_type`이 1개 이상 있어야 한다.
- 선택한 말이 `FINISHED` 상태가 아니어야 한다.
- 선택한 말이 stack에 속해 있다면 stack 대표 말이어야 한다.
- 선택한 말 또는 stack이 룰상 이동 가능해야 한다.

stack 대표 말:

- 같은 stack에 여러 말이 있을 때 가장 작은 `piece_id`를 대표 말로 둔다.
- 대표 말이 아닌 말에 대한 action은 illegal로 masking한다.
- 이렇게 해야 같은 이동을 의미하는 중복 action이 생기지 않는다.

## 6. Board와 Position 설계

보드는 그래프 또는 route table로 구현한다.

권장 내부 표현:

```python
class Piece:
    status: PieceStatus
    node_id: int | None
```

`node_id`는 이동 방향을 알 수 있도록 필요한 경우 route 정보를 포함한 논리 노드로 둔다.
단, 잡기와 업기는 실제 물리 칸 기준으로 판정해야 하므로 board는 다음 매핑을 제공해야 한다.

```python
logical_node_id -> physical_cell_id
```

보드 이동 API:

```python
class Board:
    def move(self, node_id: int | None, steps: int) -> MoveResult:
        ...
```

```python
class MoveResult:
    status: PieceStatus
    node_id: int | None
    physical_cell_id: int | None
    entered_shortcut: bool
    passed_start: bool
    landed_on_start: bool
```

이동 규칙:

- `WAITING` 말은 양수 이동 결과를 사용하면 출발점에서 해당 step만큼 이동해 보드에 진입한다.
- 출발점을 통과하면 `FINISHED`가 된다.
- 출발점에 정확히 도착하면 `ON_BOARD`로 남아 있고, 다음 양수 이동 결과를 쓰면 `FINISHED`가 된다.
- 모서리 또는 중앙 분기점에 정확히 멈추면 다음 이동부터 자동으로 지름길 경로를 따른다.
- 분기점을 지나치기만 하면 기존 경로를 유지한다.

## 7. State와 Observation

내부 상태는 절대 플레이어 ID 기준으로 저장한다.
observation은 항상 현재 actor 관점으로 변환한다.

1차 observation 구조:

```python
observation = {
    "my_positions": np.ndarray,      # shape: (4,)
    "my_status": np.ndarray,         # shape: (4,)
    "my_stack_matrix": np.ndarray,   # shape: (4, 4)
    "opp_positions": np.ndarray,     # shape: (4,)
    "opp_status": np.ndarray,        # shape: (4,)
    "opp_stack_matrix": np.ndarray,  # shape: (4, 4)
    "pool_counts": np.ndarray,       # shape: (5,)
    "action_mask": np.ndarray,       # shape: (20,)
}
```

position encoding:

- `WAITING`: 별도 special index.
- `FINISHED`: 별도 special index.
- `ON_BOARD`: board의 logical node id.

stack matrix:

```text
stack_matrix[i][j] = 1 if piece i and piece j are in the same stack else 0
```

DNN 입력용 vector는 별도 encoder에서 만든다.

```python
class ObservationEncoder:
    def encode(self, observation: dict) -> np.ndarray:
        ...
```

권장 vector encoding:

- 내 말 위치 one-hot.
- 내 말 상태 one-hot.
- 내 말 stack matrix flatten.
- 상대 말 위치 one-hot.
- 상대 말 상태 one-hot.
- 상대 말 stack matrix flatten.
- pool count 5차원.
- action mask 20차원은 네트워크 입력에 포함할 수도 있고, policy 출력 후 masking에만 사용할 수도 있다.

## 8. Policy 설계

모든 알고리즘의 policy/action-value 출력 크기는 20으로 고정한다.

Policy-gradient 계열:

```python
policy(observation) -> logits  # shape: (20,)
```

Value-based 계열:

```python
q_network(observation) -> q_values  # shape: (20,)
```

action mask 적용:

- policy-gradient: illegal action logit을 매우 작은 값으로 바꾼 뒤 softmax.
- value-based: illegal action Q-value를 매우 작은 값으로 바꾼 뒤 argmax.

공통 유틸:

```python
def mask_logits(logits, action_mask):
    ...

def mask_q_values(q_values, action_mask):
    ...
```

## 9. Reward 설계

1차 구현은 sparse terminal reward만 사용한다.

```python
winner: +1.0
loser:  -1.0
else:    0.0
```

`step`은 다음 형태의 reward dict를 반환한다.

```python
rewards = {
    0: 0.0,
    1: 0.0,
}
```

episode가 끝나면:

```python
rewards[winner] = +1.0
rewards[loser] = -1.0
```

잡기, 업기, 지름길 진입, 골인은 1차 reward에 넣지 않는다.
이 이벤트들은 `info`와 episode log에만 기록한다.

shaped reward는 환경이 안정된 뒤 별도 실험 설정으로 추가한다.

## 10. Agent Interface

모든 agent는 같은 interface를 따른다.

```python
class BaseAgent:
    def select_action(
        self,
        observation: dict,
        action_mask: np.ndarray,
        training: bool,
    ) -> int:
        ...

    def observe(self, transition) -> None:
        ...

    def update(self) -> dict:
        ...

    def save(self, path: str) -> None:
        ...

    def load(self, path: str) -> None:
        ...
```

transition 기본 구조:

```python
class Transition:
    player_id: int
    obs: dict
    action: int
    reward: float
    next_obs: dict | None
    done: bool
    action_mask: np.ndarray
    next_action_mask: np.ndarray | None
    info: dict
```

주의:

- 턴이 상대에게 넘어가면 `next_obs`의 관점도 바뀐다.
- 따라서 표준 single-agent DQN 업데이트를 그대로 적용하면 안 된다.
- self-play trainer가 player별 trajectory를 관리하거나, value-based 알고리즘에서는 zero-sum 관점 전환을 반영해야 한다.

## 11. Self-play 학습 구조

1차 구조:

```text
shared_agent vs shared_agent
```

- 같은 agent가 양쪽 플레이어를 모두 담당한다.
- 매 episode 선공은 랜덤으로 정한다.
- observation은 항상 현재 actor 관점으로 제공한다.
- episode log에는 실제 player_id를 함께 저장한다.
- 학습 데이터는 player별 trajectory로 분리해 최종 승패 reward를 부여한다.

추후 안정화 옵션:

```text
current_agent vs frozen_snapshot_agent
current_agent vs snapshot_pool
current_agent vs baseline_agents
```

처음부터 snapshot pool을 넣으면 구현 복잡도가 올라가므로, 1차 구현은 shared self-play로 시작한다.

## 12. Baseline Agent

환경 검증을 위해 RL agent보다 baseline agent를 먼저 구현한다.

필수 baseline:

- `RandomAgent`: legal action 중 균등 샘플링.
- `GreedyFinishAgent`: 골인 또는 가장 멀리 전진하는 action 선호.
- `CaptureFirstAgent`: 잡기 가능한 action을 최우선 선택.

baseline은 다음 용도로 사용한다.

- action mask 검증.
- 게임 종료 검증.
- self-play loop 검증.
- 최종 승률 평가의 기준 상대.

## 13. 로그와 Info

`step`의 `info`에는 해당 decision step에서 발생한 이벤트를 담는다.

```python
info = {
    "actor": int,
    "action": int,
    "piece_id": int,
    "yut_type": str,
    "captured": bool,
    "captured_count": int,
    "stacked": bool,
    "stack_size": int,
    "entered_shortcut": bool,
    "finished_count": int,
    "bonus_rolls": list[str],
    "pool_counts": np.ndarray,
    "turn_changed": bool,
    "winner": int | None,
}
```

episode log에는 다음을 누적한다.

- 선공 플레이어.
- 각 턴의 초기 pool.
- pool에서 선택한 결과 순서.
- action sequence.
- 잡기 횟수.
- 업기 횟수.
- 지름길 진입 횟수.
- 골인한 말 수.
- 총 decision step 수.
- 총 turn 수.
- 승자.

## 14. 구현 순서

권장 구현 순서:

1. `YutSampler`: 윷 결과 샘플링.
2. `Board`: 보드 그래프와 이동 결과.
3. `GameState`: 말, stack, pool, 턴, 승패.
4. `YutnoriEnv`: reset, step, observation, action mask.
5. `RandomAgent`: 환경 smoke test.
6. `CaptureFirstAgent`, `GreedyFinishAgent`: baseline 검증.
7. `SelfPlayRunner`: episode 실행과 trajectory 수집.
8. RL 알고리즘 연결.

## 15. 아직 남은 결정

다음 항목은 환경 1차 구현 후 결정한다.

- 비교할 강화학습 알고리즘.
- shaped reward 사용 여부와 구체식.
- episode 길이 제한.
- 선공 이점 보정 평가 방식.
- state vector의 최종 차원과 압축 여부.
