# 공통 Rule-based Agent 평가 가이드라인

## 1. 목적

각 팀원은 모델 구조, observation, reward, 학습 알고리즘을 자유롭게
설계하고 학습한다.

최종 비교에서는 학습 환경의 차이가 아니라 완성된 agent의 게임 수행
능력을 비교해야 한다. 이를 위해 모든 모델을 동일한 게임 환경과 동일한
Rule-based Agent를 상대로 평가한다.

공통으로 고정할 항목은 다음과 같다.

- 게임 규칙과 보드 경로
- Rule-based Agent 구현
- 윷 결과 확률
- 선공과 후공 배정
- 평가 seed
- 정책 실행 방식
- 게임 종료와 오류 처리
- 승률 집계 및 보고 방식

핵심 비교 지표는 다음과 같다.

```text
공통 Rule-based Agent 상대 5,000판 전체 승률
```

## 2. 기본 원칙

```text
학습 환경: 각 팀원 자유
평가 환경: 팀 공통 환경
평가 상대: 공통 Rule-based Agent
평가 중 학습 및 가중치 변경: 금지
```

각 모델은 공통 평가 환경의 상태와 action 체계에 연결하기 위한 adapter를
제공해야 한다.

권장 인터페이스는 다음과 같다.

```python
def select_action(observation, legal_actions) -> int:
    ...
```

또는 action mask를 사용하는 경우 다음 형태를 사용할 수 있다.

```python
def select_action(observation, action_mask) -> int:
    ...
```

## 3. Agent가 사용할 수 있는 정보

평가 agent가 사용할 수 있는 정보:

- 현재 observation
- 현재 legal actions 또는 action mask
- 현재까지 공개된 게임 정보
- 모델 자체의 고정된 내부 상태

평가 agent가 사용할 수 없는 정보:

- 환경의 RNG 객체 또는 내부 RNG 상태
- 다음에 나올 윷 결과
- 미리 생성된 전체 윷 결과 sequence
- 상대 agent의 내부 상태나 action score
- 공통 평가 seed 목록을 이용한 사전 최적화 정보

행동을 시뮬레이션하는 agent는 미래 윷 결과를 볼 수 없어야 한다.
시뮬레이션용 환경 복사본에 실제 평가 환경의 RNG 상태를 전달해서는 안
된다.

## 4. 공통 Rule-based Agent

모든 모델은 동일한 Rule-based Agent 구현을 상대한다. 각 팀원이 별도로
재구현한 Rule-based Agent를 평가에 사용해서는 안 된다.

Rule-based Agent는 legal action마다 다음 점수를 계산한다.

```text
완주:                       +100
상대 말 잡기:                +50
대기 중인 말 출발:            +5
업힌 말 이동: +4 * (stack_size - 1)
완주 거리:    -0.5 * distance_to_finish
```

추가 정의:

- 잡기 점수는 잡힌 말이나 stack 크기와 관계없이 action당 한 번 적용한다.
- 완주 점수는 이동한 stack 크기와 관계없이 action당 한 번 적용한다.
- 모든 legal action의 점수가 같을 때도 항상 동일한 action이 선택되어야
  한다.
- 동점일 때는 가장 작은 action ID를 선택하는 것을 기본 규칙으로 한다.
- action ID 체계와 `distance_to_finish` 계산은 공통 평가 엔진의 정의를
  따른다.

Rule-based Agent의 구현과 버전은 평가 시작 전에 동결한다.

## 5. 공통 게임 규칙

### 5.1 기본 구성

```text
플레이어 수: 2명
플레이어당 말: 4개
뒷도: 사용하지 않음
낙: 사용하지 않음
후진 이동: 사용하지 않음
플레이어의 지름길 선택: 사용하지 않음
```

### 5.2 잡기와 업기

- 잡기와 업기는 도착 칸에서만 발생한다.
- 이동 중 지나가는 칸에서는 잡기나 업기가 발생하지 않는다.
- 자기 말이 있는 칸에 도착하면 자동으로 업는다.
- 업힌 말은 분리할 수 없으며 stack 전체가 함께 이동한다.
- 상대 stack이 있는 칸에 도착하면 해당 stack 전체를 잡는다.
- 잡힌 상대 말은 모두 대기 상태로 돌아간다.

### 5.3 지름길

- 말이 분기점에 정확히 도착하면 다음 이동부터 지름길에 자동 진입한다.
- 분기점을 지나치기만 하면 기존 경로를 유지한다.
- 경로 선택은 agent의 action에 포함하지 않는다.
- 공통 보드의 물리 칸과 논리 경로는 평가 전에 route table로 고정한다.

### 5.4 골인

- 말은 HOME을 통과하면 `FINISHED` 상태가 된다.
- HOME에 정확히 도착한 경우 즉시 완주하지 않고 보드 위에 남는다.
- HOME에 머문 말은 다음 양수 이동 결과를 사용하면 `FINISHED`가 된다.
- HOME에 있는 상대 말도 잡을 수 있다.
- stack이 HOME을 통과하면 stack의 모든 말이 함께 완주한다.

### 5.5 제외할 특수 규칙

- 윷 또는 모가 연속 20번 나온 경우의 즉시 승리는 사용하지 않는다.
- 모델별 환경에만 존재하는 별도 승리 조건은 평가에서 사용하지 않는다.

## 6. 윷 결과와 확률

모든 평가에서 다음 확률을 사용한다.

| 결과 | 이동량 | 확률 |
| --- | ---: | ---: |
| 도 | 1 | 0.1536 |
| 개 | 2 | 0.3456 |
| 걸 | 3 | 0.3456 |
| 윷 | 4 | 0.1296 |
| 모 | 5 | 0.0256 |

확률 합계는 정확히 `1.0`이다.

### 6.1 추가 던지기

- 윷 또는 모가 나오면 추가로 던진다.
- 윷 또는 모가 연속으로 나오면 도, 개 또는 걸이 나올 때까지 계속 던진다.
- 상대 말을 실제로 잡으면 추가 던지기 1회를 얻는다.
- 윷 또는 모를 사용해 상대 말을 잡았을 때 추가 던지기를 중복 지급하지
  않는다.
- 한 턴에 나온 결과는 pool에 저장한다.
- agent는 pool에 저장된 결과를 원하는 순서로 사용할 수 있다.
- pool이 비거나 legal action이 없으면 턴을 종료한다.

## 7. 선공과 후공

선공과 후공은 정확히 같은 수로 평가한다.

공식 평가는 2,500개의 평가 seed를 사용하고, seed마다 두 게임을 수행한다.

```text
Game A: 평가 모델 선공, Rule-based Agent 후공
Game B: Rule-based Agent 선공, 평가 모델 후공
```

따라서 모델별 총 평가 판수는 다음과 같다.

```text
평가 모델 선공: 2,500판
평가 모델 후공: 2,500판
총 평가:       5,000판
```

Game A와 Game B에는 같은 기본 평가 seed를 사용한다. 단, 두 게임의 RNG가
서로 간섭하지 않도록 각각 독립된 환경 인스턴스를 생성한다.

모든 팀원의 모델은 동일한 평가 seed 목록을 사용한다.

## 8. 정책 실행 방식

공식 평가는 기본적으로 deterministic policy를 사용한다.

```text
PPO: legal action 중 최대 logit
DQN: legal action 중 최대 Q-value
Value model: 가장 높은 평가 점수의 action
Rule 또는 Hybrid: 동결된 고정 점수식
```

평가 시 다음 조건을 적용한다.

- 모델을 evaluation mode로 전환한다.
- dropout과 학습 전용 noise를 비활성화한다.
- exploration 및 epsilon-greedy를 비활성화한다.
- replay buffer를 업데이트하지 않는다.
- optimizer step과 가중치 업데이트를 수행하지 않는다.
- 평가 결과에 따라 모델 상태나 정책을 변경하지 않는다.

본질적으로 stochastic한 정책만 예외적으로 stochastic evaluation을 사용할
수 있다. 이 경우 정책 RNG seed를 고정하고 결과에 명시해야 하며,
deterministic 결과와 분리해 보고한다.

## 9. 모델 유형 표시

모델 구조와 학습 방법은 제한하지 않지만, 결과표에는 모델 유형을 명확히
표시한다.

권장 분류:

```text
Pure RL
Rule-based
RL + Rule Hybrid
Search + Value
Imitation + RL
```

예:

- PPO logits만 사용하는 모델: `Pure RL`
- Value Network와 수작업 전략 점수를 혼합한 모델: `RL + Rule Hybrid`
- PPO logits에 tactical prior를 더하는 모델: `RL + Rule Hybrid`

Hybrid 모델의 사용을 금지하지는 않는다. 다만 순수 학습 모델과 구분할 수
있도록 사용한 rule, search, override 또는 tactical prior를 공개해야 한다.

## 10. Action Adapter와 합법성

팀원별 학습 환경의 action encoding이 다를 수 있으므로 평가 adapter가
공통 action 체계로 변환해야 한다.

Adapter의 책임:

- 공통 observation을 모델 입력 형식으로 변환
- 모델 출력을 공통 action ID로 변환
- legal action 또는 action mask 적용
- 말과 윷 결과의 순서를 정확히 매핑

모델이 illegal action을 반환하면 다음과 같이 처리한다.

```text
해당 게임: 평가 모델의 패배
illegal action count: 1 증가
평가 로그에 상태와 action 기록
```

공식 결과에서 illegal action은 반드시 별도로 보고한다.

## 11. 게임 종료와 비정상 상황

정상 종료 조건:

- 한 플레이어의 말 4개가 모두 `FINISHED` 상태가 된 경우

안전 제한:

```text
게임당 최대 decision 수: 10,000
```

최대 decision 수를 초과한 게임은 임의로 승자를 정하지 않는다. 해당 게임은
`evaluation_error`로 기록하고 원인을 조사한 뒤 전체 평가를 다시 실행한다.

다음 상황도 평가 오류로 처리한다.

- 환경 또는 모델 예외
- observation shape 불일치
- action 변환 실패
- non-terminal 상태에서 legal action이 없음
- 동일 seed에서 재실행 결과가 달라지는 비결정적 오류

## 12. 공식 보고 지표

각 모델은 최소한 다음 결과를 제출한다.

| 지표 | 설명 |
| --- | --- |
| 전체 승률 | 전체 5,000판의 승률 |
| 선공 승률 | 모델이 선공인 2,500판의 승률 |
| 후공 승률 | 모델이 후공인 2,500판의 승률 |
| 95% 신뢰구간 | 전체 승률의 통계적 불확실성 |
| 평균 turn 수 | 게임당 평균 turn |
| 평균 decision 수 | 게임당 전체 action 수 |
| illegal action 수 | 모델이 반환한 illegal action 수 |
| evaluation error 수 | 정상 종료하지 못한 게임 수 |
| 평균 평가 시간 | 게임 또는 전체 평가 실행 시간 |

기본 순위는 전체 승률로 정한다.

승률이 유사한 경우 다음 순서로 참고한다.

1. 전체 승률의 95% 신뢰구간
2. 후공 승률
3. 여러 학습 seed 간 표준편차
4. illegal action 및 evaluation error 여부

평균 turn이나 실행 시간은 성능 순위의 기본 기준으로 사용하지 않는다.

## 13. 여러 학습 Seed

가능하면 각 모델을 최소 3개의 학습 seed로 학습한다.

```text
권장 학습 seed: 0, 1, 2
checkpoint별 평가: 5,000판
모델 방식별 총 평가: 15,000판
```

보고 항목:

- seed별 전체 승률
- seed별 선공 및 후공 승률
- 평균 승률
- 표준편차
- 최저 및 최고 승률

단일 checkpoint만 제출한 경우 결과에 `single training seed`임을 명시한다.
단일 checkpoint의 결과를 여러 seed 평균과 같은 안정성으로 해석해서는 안
된다.

## 14. 제출물

각 팀원은 다음 항목을 제출한다.

```text
1. 학습 완료 checkpoint
2. 공통 평가 환경용 agent adapter
3. 모델 설정 파일
4. 모델 유형
5. 학습 seed
6. 평가 실행 명령
7. Python 및 라이브러리 버전
8. 학습 코드의 git commit SHA
9. 모델이 사용하는 rule, search 또는 tactical prior 설명
```

최종 공식 평가는 가능하면 한 명이 동일한 머신과 동일한 공통 평가 코드로
모든 checkpoint를 실행한다.

## 15. 결과표 예시

| Agent | 유형 | Seed | 전체 승률 | 선공 승률 | 후공 승률 | 95% CI | Illegal |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: |
| Agent A | Pure RL | 0 | 0.000 | 0.000 | 0.000 | `[0.000, 0.000]` | 0 |
| Agent B | RL + Rule Hybrid | 0 | 0.000 | 0.000 | 0.000 | `[0.000, 0.000]` | 0 |

여러 seed의 최종 요약:

| Agent 방식 | Seed 수 | 평균 승률 | 표준편차 | 최저 | 최고 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Agent A | 3 | 0.000 | 0.000 | 0.000 | 0.000 |
| Agent B | 3 | 0.000 | 0.000 | 0.000 | 0.000 |

## 16. 팀 합의 문구

> 모델 구조, observation, reward, 학습 알고리즘과 학습 환경은 각 팀원이
> 자유롭게 결정한다. 최종 비교는 팀이 제공하는 동일한 게임 엔진과 동일한
> Rule-based Agent를 사용한다. 모든 모델은 동일한 2,500개 평가 seed에
> 대해 선공과 후공을 한 번씩 수행하여 총 5,000판을 평가한다. 윷 확률,
> 보드 경로, 잡기, 업기, 골인, 추가 던지기 규칙과 Rule-based Agent의
> 점수식 및 동점 처리를 고정한다. 최종 결과에는 전체 승률, 선공 승률,
> 후공 승률, 95% 신뢰구간, illegal action 수와 모델 유형을 함께 보고한다.
