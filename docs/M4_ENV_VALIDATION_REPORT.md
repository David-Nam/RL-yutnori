# M4 Gymnasium 환경 Wrapper 구현 및 검증 보고서

## 1. 변경 요약

M4 단계에서는 `GameState` 룰 엔진을 Gymnasium 호환 환경으로 감싸는 wrapper를 구현했다.

추가/변경 파일:

- `yutnori/env/yutnori_env.py`
  - `YutnoriEnv`
  - `reset(seed=None) -> (obs, info)`
  - `step(action) -> (obs, reward, terminated, truncated, info)`
  - `action_masks() -> np.ndarray[bool]`
  - learner 기준 vector observation
  - opponent turn 자동 진행
- `yutnori/env/__init__.py`
  - env public export 추가
- `tests/test_yutnori_env.py`
  - M4 env wrapper 단위 테스트 추가

설치한 의존성:

- `gymnasium`
- `numpy`는 이미 설치되어 있었다.

## 2. 구현 내용

### 2.1 Gymnasium API

`YutnoriEnv`는 다음 Gymnasium API를 제공한다.

```python
reset(seed=None) -> (obs, info)
step(action) -> (obs, reward, terminated, truncated, info)
action_masks() -> np.ndarray[bool]
```

환경의 action space는 다음과 같다.

```python
action_space = Discrete(20)
```

20개 action은 기존 설계대로 다음 인코딩을 사용한다.

```python
action = piece_id * 5 + yut_type_id
```

### 2.2 Learner 관점 환경

M4 env는 단일 learner 관점으로 동작한다.

- `learner_player`가 학습 대상이다.
- opponent turn은 env 내부에서 자동으로 진행한다.
- learner가 action을 수행한 뒤 opponent turn이 발생하면, env는 opponent action들을 실행하고 learner의 다음 decision state를 반환한다.
- 따라서 `step()` 호출자는 항상 learner가 선택해야 할 상태만 받는다.

### 2.3 Observation

observation은 고정 길이 `np.float32` vector다.

포함 정보:

- learner 말 4개의 위치
- learner 말 4개의 상태
- learner stack matrix 4x4
- opponent 말 4개의 위치
- opponent 말 4개의 상태
- opponent stack matrix 4x4
- 현재 pool counts 5개

특수 위치 값:

- `WAITING`: 29
- `FINISHED`: 30

### 2.4 Action Mask

`action_masks()`는 shape `(20,)`의 boolean vector를 반환한다.

mask는 다음 조건을 반영한다.

- pool에 해당 윷 결과가 있어야 한다.
- 선택한 말이 `FINISHED`가 아니어야 한다.
- stack에 속한 말이면 대표 말만 legal action이 된다.
- 게임이 끝났거나 learner 차례가 아니면 모든 action이 false다.

### 2.5 Reward

M4에서는 sparse terminal reward만 사용한다.

- learner 승리: `+1.0`
- learner 패배: `-1.0`
- 그 외: `0.0`

`truncated`는 항상 `False`다. episode 길이 제한은 아직 도입하지 않았다.

## 3. 검증 방식

실행한 명령:

```bash
python -m compileall yutnori tests
python -m pytest
```

추가로 mask-aware rollout 100판을 pytest 테스트에 포함했다.

## 4. 검증 로직

### 4.1 reset 반환값 검증

테스트:

- `test_reset_returns_vector_observation_and_mask_for_learner_turn`

검증 내용:

- fixed sampler로 첫 윷 결과를 `GAE`로 고정했다.
- `reset()`이 observation과 info를 반환하는지 확인했다.
- observation shape이 `OBSERVATION_SIZE`와 일치하는지 확인했다.
- observation dtype이 `np.float32`인지 확인했다.
- observation이 `observation_space`에 포함되는지 확인했다.
- pool에 `GAE`만 있을 때 4개 말의 `GAE` action만 legal인지 확인했다.

### 4.2 seed 재현성 검증

테스트:

- `test_reset_seed_reproducibly_returns_same_initial_observation_and_mask`

검증 내용:

- 같은 seed로 두 env를 reset했다.
- initial rolls, observation, action mask가 동일한지 확인했다.

### 4.3 learner 관점 observation 검증

테스트:

- `test_observation_is_from_learner_perspective`

검증 내용:

- `learner_player=1`로 설정했다.
- observation 앞부분이 player 1의 말 상태를 나타내는지 확인했다.
- 초기 말 위치가 모두 `WAITING` 값으로 인코딩되는지 확인했다.

### 4.4 action mask의 stack 대표 말 검증

테스트:

- `test_action_masks_respect_stack_representative`

검증 내용:

- learner의 0번 말과 1번 말을 같은 칸에 둬 stack 상태를 만들었다.
- 대표 말인 0번 action은 legal인지 확인했다.
- 같은 stack의 비대표 말인 1번 action은 illegal인지 확인했다.

### 4.5 step 흐름 검증

테스트:

- `test_step_applies_learner_action_and_returns_next_decision_state`

검증 내용:

- learner가 `GAE` action을 수행하도록 했다.
- learner action 이후 턴이 opponent에게 넘어가는지 확인했다.
- env가 opponent action을 자동 실행하는지 확인했다.
- 이후 learner의 다음 decision state가 반환되는지 확인했다.
- 새 pool에 맞는 action mask가 반환되는지 확인했다.

### 4.6 opponent 선공 자동 진행 검증

테스트:

- `test_reset_auto_advances_opponent_until_learner_turn`

검증 내용:

- starting player를 opponent로 설정했다.
- reset 시 opponent가 먼저 action을 수행하는지 확인했다.
- env가 learner turn까지 자동 진행한 뒤 반환되는지 확인했다.

### 4.7 illegal action 검증

테스트:

- `test_illegal_action_raises_value_error`

검증 내용:

- pool에 없는 윷 결과 action을 넣었다.
- env가 `ValueError`를 발생시키는지 확인했다.

### 4.8 terminal reward 검증

테스트:

- `test_terminal_reward_when_learner_wins`

검증 내용:

- learner의 말 3개를 이미 `FINISHED`로 두고, 마지막 말만 골인 직전 위치에 배치했다.
- 마지막 말을 골인시키는 action을 수행했다.
- `reward == +1.0`인지 확인했다.
- `terminated == True`인지 확인했다.
- 게임 종료 후 action mask가 모두 false인지 확인했다.

### 4.9 Gymnasium space 계약 검증

테스트:

- `test_gymnasium_spaces_accept_reset_outputs`

검증 내용:

- `action_space.n == 20`인지 확인했다.
- reset observation이 `observation_space`에 포함되는지 확인했다.
- info가 dict인지 확인했다.

주의:

- Gymnasium 기본 `check_env`는 action mask를 모르고 invalid action을 샘플링한다.
- 우리 env는 illegal action을 명시적으로 `ValueError` 처리한다.
- 따라서 `check_env`를 pass/fail 기준으로 쓰지 않고, reset/space 계약과 mask-aware rollout으로 검증했다.

### 4.10 mask-aware random rollout 100판 검증

테스트:

- `test_mask_aware_random_rollouts_finish_without_illegal_actions`

검증 내용:

- seed 0~99에 대해 100개 episode를 실행했다.
- 매 decision state에서 `action_masks()`를 읽었다.
- legal action 중 하나를 선택해 `step()`을 호출했다.
- non-terminal learner state에서 legal action이 항상 존재하는지 확인했다.
- `truncated`가 발생하지 않는지 확인했다.
- 각 episode가 safety bound 안에서 종료되는지 확인했다.

## 5. 실행 결과

문법 검증:

```text
python -m compileall yutnori tests
```

결과:

- 통과

전체 테스트:

```text
python -m pytest
```

결과:

```text
34 passed in 0.36s
```

테스트 구성:

- board: 10개
- game state: 10개
- yut sampler: 3개
- env: 10개
- skeleton: 1개

## 6. 해석

M4 범위인 Gymnasium wrapper는 다음 조건을 만족한다.

- reset/step/action mask API가 동작한다.
- learner 관점 observation을 반환한다.
- opponent turn을 내부에서 자동 진행한다.
- terminal reward를 반환한다.
- illegal action을 명시적으로 거부한다.
- mask-aware rollout으로 실제 episode가 종료되는 것을 확인했다.

## 7. 예상과 달랐던 점

- `gymnasium`이 설치되어 있지 않아 설치했다.
- Gymnasium 기본 checker는 invalid action mask를 고려하지 않아 우리 env의 예외 정책과 충돌했다.
- 이에 따라 `check_env` 대신 reset/space 계약 테스트와 mask-aware rollout 테스트를 사용했다.

## 8. 남은 리스크

- opponent는 아직 정식 baseline agent가 아니다.
- observation vector는 동작하지만 학습 성능 관점의 feature 개선은 아직 하지 않았다.
- PPO/C51 연결은 아직 구현하지 않았다.
- M5에서 `RandomAgent`, `CaptureFirstAgent`, `GreedyFinishAgent`를 구현해야 한다.
