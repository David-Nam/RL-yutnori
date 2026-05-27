# M4 Gymnasium 환경 Wrapper 구현 및 검증 보고서

## 1. 단계 요약

M4에서는 `GameState` 룰 엔진을 Gymnasium 호환 환경으로 감싸는 `YutnoriEnv`를 구현했다.

목표는 학습 알고리즘이 사용할 수 있는 다음 인터페이스를 제공하는 것이다.

- `reset(seed=None) -> (obs, info)`
- `step(action) -> (obs, reward, terminated, truncated, info)`
- `action_masks() -> np.ndarray[bool]`

환경은 단일 learner 관점으로 동작한다. learner가 action을 선택하면, 상대 player의 턴은 환경 내부에서 자동으로 진행하고 learner의 다음 decision state를 반환한다.

## 2. 변경 파일

- `yutnori/env/yutnori_env.py`
  - `YutnoriEnv`
  - `encode_observation`
  - observation 상수
  - opponent 자동 진행 로직
- `yutnori/env/__init__.py`
  - env public export 추가
- `tests/test_yutnori_env.py`
  - M4 env wrapper 단위 테스트 추가

## 3. 구현 내용

### 3.1 Gymnasium API

`YutnoriEnv`는 Gymnasium `Env`를 상속한다.

```python
action_space = Discrete(20)
observation_space = Box(shape=(53,), dtype=np.float32)
```

action은 기존 설계대로 유지한다.

```python
action = piece_id * 5 + yut_type_id
```

총 action 수는 `4 pieces x 5 yut types = 20`이다.

### 3.2 Observation

observation은 learner 관점의 고정 길이 vector다.

구성:

- learner 말 위치 4개
- learner 말 상태 4개
- learner stack matrix 16개
- opponent 말 위치 4개
- opponent 말 상태 4개
- opponent stack matrix 16개
- pool count 5개

총 길이:

```text
(4 + 4 + 16) * 2 + 5 = 53
```

### 3.3 Action Mask

`action_masks()`는 현재 learner가 선택 가능한 action만 `True`로 반환한다.

mask 조건:

- 현재 player가 learner여야 한다.
- pool에 해당 윷 결과가 있어야 한다.
- 선택한 말이 `FINISHED`가 아니어야 한다.
- stack에 속한 말이면 대표 말만 선택 가능하다.

### 3.4 Opponent 자동 진행

reset 또는 step 이후 현재 player가 opponent라면, 환경이 opponent action을 자동 실행한다.

흐름:

```text
learner action
-> GameState.apply_action
-> opponent turn이면 legal action 선택
-> opponent action 반복
-> learner turn 또는 game 종료 시 반환
```

M4에서는 정식 baseline agent를 아직 만들지 않았다. opponent는 기본적으로 legal action 중 하나를 선택하는 내부 policy를 사용한다. 정식 `RandomAgent`, `CaptureFirstAgent`, `GreedyFinishAgent`는 M5 범위다.

### 3.5 Reward

M4의 reward는 terminal sparse reward다.

- learner 승리: `+1.0`
- learner 패배: `-1.0`
- 그 외: `0.0`

`truncated`는 항상 `False`다. episode 길이 제한은 아직 도입하지 않았다.

## 4. 검증 방식

다음 명령으로 검증했다.

```bash
python -m compileall yutnori tests
python -m pytest
```

추가로 M4 계획의 smoke 기준에 맞춰 mask-aware rollout 100판을 pytest 테스트에 포함했다.

## 5. 검증 로직

### 5.1 Reset / Observation / Mask

고정 sampler로 첫 윷 결과를 `GAE`로 만들었다.

검증 내용:

- reset 결과 observation shape가 `(53,)`인지 확인.
- dtype이 `np.float32`인지 확인.
- observation이 `observation_space`에 포함되는지 확인.
- pool에 `GAE`만 있으므로 4개 말의 `GAE` action만 mask에 켜지는지 확인.

### 5.2 Seed 재현성

같은 seed로 두 개의 env를 reset했다.

검증 내용:

- initial rolls가 같은지 확인.
- observation이 같은지 확인.
- action mask가 같은지 확인.

### 5.3 Learner 관점 Observation

`learner_player=1`로 환경을 만들고 reset했다.

검증 내용:

- observation의 앞쪽 말 위치/상태가 player 1 기준인지 확인.
- 초기 learner 말들이 모두 `WAITING`으로 인코딩되는지 확인.

### 5.4 Stack 대표 말 Action Mask

learner의 0번 말과 1번 말을 같은 칸에 배치했다.

검증 내용:

- 대표 말인 0번 말 action은 legal.
- 같은 stack의 비대표 말인 1번 말 action은 illegal.

### 5.5 Step 이후 다음 Decision State 반환

learner가 action을 수행한 뒤 opponent에게 턴이 넘어가는 상황을 만들었다.

검증 내용:

- learner event가 info에 기록되는지 확인.
- opponent event가 자동으로 실행되는지 확인.
- 최종 반환 상태가 다시 learner decision state인지 확인.
- 다음 learner pool에 맞는 action mask가 반환되는지 확인.

### 5.6 Opponent 선공 자동 진행

starting player를 opponent로 설정했다.

검증 내용:

- reset 도중 opponent action이 자동 실행되는지 확인.
- reset 반환 시점에는 learner가 action할 수 있는 상태인지 확인.

### 5.7 Illegal Action

pool에 없는 윷 결과 action을 learner가 선택하도록 했다.

검증 내용:

- `ValueError`가 발생하는지 확인.

### 5.8 Terminal Reward

learner가 마지막 말을 골인시키는 상태를 직접 구성했다.

검증 내용:

- reward가 `+1.0`인지 확인.
- `terminated=True`인지 확인.
- `truncated=False`인지 확인.
- winner가 learner인지 확인.
- 종료 후 action mask가 모두 `False`인지 확인.

### 5.9 Gymnasium Space 계약

Gymnasium 기본 checker는 action mask를 모르고 invalid action을 샘플링한다. 이 프로젝트는 invalid action을 예외로 처리하므로 기본 checker를 pass/fail 기준으로 쓰지 않았다.

대신 다음을 검증했다.

- `action_space.n == 20`
- reset observation이 `observation_space`에 포함됨.
- reset info가 dict로 반환됨.

### 5.10 Mask-aware Rollout 100판

seed 0부터 99까지 100개 episode를 실행했다.

각 episode에서:

- `action_masks()`에서 legal action을 가져온다.
- legal action 중 첫 번째를 선택한다.
- non-terminal 상태에서 legal action이 없는 경우 실패 처리한다.
- 5000 learner step 안에 종료되지 않으면 실패 처리한다.
- `truncated`가 발생하면 실패 처리한다.

결과:

- 100개 episode 모두 정상 종료.
- illegal action 없음.
- truncated 없음.

## 6. 실행 결과

```text
python -m compileall yutnori tests
```

결과:

- 통과.

```text
python -m pytest
```

결과:

```text
34 passed in 0.36s
```

테스트 구성:

- board tests: 10개
- game state tests: 10개
- yut sampler tests: 3개
- env tests: 10개
- skeleton test: 1개

## 7. 예상과 달랐던 점

### 7.1 Gymnasium 미설치

로컬 Python 환경에 `gymnasium`이 설치되어 있지 않았다.

조치:

- `gymnasium`을 설치했다.
- `numpy`는 이미 설치되어 있었다.

### 7.2 Gymnasium 기본 checker와 action mask 충돌

Gymnasium 기본 checker는 action mask를 고려하지 않고 `action_space.sample()`로 action을 넣는다.
이 프로젝트의 env는 invalid action을 `ValueError`로 처리하므로 checker와 충돌했다.

조치:

- 기본 checker를 통과 조건으로 사용하지 않았다.
- 대신 reset/space 계약과 mask-aware rollout으로 검증했다.

## 8. 해석

M4 구현으로 학습 알고리즘이 사용할 수 있는 최소 Gymnasium wrapper가 준비됐다.

현재 보장되는 것:

- learner 관점 observation 반환.
- legal action mask 반환.
- learner action 적용.
- opponent turn 자동 진행.
- terminal reward 반환.
- invalid action 예외 처리.
- 100판 mask-aware rollout 안정성.

## 9. 남은 리스크

- opponent는 아직 정식 baseline agent가 아니다.
- `RandomAgent`, `CaptureFirstAgent`, `GreedyFinishAgent`는 M5에서 구현해야 한다.
- observation vector는 학습 가능 형태지만 feature engineering은 아직 최소 수준이다.
- PPO/C51 연결은 아직 구현하지 않았다.
- episode 길이 제한은 아직 없다.

## 10. 다음 단계

다음 단계는 M5 Baseline Agent 구현이다.

구현 대상:

- `RandomAgent`
- `CaptureFirstAgent`
- `GreedyFinishAgent`
- baseline 대전 runner 또는 smoke 평가
- baseline들이 illegal action을 선택하지 않는지 검증
- `Random vs Random` 1000판 종료 검증
