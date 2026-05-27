# M5 Baseline Agent 구현 및 검증 보고서

## 1. 단계 요약

M5에서는 강화학습 알고리즘을 붙이기 전에 환경 안정성을 검증할 baseline agent와 tournament runner를 구현했다.

구현 대상:

- `RandomAgent`
- `CaptureFirstAgent`
- `GreedyFinishAgent`
- `play_game`
- `run_tournament`

이 단계의 목적은 학습 전에도 합법 action 선택, 게임 종료, 승률/turn/decision metric 수집이 가능한지 확인하는 것이다.

## 2. 변경 파일

- `yutnori/agents/baseline.py`
  - baseline agent 3종 구현
  - action 평가 helper 구현
- `yutnori/agents/__init__.py`
  - baseline public export 추가
- `yutnori/eval/tournament.py`
  - 단일 게임 실행 helper
  - 여러 판 대전 runner
  - 결과 dataclass
- `yutnori/eval/__init__.py`
  - evaluation public export 추가
- `tests/test_baseline_agents.py`
  - baseline agent 선택 로직 테스트
- `tests/test_tournament.py`
  - baseline 대전 및 smoke 테스트

## 3. 구현 내용

### 3.1 RandomAgent

`RandomAgent`는 legal action 중 하나를 균등 무작위로 선택한다.

특징:

- seed를 받을 수 있다.
- `legal_actions`가 비어 있으면 `ValueError`를 발생시킨다.
- tournament smoke에서 illegal action을 선택하지 않는지 검증한다.

### 3.2 CaptureFirstAgent

`CaptureFirstAgent`는 실제 잡기가 가능한 action을 우선 선택한다.

선택 기준:

1. 잡는 말 수가 많은 action
2. 동시에 골인하는 말 수가 많은 action
3. 움직이는 stack 크기가 큰 action
4. 이동량이 큰 action
5. action id가 작은 action

잡기 가능한 action이 없으면 `GreedyFinishAgent`와 같은 기준으로 fallback한다.

중요한 점:

- 지나가는 칸의 상대 말은 잡기로 보지 않는다.
- 도착 칸에 있는 상대 말 또는 상대 stack만 잡기 대상으로 본다.

### 3.3 GreedyFinishAgent

`GreedyFinishAgent`는 골인과 전진을 우선하는 deterministic baseline이다.

선택 기준:

1. 골인하는 말 수가 많은 action
2. 움직이는 stack 크기가 큰 action
3. 지름길에 진입하는 action
4. 이동량이 큰 action
5. action id가 작은 action

### 3.4 Tournament Runner

`play_game`은 두 agent가 `GameState`에서 직접 대전하도록 실행한다.

반환 정보:

- winner
- starting player
- turn count
- decision count
- 선택적으로 event log

`run_tournament`는 여러 판을 실행하고 aggregate metric을 계산한다.

반환 정보:

- 총 게임 수
- player별 승수
- 선공 player 분포
- 평균 turn 수
- 평균 decision 수
- player별 승률 helper

## 4. 검증 방식

실행한 명령:

```bash
python -m compileall yutnori tests
python -m pytest
```

M5 핵심 smoke:

- `RandomAgent vs RandomAgent` 1000판 실행
- 모든 게임 종료 확인
- 승수 합계 확인
- 선공 분포 기록 확인
- 평균 turn/decision metric 산출 확인

## 5. 검증 로직

### 5.1 RandomAgent legal action 선택

테스트:

- `test_random_agent_selects_one_of_the_legal_actions`

검증 내용:

- pool에 `DO`만 있는 상태를 만든다.
- legal action 목록을 계산한다.
- `RandomAgent`가 선택한 action이 legal action 안에 있는지 확인한다.

### 5.2 CaptureFirstAgent 실제 잡기 우선

테스트:

- `test_capture_first_agent_prefers_actual_capture`

검증 내용:

- learner 말이 `GAE`로 이동하면 상대 말을 잡는 상태를 만든다.
- pool에는 `DO`, `GAE`를 넣는다.
- agent가 잡기가 발생하는 `GAE` action을 선택하는지 확인한다.

### 5.3 상대 stack 잡기 우선

테스트:

- `test_capture_first_agent_counts_stacked_opponents`

검증 내용:

- 상대 말 2개가 같은 칸에 업힌 상태를 만든다.
- agent가 상대 stack 2개를 잡는 action을 선택하는지 확인한다.

### 5.4 지나가는 상대 말은 잡기로 보지 않음

테스트:

- `test_capture_first_agent_ignores_passed_opponent_piece`

검증 내용:

- 이동 중 지나가는 칸에는 상대 말이 있지만 도착 칸에는 상대 말이 없는 상태를 만든다.
- 다른 내 말들은 `FINISHED`로 두어 테스트 대상 말만 움직일 수 있게 한다.
- agent가 해당 action을 선택하더라도 잡기 action으로 계산하지 않는지 확인한다.

초기 테스트 작성 중 한 번 실패가 있었다.

- 원인: 다른 대기 말이 같은 윷 결과로 상대 말이 있는 칸에 정확히 도착할 수 있었다.
- 조치: 의도한 검증을 위해 대상 외 내 말들을 `FINISHED`로 고정했다.

### 5.5 GreedyFinishAgent 골인 우선

테스트:

- `test_greedy_finish_agent_prefers_finishing_action`

검증 내용:

- 한 말은 골인 직전, 다른 말은 일반 위치에 둔다.
- pool에 `DO`, `GAE`를 넣는다.
- agent가 골인 가능한 action을 우선 선택하는지 확인한다.

### 5.6 GreedyFinishAgent stack 이동 우선

테스트:

- `test_greedy_finish_agent_prefers_moving_a_stack_when_no_finish_exists`

검증 내용:

- 골인 가능한 action이 없는 상태를 만든다.
- 2개 말이 업힌 stack과 단일 말이 모두 움직일 수 있게 한다.
- agent가 stack을 움직이는 action을 선택하는지 확인한다.

### 5.7 Baseline action mutation 안정성

테스트:

- `test_baseline_agent_selected_actions_remain_legal_after_mutation`

검증 내용:

- stack, 잡기 가능성, 여러 pool 결과가 섞인 상태를 만든다.
- `RandomAgent`, `CaptureFirstAgent`, `GreedyFinishAgent`가 선택한 action이 모두 legal action인지 확인한다.

### 5.8 Env opponent policy 호환성

테스트:

- `test_capture_first_agent_can_be_used_as_env_opponent_policy`

검증 내용:

- `CaptureFirstAgent.select_action(state, legal_actions)`가 M4 env의 opponent policy signature와 호환되는지 확인한다.
- 선택된 action을 `GameState.apply_action`에 넣었을 때 실제 잡기가 발생하는지 확인한다.

### 5.9 단일 게임 실행

테스트:

- `test_play_game_returns_winner_and_metrics`

검증 내용:

- `RandomAgent` 두 개로 한 게임을 실행한다.
- winner가 0 또는 1인지 확인한다.
- starting player가 0 또는 1인지 확인한다.
- turn count와 decision count가 양수인지 확인한다.

### 5.10 Random vs Random 1000판 smoke

테스트:

- `test_random_vs_random_1000_games_finish_without_illegal_actions`

검증 내용:

- `RandomAgent` 두 개로 1000판을 실행한다.
- 모든 게임이 종료되는지 확인한다.
- player별 승수 합이 1000인지 확인한다.
- 선공 count 합이 1000인지 확인한다.
- 평균 turn 수와 decision 수가 양수인지 확인한다.

이 테스트는 agent가 illegal action을 선택하면 `play_game`에서 즉시 실패한다.

### 5.11 Heuristic baseline smoke

테스트:

- `test_heuristic_baselines_beat_random_in_smoke_tournaments`

검증 내용:

- `CaptureFirstAgent vs RandomAgent` 100판 실행.
- `GreedyFinishAgent vs RandomAgent` 100판 실행.
- 각 tournament가 정상 종료되고 승수 합이 게임 수와 일치하는지 확인한다.

이 테스트는 승률 우위 자체를 고정하지 않는다. stochastic game 특성상 작은 판수에서 승률 우위를 단정하면 테스트가 불안정해질 수 있기 때문이다.

## 6. 실행 결과

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
45 passed in 1.03s
```

테스트 구성:

- baseline agent tests: 8개
- tournament tests: 3개
- board tests: 10개
- game state tests: 10개
- yut sampler tests: 3개
- env tests: 10개
- skeleton test: 1개

## 7. 해석

M5 구현으로 다음이 가능해졌다.

- rule engine만 사용한 agent 대전
- baseline agent를 env opponent policy로 사용
- 1000판 단위 smoke evaluation
- 승률, 선공 분포, 평균 turn/decision metric 산출

또한 baseline들이 legal action만 선택한다는 것을 테스트로 검증했다.

## 8. 예상과 달랐던 점

- 지나가는 말 검증 테스트에서 다른 대기 말이 실제 잡기 action을 만들 수 있었다.
- 테스트 의도에 맞게 대상 외 내 말들을 `FINISHED` 처리해 수정했다.
- 구현 계획 자체 변경은 없었다.

## 9. 남은 리스크

- baseline은 아직 학습 agent가 아니다.
- PPO/C51 학습 loop는 아직 없다.
- tournament metric은 기본 승률/turn/decision 중심이며, 잡기/업기/지름길 빈도 집계는 M8 분석 단계에서 확장해야 한다.
- episode 길이 제한은 여전히 게임 규칙으로는 도입하지 않았다. tournament runner의 `max_decisions`는 테스트 안전장치다.

## 10. 다음 단계

다음 단계는 M6 PPO 학습이다.

구현 대상:

- `MaskablePPO` 학습 스크립트
- baseline opponent 연결
- action mask evaluation loop
- 짧은 학습 smoke test
- 학습 전후 Random 상대 승률 저장
