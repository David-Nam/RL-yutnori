# Project-RF Rule-based Opponent PPO Plan

작성일: 2026-05-29

## 1. 목적

이 계획의 목적은 `project-RF-` repository의 Rule-based Agent 의사결정
로직을 현재 `RL-yutnori` 환경에 포팅하고, 이를 고정 opponent로 사용해
PPO 또는 PPO+hybrid 정책이 랜덤 선/후공 평가에서 승률 60% 이상을
달성하는 것이다.

최종 검증은 `RL-yutnori`의 `YutnoriEnv`에 포팅한
`project_rf_rule` opponent를 기준으로 한다. 공식 평가 기본 판수는
5000판이며, CLI 인자로 쉽게 변경할 수 있어야 한다.

## 2. 확정 사항

- C51 학습은 이번 계획 범위에서 제외한다.
- action encoding은 현재 `RL-yutnori` 방식을 유지한다.
  - `action = piece_id * 5 + yut_id`
- project-RF Rule-based Agent 재현은 action id ordering이 아니라
  점수 기반 의사결정 로직 재현을 의미한다.
- `project_rf_rule`은 학습 및 평가에서 선택 가능한 opponent 이름으로
  사용한다.
- 목표는 60% 이상이지만, 내부 안정 기준은 평균 63% 이상으로 본다.

## 3. Project-RF Rule-based 로직

project-RF의 Rule-based Agent는 legal action마다 다음 점수식을 적용하고,
가장 높은 점수의 action을 선택한다.

```text
finish destination: +100
capture destination: +50
waiting piece start: +5
stack move: +4 * (stack_size - 1)
distance penalty: -0.5 * distance_to_finish
```

현재 repo에서는 다음처럼 해석한다.

- capture는 상대 stack 크기와 무관하게 한 번만 `+50`을 부여한다.
- finish는 이동한 stack 크기와 무관하게 destination 기준 한 번만 `+100`을
  부여한다.
- `HOME`에 정확히 도착한 말은 아직 `FINISHED`가 아니므로 finish까지 남은
  거리를 `1`로 본다.
- 동점은 현재 repo action id 기준으로 처리한다.

## 4. 개발 단계

각 단계는 개발 후 검증까지 완료하면 커밋 가능한 크기로 나눈다. 단계가
끝나면 변경 내용, 검증 방법, 검증 결과를 보고하고 사용자 확인 후 다음
단계로 진행한다.

### Step 1. RF action score helper 구현

상태: 완료

구현:

- legal action 하나를 RF 점수로 평가하는 helper를 추가한다.
- finish, capture, waiting start, stack, distance penalty를 반영한다.

검증:

- finish/capture/start/stack/distance 각각의 unit test를 추가한다.
- 전체 regression test를 통과해야 한다.

### Step 2. ProjectRFRuleBasedAgent 구현

상태: 완료

구현:

- RF score helper로 legal action 중 최고 점수 action을 선택하는 agent를
  추가한다.
- 기존 baseline agent와 같은 `select_action(state, legal_actions)`
  interface를 사용한다.

검증:

- finish가 capture보다 우선되는지 확인한다.
- capture가 단순 전진보다 우선되는지 확인한다.
- stack 이동과 finish까지 더 가까운 action을 선호하는지 확인한다.
- 빈 legal action에서 `ValueError`가 발생하는지 확인한다.

### Step 3. opponent registry 연결

상태: 완료

구현:

- `project_rf_rule`을 opponent 이름 목록과 factory에 추가한다.
- `ProjectRFRuleBasedAgent`를 public export에 추가한다.

검증:

- `make_opponent("project_rf_rule")`가 올바른 agent를 반환해야 한다.
- `make_yutnori_env(opponent="project_rf_rule")`가 reset 후 정상 observation과
  action mask를 반환해야 한다.
- vector env action mask가 정상 동작해야 한다.

### Step 4. RF opponent game smoke 검증 추가

상태: 예정

구현:

- tournament helper 기반 테스트에 `project_rf_rule` smoke tournament를
  추가한다.
- 대상은 RF vs random, RF vs capture_first, RF vs greedy_finish로 한다.

검증:

- 각 matchup은 100~1000판 규모로 실행한다.
- 모든 게임이 종료되어야 한다.
- wins 합계가 games와 같아야 한다.
- agent가 illegal action을 선택하지 않아야 한다.

### Step 5. 공식 RF 평가 harness 구현

상태: 예정

구현:

- saved PPO model을 `project_rf_rule` 상대로 평가하는 스크립트를 추가한다.
- 기본값은 `--episodes 5000`으로 둔다.
- `--episodes`, `--seed`, `--device`, deterministic/stochastic 옵션을 제공한다.
- 출력 JSON에는 win rate, wins/losses, starting player counts,
  average turns, average decisions, illegal action count, pass 여부를 포함한다.

검증:

- `--episodes 10`, `--episodes 17` smoke 실행으로 JSON 생성과 판수 반영을
  확인한다.
- illegal action이 발생하면 평가가 실패해야 한다.

### Step 6. 현재 PPO checkpoint baseline 평가

상태: 예정

구현:

- 기존 PPO checkpoint를 공식 RF 평가 harness로 평가한다.
- 최소 1000판 smoke를 먼저 수행하고, 가능하면 5000판 평가를 수행한다.

검증:

- 평가 JSON에 승률, illegal action count, 선/후공 분포가 기록되어야 한다.
- `illegal_action_count`는 0이어야 한다.

### Step 7. tactical action feature helper 구현

상태: 예정

구현:

- 20개 action 각각에 대해 tactical feature를 계산하는 helper를 추가한다.
- feature 순서는 다음으로 고정한다.

```text
legal
capture
captured_count
finish
finished_count
moved_count
waiting_move
stack_size
distance_after
rf_score
```

- illegal action row는 모두 0으로 둔다.

검증:

- capture, finish, illegal action, stack, waiting 상황별 feature unit test를
  추가한다.

### Step 8. tactical observation mode 구현

상태: 예정

구현:

- 기존 base observation은 그대로 유지한다.
- `observation_mode="tactical"`이면 base observation 뒤에
  `20 actions x 10 features`를 붙인다.
- observation space shape는 mode별로 맞춘다.

검증:

- base mode shape가 기존과 동일해야 한다.
- tactical mode shape가 증가해야 한다.
- legal feature와 action mask가 일치해야 한다.

### Step 9. PPO train/eval에 observation mode 연결

상태: 예정

구현:

- train/evaluate/factory 경로에 `observation_mode`를 전달한다.
- train config에 observation mode를 저장한다.
- 평가 시 CLI 값이 있으면 우선하고, 없으면 model run config에서 자동
  추론한다.

검증:

- base/tactical 각각 짧은 PPO smoke 학습 및 평가를 수행한다.
- observation mode mismatch 없이 저장 모델 평가가 가능해야 한다.

### Step 10. RF shaped reward helper 구현

상태: 예정

구현:

- learner/opponent event를 받아 shaping reward를 계산하는 helper를 추가한다.
- 기본 shaping은 다음으로 둔다.

```text
learner capture: +0.08 * captured_count
learner finish: +0.15 * finished_count
learner shortcut: +0.02
opponent capture: -0.08 * captured_count
opponent finish: -0.15 * finished_count
```

검증:

- learner/opponent capture, finish, shortcut event별 unit test를 추가한다.

### Step 11. reward mode env 연결

상태: 예정

구현:

- env에 `reward_mode="terminal"|"rf_shaped"`를 추가한다.
- `terminal`은 기존 동작을 유지한다.
- `rf_shaped`는 terminal reward에 non-terminal shaping을 더한다.
- opponent events는 learner에게 불리한 shaping으로 반영한다.

검증:

- 기존 terminal reward 테스트가 그대로 통과해야 한다.
- shaped reward 전용 테스트를 추가한다.

### Step 12. PPO train/eval에 reward mode 연결

상태: 예정

구현:

- train/evaluate/factory 경로에 `reward_mode`를 전달한다.
- train config와 summary에 reward mode를 저장한다.

검증:

- `terminal`, `rf_shaped` 각각 짧은 PPO smoke 학습 및 평가를 수행한다.
- reward mode별 run artifact가 분리되고 평가 가능해야 한다.

### Step 13. 소규모 PPO sweep 실행

상태: 예정

실험 조합:

- base + terminal
- base + rf_shaped
- tactical + terminal
- tactical + rf_shaped

권장 설정:

- opponent: `project_rf_rule`
- seeds: `0, 1, 2`
- timesteps: `3M~5M`
- eval games: `1000`

검증:

- 각 run의 model, config, summary, eval JSON이 존재해야 한다.
- 모든 평가에서 `illegal_action_count == 0`이어야 한다.
- 장기 학습 후보 1~2개를 수치로 선택할 수 있어야 한다.

### Step 14. 장기 PPO 후보 학습 및 공식 검증

상태: 예정

구현:

- 소규모 sweep 상위 후보를 `10M~20M` timesteps로 학습한다.
- 공식 harness로 5000판 평가를 seed 3개 이상 실행한다.

검증:

- `illegal_action_count == 0`이어야 한다.
- starting player 분포가 정상이어야 한다.
- pure PPO 후보가 60% 이상인지 판단 가능해야 한다.

### Step 15. hybrid evaluation policy 구현

상태: 예정

구현:

- 학습된 PPO action을 기본으로 사용하되, 즉시 finish 또는 capture가 있으면
  override한다.
- 우선순위는 finish > capture > PPO로 둔다.
- finish/capture 후보 선택 기준은 finished_count, captured_count,
  moved_count, 낮은 distance_after, action id 순서로 고정한다.
- 학습 로직은 변경하지 않고 평가/제출 policy wrapper로 구현한다.

검증:

- finish override, capture override, no override 상황 unit test를 추가한다.
- 같은 checkpoint로 pure PPO와 hybrid PPO를 비교할 수 있어야 한다.

### Step 16. hybrid 공식 검증

상태: 예정

구현:

- 가장 좋은 PPO checkpoint에 hybrid wrapper를 적용해 공식 RF 평가를 실행한다.
- 기본 5000판, seed 3개 이상으로 평가한다.

검증:

- pure PPO 대비 승률 차이를 확인한다.
- `illegal_action_count == 0`이어야 한다.
- pass 여부를 확인해 최종 후보를 pure PPO 또는 hybrid 중 하나로 결정한다.

## 5. 공통 검증 기준

- unit test: RF score, RF agent 선택, tactical feature, shaped reward,
  hybrid override.
- integration test: env factory, action mask, observation mode, reward mode,
  PPO smoke train/eval.
- game smoke: RF opponent tournament 100~1000판.
- official verification: RF opponent 5000판, seed 3개 이상.
- regression: base observation + terminal reward에서는 기존 테스트가 그대로
  통과해야 한다.
