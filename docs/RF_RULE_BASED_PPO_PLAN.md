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
끝나면 변경 내용, 검증 방법, 검증 결과를 이 문서에 먼저 반영한 뒤
보고하고, 사용자 확인 후 다음 단계로 진행한다.

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

상태: 완료

구현:

- tournament helper 기반 테스트에 `project_rf_rule` smoke tournament를
  추가한다.
- 대상은 RF vs random, RF vs capture_first, RF vs greedy_finish로 한다.

검증:

- 각 matchup은 100~1000판 규모로 실행한다.
- 모든 게임이 종료되어야 한다.
- wins 합계가 games와 같아야 한다.
- agent가 illegal action을 선택하지 않아야 한다.

결과:

- `ProjectRFRuleBasedAgent`를 player 0/player 1 양쪽 위치에 두고
  random, capture_first, greedy_finish 상대 100판 smoke tournament를
  수행했다.
- 전체 regression test 기준 `74 passed`를 확인했다.

### Step 5. 공식 RF 평가 harness 구현

상태: 완료

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

결과:

- `scripts/evaluate_rf_target.py`를 추가했다.
- 기본 평가 판수는 `5000`이고, `--episodes`로 변경 가능하다.
- `pass_threshold=0.6`, `passed`, `target_opponent`,
  `official_episodes`를 평가 JSON에 기록한다.
- 실제 PPO checkpoint로 10판/17판 smoke 평가를 수행했고,
  `illegal_action_count == 0`을 확인했다.
- 전체 regression test 기준 `79 passed`를 확인했다.

### Step 6. 현재 PPO checkpoint baseline 평가

상태: 완료

구현:

- 기존 PPO checkpoint를 공식 RF 평가 harness로 평가한다.
- 최소 1000판 smoke를 먼저 수행하고, 가능하면 5000판 평가를 수행한다.

검증:

- 평가 JSON에 승률, illegal action count, 선/후공 분포가 기록되어야 한다.
- `illegal_action_count`는 0이어야 한다.

평가 명령:

```bash
.venv/bin/python scripts/evaluate_rf_target.py \
  --model-path <run-dir>/model.zip \
  --episodes 1000 \
  --seed <seed> \
  --device cpu \
  --output <run-dir>/eval_project_rf_rule_1000.json \
  --no-progress-bar
```

1000판 baseline 결과:

| checkpoint | win rate | illegal actions |
| --- | ---: | ---: |
| `random_seed_1_10m_nenv16` | 0.433 | 0 |
| `random_seed_2_10m_nenv16` | 0.422 | 0 |
| `greedy_finish_seed_1_10m_nenv16` | 0.396 | 0 |
| `random_seed_0_10m_nenv16` | 0.386 | 0 |
| `greedy_finish_seed_2_10m_nenv16` | 0.379 | 0 |
| `greedy_finish_seed_0_10m_nenv16` | 0.351 | 0 |
| `capture_first_seed_0_10m_nenv16` | 0.340 | 0 |
| `capture_first_seed_1_10m_nenv16` | 0.325 | 0 |
| `random_seed_0_1m_nenv16` | 0.299 | 0 |
| `capture_first_seed_2_10m_nenv16` | 0.284 | 0 |

최고 1000판 후보인 `random_seed_1_10m_nenv16`은 5000판 공식 조건으로
재평가했다.

```text
episodes: 5000
wins: 2133
losses: 2867
win_rate: 0.4266
illegal_action_count: 0
starting_player_counts: 0=2550, 1=2450
passed: false
```

해석:

- 기존 PPO checkpoint의 현재 최고 RF 상대 기준선은 42.66%다.
- 목표 60%까지 약 17.34%p 차이가 있어, 기존 checkpoint를 그대로 쓰는
  것으로는 목표 달성이 어렵다.
- 다음 단계부터는 tactical action feature와 reward shaping을 통해
  RF 점수식 opponent에 더 직접적으로 대응하도록 학습 입력과 보상을
  개선한다.

### Step 7. tactical action feature helper 구현

상태: 완료

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

결과:

- `yutnori/agents/tactical_features.py`를 추가했다.
- `tactical_action_features(state)`는 action id 순서의 `(20, 10)`
  `np.float32` feature matrix를 반환한다.
- `tactical_action_feature_row(state, action)`은 legal action 하나의 feature
  row를 반환하고, illegal action에는 `ValueError`를 발생시킨다.
- illegal action row는 모두 0으로 둔다.
- feature 순서는 계획의 10개 항목과 동일하게 고정했다.
- capture, finish, illegal action, stack, waiting move 상황별 unit test를
  추가했다.
- 검증 결과:
  - `tests/test_tactical_action_features.py`: `7 passed`
  - `tests/test_baseline_agents.py`: `20 passed`
  - 전체 regression: `86 passed`

### Step 8. tactical observation mode 구현

상태: 완료

구현:

- 기존 base observation은 그대로 유지한다.
- `observation_mode="tactical"`이면 base observation 뒤에
  `20 actions x 10 features`를 붙인다.
- observation space shape는 mode별로 맞춘다.

검증:

- base mode shape가 기존과 동일해야 한다.
- tactical mode shape가 증가해야 한다.
- legal feature와 action mask가 일치해야 한다.

결과:

- `YutnoriEnv`에 `observation_mode="base"|"tactical"` 옵션을 추가했다.
- 기본값은 `base`로 유지해 기존 observation shape와 동작을 보존한다.
- `tactical` mode는 base observation 뒤에
  `20 actions x 10 features`를 flatten해 붙인다.
- 다음 public 상수를 추가했다.

```text
OBSERVATION_MODE_BASE
OBSERVATION_MODE_TACTICAL
OBSERVATION_MODES
TACTICAL_OBSERVATION_SIZE
```

- `observation_size(observation_mode)` helper를 추가했다.
- tactical mode의 observation space low는 tactical feature의 음수 RF score를
  수용할 수 있도록 `-1_000_000.0`으로 설정했다.
- 검증 결과:
  - `tests/test_yutnori_env.py`: `14 passed`
  - tactical/factory 관련 테스트: `15 passed`
  - 전체 regression: `90 passed`

### Step 9. PPO train/eval에 observation mode 연결

상태: 완료

구현:

- train/evaluate/factory 경로에 `observation_mode`를 전달한다.
- train config에 observation mode를 저장한다.
- 평가 시 CLI 값이 있으면 우선하고, 없으면 model run config에서 자동
  추론한다.

검증:

- base/tactical 각각 짧은 PPO smoke 학습 및 평가를 수행한다.
- observation mode mismatch 없이 저장 모델 평가가 가능해야 한다.

결과:

- `make_yutnori_env()`와 `make_yutnori_vec_env()`에
  `observation_mode` 인자를 추가했다.
- 기본값은 `base`로 유지해 기존 학습/평가 경로의 observation shape를
  보존했다.
- `scripts/train_ppo.py`에 `--observation-mode base|tactical` 옵션을
  추가했다.
- train vec env, 학습 전/후 평가, early stopping 평가가 모두 같은
  observation mode를 사용하도록 연결했다.
- `config.json`, `summary.json`, early stopping eval log에
  `observation_mode`를 기록하도록 했다.
- `resolve_model_observation_mode(model_path, requested_observation_mode)`를
  추가했다.
  - CLI에서 명시한 mode가 있으면 우선한다.
  - 명시하지 않으면 `model.zip`과 같은 directory의 `config.json`에서
    `observation_mode`를 읽는다.
  - `config.json`이 없으면 backward compatibility를 위해 `base`를
    사용한다.
  - 알 수 없는 mode는 `ValueError`로 실패시킨다.
- `scripts/evaluate_ppo.py`와 `scripts/evaluate_rf_target.py`에
  `--observation-mode` 옵션을 추가했다.
- 평가 결과 JSON에 `observation_mode`를 기록하도록 했다.
- `scripts/run_ppo_long_sweep.py`도 observation mode를 train/eval command에
  전달하도록 했다.
- long sweep run name은 `base`에서는 기존 이름을 유지하고, `tactical`에서는
  `_tactical` suffix를 붙이도록 했다.

검증 결과:

- 관련 테스트:

```text
.venv/bin/python -m pytest \
  tests/test_training_env_factory.py \
  tests/test_model_config.py \
  tests/test_evaluate_rf_target.py \
  tests/test_ppo_long_sweep.py -q

27 passed
```

- tactical PPO smoke 학습:

```text
.venv/bin/python scripts/train_ppo.py \
  --total-timesteps 64 \
  --n-steps 8 \
  --batch-size 8 \
  --n-envs 1 \
  --opponent project_rf_rule \
  --observation-mode tactical \
  --run-dir /tmp/ppo_tactical_smoke \
  --eval-episodes 0 \
  --device cpu \
  --overwrite \
  --no-progress-bar
```

  - `trained_timesteps`: `64`
  - `observation_mode`: `tactical`
  - model 저장 위치: `/tmp/ppo_tactical_smoke/model.zip`

- 일반 evaluator smoke:

```text
.venv/bin/python scripts/evaluate_ppo.py \
  --model-path /tmp/ppo_tactical_smoke/model.zip \
  --episodes 2 \
  --opponent project_rf_rule \
  --device cpu \
  --output /tmp/ppo_tactical_eval.json \
  --no-progress-bar
```

  - `--observation-mode` 생략 상태에서 `config.json` 기반 tactical mode
    자동 추론 확인
  - `illegal_action_count`: `0`

- RF target evaluator smoke:

```text
.venv/bin/python scripts/evaluate_rf_target.py \
  --model-path /tmp/ppo_tactical_smoke/model.zip \
  --episodes 2 \
  --device cpu \
  --output /tmp/ppo_tactical_rf_eval.json \
  --no-progress-bar
```

  - `--observation-mode` 생략 상태에서 `config.json` 기반 tactical mode
    자동 추론 확인
  - `illegal_action_count`: `0`

- 전체 regression:

```text
.venv/bin/python -m pytest -q

101 passed
```

- diff whitespace check:

```text
git diff --check

no output
```

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
