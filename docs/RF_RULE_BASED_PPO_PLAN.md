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

상태: 완료

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

결과:

- `yutnori/training/reward_shaping.py`를 추가했다.
- 다음 shaping 상수를 추가했다.

```text
RF_SHAPING_CAPTURE_WEIGHT = 0.08
RF_SHAPING_FINISH_WEIGHT = 0.15
RF_SHAPING_SHORTCUT_BONUS = 0.02
```

- `project_rf_event_shaping_reward(event, learner_player=...)`를 추가했다.
  - learner event의 capture는 `+0.08 * captured_count`로 계산한다.
  - learner event의 finish는 `+0.15 * finished_count`로 계산한다.
  - learner event의 shortcut은 `+0.02`로 계산한다.
  - opponent event의 capture는 `-0.08 * captured_count`로 계산한다.
  - opponent event의 finish는 `-0.15 * finished_count`로 계산한다.
  - opponent shortcut은 계획대로 패널티를 주지 않는다.
- `project_rf_events_shaping_reward(learner_event, opponent_events, ...)`를
  추가했다.
  - Step 11에서 `YutnoriEnv.step()`의 learner event와 opponent events를
    합산할 때 사용할 수 있다.
- 잘못된 `learner_player` 또는 `event.actor` 값은 `ValueError`로 실패하도록
  했다.
- helper와 상수는 `yutnori.training`에서 public import 가능하도록 export했다.

검증 결과:

- 관련 테스트:

```text
.venv/bin/python -m pytest \
  tests/test_reward_shaping.py \
  tests/test_training_env_factory.py -q

18 passed
```

- 전체 regression:

```text
.venv/bin/python -m pytest -q

109 passed
```

### Step 11. reward mode env 연결

상태: 완료

구현:

- env에 `reward_mode="terminal"|"rf_shaped"`를 추가한다.
- `terminal`은 기존 동작을 유지한다.
- `rf_shaped`는 terminal reward에 non-terminal shaping을 더한다.
- opponent events는 learner에게 불리한 shaping으로 반영한다.

검증:

- 기존 terminal reward 테스트가 그대로 통과해야 한다.
- shaped reward 전용 테스트를 추가한다.

결과:

- `YutnoriEnv`에 `reward_mode="terminal"|"rf_shaped"` 옵션을 추가했다.
- 기본값은 `terminal`로 유지해 기존 env reward 동작을 보존했다.
- 다음 public 상수를 추가했다.

```text
REWARD_MODE_TERMINAL
REWARD_MODE_RF_SHAPED
REWARD_MODES
```

- `terminal` mode에서는 기존처럼 승리 시 `+1.0`, 패배 시 `-1.0`,
  비종료 상태에서는 `0.0`만 반환한다.
- `rf_shaped` mode에서는 terminal reward에 Step 10의
  `project_rf_events_shaping_reward()` 결과를 더한다.
- learner event와 opponent events를 모두 shaping 계산에 반영한다.
  - learner capture/finish/shortcut은 양수 shaping이다.
  - opponent capture/finish는 음수 shaping이다.
  - opponent shortcut은 패널티가 없다.
- step info에 다음 reward breakdown을 기록한다.

```text
reward_mode
terminal_reward
shaping_reward
```

- 잘못된 reward mode는 env 생성 시 `ValueError`로 실패한다.

검증 결과:

- reward mode 관련 테스트:

```text
.venv/bin/python -m pytest \
  tests/test_yutnori_env.py \
  tests/test_reward_shaping.py -q

28 passed
```

- 전체 regression:

```text
.venv/bin/python -m pytest -q

115 passed
```

### Step 12. PPO train/eval에 reward mode 연결

상태: 완료

구현:

- train/evaluate/factory 경로에 `reward_mode`를 전달한다.
- train config와 summary에 reward mode를 저장한다.

검증:

- `terminal`, `rf_shaped` 각각 짧은 PPO smoke 학습 및 평가를 수행한다.
- reward mode별 run artifact가 분리되고 평가 가능해야 한다.

결과:

- `make_yutnori_env()`와 `make_yutnori_vec_env()`에 `reward_mode` 인자를
  추가했다.
- `evaluate_maskable_policy()`에 `reward_mode` 인자를 추가해 평가 env도
  같은 reward mode로 생성할 수 있게 했다.
- `scripts/train_ppo.py`에 `--reward-mode terminal|rf_shaped` 옵션을
  추가했다.
- train vec env, 학습 전/후 평가, early stopping 평가가 모두 같은
  reward mode를 사용하도록 연결했다.
- `config.json`, `summary.json`, early stopping eval log에 `reward_mode`를
  기록하도록 했다.
- `resolve_model_reward_mode(model_path, requested_reward_mode)`를 추가했다.
  - CLI에서 명시한 mode가 있으면 우선한다.
  - 명시하지 않으면 `model.zip`과 같은 directory의 `config.json`에서
    `reward_mode`를 읽는다.
  - `config.json`이 없거나 오래된 config에 `reward_mode`가 없으면
    backward compatibility를 위해 `terminal`을 사용한다.
  - 알 수 없는 mode는 `ValueError`로 실패시킨다.
- `scripts/evaluate_ppo.py`와 `scripts/evaluate_rf_target.py`에
  `--reward-mode` 옵션을 추가했다.
- 평가 결과 JSON에 `reward_mode`를 기록하도록 했다.
- `scripts/run_ppo_long_sweep.py`도 reward mode를 train/eval command에
  전달하도록 했다.
- long sweep run name은 `base + terminal`에서는 기존 이름을 유지한다.
  - `base + rf_shaped`: `_rf_shaped` suffix
  - `tactical + terminal`: `_tactical` suffix
  - `tactical + rf_shaped`: `_tactical_rf_shaped` suffix

검증 결과:

- 관련 테스트:

```text
.venv/bin/python -m pytest \
  tests/test_model_config.py \
  tests/test_training_env_factory.py \
  tests/test_evaluate_rf_target.py \
  tests/test_ppo_long_sweep.py -q

39 passed
```

- `rf_shaped` PPO smoke 학습:

```text
.venv/bin/python scripts/train_ppo.py \
  --total-timesteps 64 \
  --n-steps 8 \
  --batch-size 8 \
  --n-envs 1 \
  --opponent project_rf_rule \
  --reward-mode rf_shaped \
  --run-dir /tmp/ppo_rf_shaped_smoke \
  --eval-episodes 0 \
  --device cpu \
  --overwrite \
  --no-progress-bar
```

  - `trained_timesteps`: `64`
  - `observation_mode`: `base`
  - `reward_mode`: `rf_shaped`
  - model 저장 위치: `/tmp/ppo_rf_shaped_smoke/model.zip`

- 일반 evaluator smoke:

```text
.venv/bin/python scripts/evaluate_ppo.py \
  --model-path /tmp/ppo_rf_shaped_smoke/model.zip \
  --episodes 2 \
  --opponent project_rf_rule \
  --device cpu \
  --output /tmp/ppo_rf_shaped_eval.json \
  --no-progress-bar
```

  - `--reward-mode` 생략 상태에서 `config.json` 기반 `rf_shaped` mode
    자동 추론 확인
  - `illegal_action_count`: `0`

- RF target evaluator smoke:

```text
.venv/bin/python scripts/evaluate_rf_target.py \
  --model-path /tmp/ppo_rf_shaped_smoke/model.zip \
  --episodes 2 \
  --device cpu \
  --output /tmp/ppo_rf_shaped_rf_eval.json \
  --no-progress-bar
```

  - `--reward-mode` 생략 상태에서 `config.json` 기반 `rf_shaped` mode
    자동 추론 확인
  - `illegal_action_count`: `0`

- 전체 regression:

```text
.venv/bin/python -m pytest -q

127 passed
```

### Step 13. 소규모 PPO sweep 실행

상태: 완료

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

실행 준비:

- `scripts/run_step13_gpu_sweep.sh`를 추가했다.
- 스크립트는 아래 네 조합을 순차 실행한다.

```text
base + terminal
base + rf_shaped
tactical + terminal
tactical + rf_shaped
```

- 각 조합은 `run_ppo_long_sweep.py`를 호출하며, progress bar가 보이도록
  `--no-progress-bar`를 전달하지 않는다.
- 기본 설정은 다음과 같다.

```text
opponent: project_rf_rule
seeds: 0 1 2
total_timesteps: 3000000
timesteps_label: 3m
n_envs: 16
vec_env: dummy
device: cuda
train_eval_episodes: 100
final_eval_episodes: 1000
checkpoint_freq: 0
runs_root: runs/ppo_step13_gpu_sweep_full
logs_root: logs/ppo_step13_gpu_sweep_full
```

- 사용자 터미널에서 GPU가 보이는 환경이면 다음처럼 실행한다.

```bash
cd /home/david-nam/work-space/RL-yutnori
scripts/run_step13_gpu_sweep.sh
```

- 특정 GPU를 지정하려면 다음처럼 실행한다.

```bash
CUDA_VISIBLE_DEVICES=0 scripts/run_step13_gpu_sweep.sh
```

- 설정을 바꾸려면 환경변수를 사용한다.

```bash
TOTAL_TIMESTEPS=5000000 \
TIMESTEPS_LABEL=5m \
SEEDS="0 1 2" \
scripts/run_step13_gpu_sweep.sh
```

- 기존 run directory와 충돌하는 경우에는 새 `RUNS_ROOT`/`LOGS_ROOT`를
  지정하거나, 의도적으로 다시 돌릴 때만 `--overwrite`를 추가한다.

검증 결과:

- 스크립트 문법 검사:

```text
bash -n scripts/run_step13_gpu_sweep.sh

passed
```

- `/tmp` 출력 경로로 dry-run을 실행해 네 조합이 순차 생성되는지 확인했다.

```text
RUNS_ROOT=/tmp/ppo_step13_gpu_sweep_dry_runs \
LOGS_ROOT=/tmp/ppo_step13_gpu_sweep_dry_logs \
scripts/run_step13_gpu_sweep.sh --dry-run

passed
```

실행 결과:

- 실행 경로: `runs/ppo_step13_gpu_sweep_full`
- log 경로: `logs/ppo_step13_gpu_sweep_full`
- 실행 환경:
  - device: `cuda`
  - GPU: `NVIDIA A100-SXM4-40GB`
  - seeds: `0, 1, 2`
  - timesteps: `3,000,000`
  - n_envs: `16`
  - final eval: `project_rf_rule` 상대 `1000`판
- artifact 검증:
  - 12개 run 모두 `model.zip`, `config.json`, `summary.json`,
    `eval_project_rf_rule_1000.json` 생성 확인
  - 모든 RF target 평가에서 `illegal_action_count == 0`
  - starting player 분포는 각 평가에서 `508/492` 또는 `509/491`로 정상

RF target 1000판 평가 결과:

| observation | reward | seed 0 | seed 1 | seed 2 | mean | min | max |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base | terminal | 0.397 | 0.372 | 0.353 | 0.374 | 0.353 | 0.397 |
| base | rf_shaped | 0.283 | 0.323 | 0.365 | 0.324 | 0.283 | 0.365 |
| tactical | terminal | 0.525 | 0.545 | 0.528 | 0.533 | 0.525 | 0.545 |
| tactical | rf_shaped | 0.512 | 0.519 | 0.539 | 0.523 | 0.512 | 0.539 |

보조 상대 1000판 평균 승률:

| observation | reward | random | capture_first | greedy_finish |
| --- | --- | ---: | ---: | ---: |
| base | terminal | 0.650 | 0.286 | 0.738 |
| base | rf_shaped | 0.582 | 0.347 | 0.588 |
| tactical | terminal | 0.797 | 0.471 | 0.751 |
| tactical | rf_shaped | 0.757 | 0.510 | 0.616 |

해석:

- `tactical` observation은 RF target 승률을 크게 끌어올렸다.
  - `base + terminal`: 평균 `0.374`
  - `tactical + terminal`: 평균 `0.533`
  - `base + rf_shaped`: 평균 `0.324`
  - `tactical + rf_shaped`: 평균 `0.523`
- `rf_shaped` reward는 이번 3M sweep에서 RF target 평균 승률을 올리지 못했다.
  - base에서는 `0.374 -> 0.324`로 하락했다.
  - tactical에서는 `0.533 -> 0.523`으로 소폭 하락했다.
- 다만 `tactical + rf_shaped`는 `capture_first` 상대 평균 승률이 가장 높다.
  - `tactical + terminal`: `0.471`
  - `tactical + rf_shaped`: `0.510`
  - capture 성향 전술은 개선됐지만, RF target과 greedy_finish 상대에서는
    terminal reward 쪽이 더 안정적이었다.
- Step 14의 1순위 장기 학습 후보는 `tactical + terminal`로 둔다.
- Step 14의 2순위 비교 후보는 `tactical + rf_shaped`로 둔다.
- `base` observation 조합은 RF target 기준으로 60% 목표와 거리가 있어
  장기 학습 우선순위에서 제외한다.

결론:

- 현재까지의 가장 강한 결론은 성능 향상의 핵심이 reward shaping보다
  `tactical` observation에 있다는 점이다.
- 3M sweep 기준 1순위 후보는 `tactical + terminal`이다.
  - RF target 평균 승률: `0.533`
  - seed별 범위: `0.525~0.545`
  - seed 편차가 작아 장기 학습 후보로 안정적이다.
- `rf_shaped` reward는 capture 관련 전술을 강화하는 효과는 보였지만,
  RF target 전체 승률을 올리지는 못했다.
  - `tactical + terminal`의 `capture_first` 평균 승률: `0.471`
  - `tactical + rf_shaped`의 `capture_first` 평균 승률: `0.510`
  - 반면 RF target 평균은 `0.533 -> 0.523`으로 내려갔다.
- 현재 `rf_shaped` reward는 실제 capture/captured event에 대한 결과 기반
  보상/패널티는 포함하지만, 상대가 다음 턴에 잡기 쉬운 위치를 피하는
  위험도 기반 penalty는 포함하지 않는다.
- 따라서 다음 실험에서 reward를 더 조정하기보다 먼저
  `tactical + terminal`의 장기 학습 상한을 확인하는 것이 합리적이다.
- 장기 학습 후에도 승률이 목표에 부족하면, reward weight를 단순 조정하기보다
  hybrid policy 또는 capture-risk feature/reward를 별도 개선 후보로 검토한다.

### Step 14. 장기 PPO 후보 학습 및 공식 검증

상태: 30M 확장 실행 준비 완료

구현:

- 소규모 sweep 1순위 후보인 `tactical + terminal`을 `10M~30M`
  timesteps로 학습한다.
- 비교가 필요하면 2순위 후보인 `tactical + rf_shaped`도 같은 조건으로
  학습한다.
- 공식 harness로 5000판 평가를 seed 3개 이상 실행한다.

권장 진행:

- 1차 장기 학습:
  - observation: `tactical`
  - reward: `terminal`
  - opponent: `project_rf_rule`
  - seeds: `0, 1, 2`
  - timesteps: `10M`
  - eval games: `5000`
- 1차 결과가 `58~60%` 근처까지 상승하면 같은 구성을 `30M`까지 확장한다.
- 1차 결과가 `53~55%`대에 머물면 pure PPO만으로는 목표 달성이 불확실하므로
  Step 15 hybrid policy 구현 우선순위를 높인다.
- `tactical + terminal`의 장기 학습이 불안정하거나 특정 seed에서 크게 무너지면
  `tactical + rf_shaped`를 같은 조건으로 비교한다.
- `tactical + rf_shaped`가 장기에서도 RF target을 역전하지 못하면 현재
  `rf_shaped` reward는 우선 후보에서 제외하고, 필요 시 capture-risk 기반
  reward/feature로 재설계한다.

실행 스크립트:

- `scripts/run_step14_long_training.sh`를 추가한다.
- 기본 실행은 1순위 후보인 `tactical + terminal`만 학습/평가한다.
- 학습은 `run_ppo_long_sweep.py`를 사용하되 final eval은 생략하고,
  학습 완료 후 `evaluate_rf_target.py`로 공식 RF target 5000판 평가를
  각 seed별로 실행한다.
- 공식 평가 결과는 각 run directory에 아래 이름으로 저장된다.

```text
eval_project_rf_rule_official_5000.json
```

- 기본 설정은 다음과 같다.

```text
profile: primary
opponent: project_rf_rule
observation: tactical
reward: terminal
seeds: 0 1 2
total_timesteps: 10000000
timesteps_label: 10m
n_envs: 16
device: cuda
train_eval_episodes: 100
official_eval_episodes: 5000
pass_threshold: 0.60
checkpoint_freq: 1000000
runs_root: runs/ppo_step14_long_training_v2
logs_root: logs/ppo_step14_long_training_v2
```

- 사용자는 먼저 dry-run으로 실행될 명령을 확인한다.

```bash
cd /home/david-nam/work-space/RL-yutnori
scripts/run_step14_long_training.sh --dry-run
```

- 문제가 없으면 실제 1차 장기 학습을 실행한다.

```bash
scripts/run_step14_long_training.sh
```

- 특정 GPU를 지정하려면 다음처럼 실행한다.

```bash
CUDA_VISIBLE_DEVICES=0 scripts/run_step14_long_training.sh
```

- 2순위 후보인 `tactical + rf_shaped` 비교가 필요하면 다음처럼 실행한다.

```bash
PROFILE=comparison scripts/run_step14_long_training.sh
```

- 두 후보를 연속으로 모두 실행하려면 다음처럼 실행한다.

```bash
PROFILE=both scripts/run_step14_long_training.sh
```

- 확장 학습은 현재 resume 방식이 아니라 fresh run으로 실행한다.

```bash
scripts/run_step14_30m_training.sh
```

- 기존 run directory가 있으면 기본적으로 재사용/skip한다.
  의도적으로 다시 실행할 때만 `--overwrite`를 추가한다.
- 공식 평가 JSON만 다시 만들고 싶으면 `OVERWRITE_EVAL=1`을 지정한다.

실행 준비 검증:

- 스크립트 문법 검사:

```text
bash -n scripts/run_step14_long_training.sh

passed
```

- `/tmp` 출력 경로로 기본 primary dry-run을 실행해 아래 명령 생성 확인:
  - `tactical + terminal` 10M 학습 3개 seed
  - `evaluate_rf_target.py` 공식 RF 5000판 평가 3개 seed

```text
RUNS_ROOT=/tmp/ppo_step14_long_training_dry_runs \
LOGS_ROOT=/tmp/ppo_step14_long_training_dry_logs \
scripts/run_step14_long_training.sh --dry-run

passed
```

- comparison dry-run으로 `tactical + rf_shaped` 경로와 run name도 확인했다.

실행 중 발견한 이슈와 수정:

- 최초 Step 14 실행은 아래 오류로 seed 0 학습 중단:

```text
RuntimeError: step called while learner is not the current player
```

- 원인:
  - single-learner env는 `reset()`에서 opponent 선공이면 opponent 턴을
    내부에서 자동 진행한 뒤 learner decision state를 반환한다.
  - 아주 긴 보너스 윷/모 연속이 reset 중 발생하면 opponent가 learner의
    첫 decision 전에 게임을 끝낼 수 있다.
  - 기존 구현은 이 terminal-before-learner-decision 상태를 그대로 반환할 수
    있었고, 다음 PPO step에서 learner 턴이 아니라는 RuntimeError가 발생했다.
- 수정:
  - `YutnoriEnv.reset()`이 opponent 자동 진행 중 terminal이 된 opening을
    버리고 새 게임을 재샘플링하도록 수정한다.
  - 성공한 reset info에 `skipped_terminal_resets`를 기록한다.
  - deterministic regression test를 추가해 opponent opening terminal을
    재현하고, reset이 learner decision state를 반환하는지 확인한다.
- 실패한 최초 실행 artifact가 `runs/ppo_step14_long_training`에 남아 있으므로,
  기본 Step 14 출력 경로는 충돌을 피하기 위해 `*_v2`로 변경한다.
  사용자는 수정 후 다시 아래 명령만 실행하면 된다.

```bash
scripts/run_step14_long_training.sh
```

1차 장기 학습 결과:

- 실행 경로: `runs/ppo_step14_long_training_v2`
- log 경로: `logs/ppo_step14_long_training_v2`
- 학습 설정:
  - observation: `tactical`
  - reward: `terminal`
  - opponent: `project_rf_rule`
  - seeds: `0, 1, 2`
  - timesteps: `10M`
  - n_envs: `16`
  - device: `cuda`
  - GPU: `NVIDIA A100-SXM4-40GB`
  - official eval: `project_rf_rule` 상대 `5000`판
- artifact 검증:
  - 3개 run 모두 `model.zip`, `summary.json`, `config.json` 생성 확인
  - 각 run마다 checkpoint 10개 생성 확인
  - 각 run마다 `eval_project_rf_rule_official_5000.json` 생성 확인

공식 RF target 5000판 평가 결과:

| seed | wins | losses | win_rate | passed | illegal | starting player |
| ---: | ---: | ---: | ---: | :---: | ---: | --- |
| 0 | 2846 | 2154 | 0.5692 | false | 0 | 2509 / 2491 |
| 1 | 2919 | 2081 | 0.5838 | false | 0 | 2509 / 2491 |
| 2 | 2928 | 2072 | 0.5856 | false | 0 | 2510 / 2490 |

집계:

```text
mean: 0.5795
min: 0.5692
max: 0.5856
population stdev: 0.0073
total official games: 15000
illegal_action_count sum: 0
all passed: false
```

해석:

- 3M sweep의 `tactical + terminal` 평균 `0.533`에서 10M 공식 평가 평균
  `0.5795`까지 상승했다.
- seed 1, 2는 `58%` 중반까지 올라왔고, seed 간 편차도 작다.
- 하지만 모든 seed가 공식 pass threshold `0.60`에는 도달하지 못했다.
- 현재 결과는 Step 14의 사전 판단 기준 중 `58~60% 근처`에 해당하므로,
  pure PPO를 바로 포기하기보다 `30M` 확장을 먼저 시도할 근거가 있다.
- 현재 학습 스크립트는 resume을 지원하지 않으므로 30M은 fresh run이다.
  10M checkpoint에서 이어 학습하려면 별도 resume 기능 구현이 필요하다.

결론:

- 현재 pure PPO 최고 후보는 계속 `tactical + terminal`이다.
- 10M 기준 목표 60%는 미달이다.
- 다음 개발/실험 우선순위는 `tactical + terminal` 30M 확장이다.
- 30M에서도 60%를 넘지 못하면 Step 15 hybrid policy 구현으로 넘어간다.

30M 실행과 CPU 병렬화:

- 기존 학습은 `DummyVecEnv`를 사용했다. `n_envs=16`이어도 한 Python
  process에서 env를 순차 실행하므로 CPU 12 core를 병렬로 활용하지 못했다.
- `make_yutnori_vec_env()`에 아래 vector env mode를 추가한다.

```text
dummy: 기존 단일 process 순차 실행
subproc: env별 subprocess 병렬 실행
```

- `scripts/train_ppo.py`와 `scripts/run_ppo_long_sweep.py`에
  `--vec-env dummy|subproc` 옵션을 추가한다.
- 기존 학습의 기본값은 `dummy`로 유지해 이전 실험 재현성을 보존한다.
- 30M 전용 `scripts/run_step14_30m_training.sh`는 다음 설정을 사용한다.

```text
observation: tactical
reward: terminal
opponent: project_rf_rule
seeds: 0 1 2
total_timesteps: 30000000
timesteps_label: 30m
n_envs: 12
vec_env: subproc
device: cuda
checkpoint_freq: 3000000
official_eval_episodes: 5000
runs_root: runs/ppo_step14_30m_subproc
logs_root: logs/ppo_step14_30m_subproc
```

- 각 env worker가 별도 process로 실행되며, BLAS/OpenMP thread는 process당
  1개로 제한해 12개 worker가 불필요하게 thread를 중첩 생성하지 않게 한다.
- 10M은 `n_envs=16 + dummy`, 30M은 `n_envs=12 + subproc`이므로
  학습량만 바꾼 완전한 단일 변수 비교는 아니다. 이번 30M 설정은
  12 core 활용과 최종 성능 탐색을 우선한다.
- 30M rollout size는 `12 * 2048 = 24576`이고, batch size `2048`로 정확히
  나누어져 PPO minibatch 구성에는 나머지가 생기지 않는다.
- 사용자는 먼저 dry-run을 확인한다.

```bash
scripts/run_step14_30m_training.sh --dry-run
```

- 실제 실행은 아래 명령 하나로 진행한다.

```bash
scripts/run_step14_30m_training.sh
```

- 실행 중 CPU 활용은 별도 terminal에서 아래처럼 확인한다.

```bash
htop
```

- 30M은 10M model에서 이어가는 resume 학습이 아니라 seed별 fresh run이다.

30M 실행 준비 검증:

- `SubprocVecEnv`가 tactical observation과 action mask를 정상 전달하는
  unit test 통과
- `run_ppo_long_sweep.py`가 `--vec-env subproc`를 train command에 전달하는
  unit test 통과
- subprocess env를 사용한 MaskablePPO CPU smoke 학습 통과
- 30M script dry-run에서 seed 3개 학습과 공식 5000판 평가 command 생성 확인
- 전체 regression: `131 passed`

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
