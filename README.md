# RL Yutnori

강화학습으로 윷놀이 agent를 학습하고, 고정된 Rule-based Agent와의
1:1 대전에서 승률 60% 이상을 목표로 하는 프로젝트입니다.

현재 주력 모델은 action masking을 적용한 `MaskablePPO`이며, 윷놀이 규칙
엔진, Gymnasium 환경, baseline agent, 학습 및 평가 도구를 함께 제공합니다.

## Project Goal

최종 목표와 평가 조건은 다음과 같습니다.

```text
opponent: project_rf_rule
대전 방식: 1:1
선/후공: 무작위
평가 판수: 5000판
목표 승률: 60% 이상
```

`project_rf_rule`은 별도 `project-RF-` 프로젝트의 Rule-based Agent 동작을
이 환경에 포팅한 고정 평가 상대입니다. 완주, 상대 말 잡기, 새 말 출발,
업기, 도착까지 남은 거리 등을 기준으로 action을 선택합니다.

## Overview

프로젝트는 다음 구성으로 나뉩니다.

```text
yutnori/core       윷 결과, 보드 이동, 턴과 잡기 등 게임 규칙
yutnori/env        Gymnasium 환경, observation, reward, action mask
yutnori/agents     baseline 및 project-RF Rule-based Agent
yutnori/training   PPO 환경 factory, 평가, reward shaping
yutnori/eval       agent 간 tournament 도구
scripts            PPO 학습, sweep, 공식 평가 실행 스크립트
tests              규칙, 환경, agent, 학습 경로 회귀 테스트
docs               설계, 단계별 계획, 실험 결과 보고서
```

PPO는 legal action만 선택하도록 action mask를 사용합니다. 평가 결과에서도
illegal action이 발생하지 않았는지 별도로 기록합니다.

## PPO Design

### Observation

두 가지 observation 구성을 비교했습니다.

- `base`: 양측 말의 위치와 상태, stack, 현재 사용할 수 있는 윷 결과
- `tactical`: base 정보와 함께 각 action의 전술적 결과를 추가

`tactical` mode에는 legal 여부, 잡기, 완주, 이동할 말 수, 대기 중인 말의
출발 여부, stack 크기, 도착까지 남은 거리, Rule-based score 등의 action
feature가 포함됩니다.

### Reward

두 가지 reward 구성을 실험했습니다.

- `terminal`: 승리 `+1`, 패배 `-1`, 나머지 step `0`
- `rf_shaped`: terminal reward에 잡기, 완주, 지름길 보상과 상대의 잡기,
  완주에 대한 penalty 추가

사전 실험에서는 `rf_shaped`가 잡기 성향을 강화했지만, RF Agent 상대 전체
승률은 `tactical + terminal` 조합이 가장 높았습니다.

## Experiment Results

### 3M Candidate Sweep

네 가지 PPO 구성을 seed 3개로 학습하고, 각 모델을 RF Agent 상대
1000판으로 평가했습니다.

| Observation | Reward | Mean Win Rate |
| --- | --- | ---: |
| base | terminal | 37.4% |
| base | rf_shaped | 32.4% |
| tactical | terminal | **53.3%** |
| tactical | rf_shaped | 52.3% |

이 결과를 바탕으로 `tactical + terminal`을 장기 학습의 1순위 후보로
선정했습니다. 성능 향상에는 reward shaping보다 action의 전술적 의미를
직접 제공한 tactical observation이 더 큰 영향을 주었습니다.

### 10M Long Training

선정된 `tactical + terminal` PPO를 A100 GPU에서 seed별 10M timesteps로
학습한 뒤, 각 모델을 공식 조건인 RF Agent 상대 5000판으로 평가했습니다.

| Seed | Wins | Losses | Win Rate | Illegal Actions |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 2846 | 2154 | 56.92% | 0 |
| 1 | 2919 | 2081 | 58.38% | 0 |
| 2 | 2928 | 2072 | 58.56% | 0 |

```text
평균 승률: 57.95%
평가 게임: 총 15,000판
seed 표준편차: 0.73%p
illegal actions: 0
```

목표 60%에는 아직 도달하지 못했지만, 3M의 평균 53.3%에서 10M의
57.95%로 상승했고 seed 간 편차도 작았습니다. 따라서 현재 pure PPO
최고 후보는 계속 `tactical + terminal`입니다.

### 30M Parallel Training

같은 PPO 후보를 seed별 30M timesteps로 fresh training했습니다. 12개 CPU
core를 활용하도록 `SubprocVecEnv`와 `n_envs=12`를 사용했고, 학습은 A100
GPU에서 수행했습니다. 아래 결과는 공통 평가 가이드 도입 전의 기존
`project_rf_rule` 무작위 선공 평가 결과입니다.

| Seed | Wins | Losses | Win Rate | Passed | Illegal Actions |
| ---: | ---: | ---: | ---: | :---: | ---: |
| 0 | 3000 | 2000 | **60.00%** | true | 0 |
| 1 | 2944 | 2056 | 58.88% | false | 0 |
| 2 | 2983 | 2017 | 59.66% | false | 0 |

```text
평균 승률: 59.51%
평가 게임: 총 15,000판
seed 표준편차: 0.47%p
10M 대비 개선: +1.56%p
60% 통과: 3개 seed 중 1개
illegal actions: 0
```

30M 확장은 평균 성능과 seed 안정성을 개선했고, seed 0은 공식 기준을
정확히 통과했습니다. 따라서 최소 목표를 만족하는 pure PPO 후보는
확보했습니다. 다만 전체 seed 평균은 60%에 0.49%p 부족하고 seed 1, 2는
기준 미달이므로, 안정적인 최종 후보가 확정됐다고 보지는 않습니다.

### Common Paired Evaluation on 30M Models

이후 팀 공통 평가 가이드를 적용했습니다. 공통 평가는 2,500개 base seed마다
모델 선공과 후공을 한 번씩 실행해 정확히 2,500판씩, 총 5,000판을
평가합니다. Rule-based Agent의 점수가 같으면 가장 작은 action ID를
선택합니다.

| Seed | 전체 승률 | 선공 승률 | 후공 승률 | 95% CI |
| ---: | ---: | ---: | ---: | --- |
| 0 | **57.34%** | 58.80% | 55.88% | 55.96~58.70% |
| 1 | 55.94% | 57.60% | 54.28% | 54.56~57.31% |
| 2 | 56.20% | 57.56% | 54.84% | 54.82~57.57% |

```text
3-seed 평균/pooled 승률: 56.49%
선공 pooled 승률: 57.99%
후공 pooled 승률: 55.00%
seed 표준편차: 0.61%p
illegal actions: 0
evaluation errors: 0
```

같은 paired seed에서 기존 큰 action ID 동점 규칙을 사용하면 평균은
58.78%였습니다. 평가 pairing 변화로 약 0.73%p, 공통 opponent의 동점 정책
변화로 추가 약 2.29%p가 내려갔습니다. 현재 PPO는 기존 opponent의 말 ID
선택 패턴에도 적응했으므로 공통 opponent를 상대로 다시 학습해야 합니다.

### Common Rule 40M Retraining

공통 opponent를 직접 학습 상대로 사용해 같은 `tactical + terminal` PPO를
seed별 40M timesteps로 다시 학습했습니다.

| Seed | 전체 승률 | 선공 승률 | 후공 승률 | Passed | 95% CI |
| ---: | ---: | ---: | ---: | :---: | --- |
| 0 | 58.42% | 59.00% | 57.84% | false | 57.05~59.78% |
| 1 | **60.46%** | 60.76% | 60.16% | true | 59.10~61.81% |
| 2 | **60.40%** | 61.32% | 59.48% | true | 59.04~61.75% |

```text
3-seed 평균/pooled 승률: 59.76%
30M common 대비 개선: +3.27%p
선공 pooled 승률: 60.36%
후공 pooled 승률: 59.16%
60% 통과: 3개 seed 중 2개
illegal actions: 0
evaluation errors: 0
```

40M 재학습으로 공통 평가 기준을 통과하는 pure PPO 후보를 확보했습니다.
3-seed 평균은 60%에 0.24%p 부족하지만, seed 1과 seed 2는 선공/후공
분할에서도 균형 있게 개선됐습니다.

자세한 분석은 [팀 회의 보고서](docs/RF_PPO_TEAM_MEETING_REPORT.md)와
[개발 계획 및 진행 기록](docs/RF_RULE_BASED_PPO_PLAN.md)에서 확인할 수
있습니다.

## Current Status

공통 opponent 40M 재학습까지 완료했습니다. 현재 최고 pure PPO 후보는
seed 1의 `60.46%`이고, seed 2도 `60.40%`로 통과했습니다.

```text
현재 판단: 공통 기준 통과 pure PPO 후보 확보
다음 검증: 32M, 36M, 40M checkpoint 공통 평가
후속 판단: margin 확대 실패 시 hybrid 또는 observation 개선
```

seed 0은 `58.42%`로 미달했고, 3-seed 평균도 아직 60% 바로 아래입니다.
따라서 최종 제출 후보는 seed 1 또는 seed 2로 둘 수 있지만, 더 안전한
margin을 확보하기 위해 저장된 후반 checkpoint를 먼저 비교합니다.

## Setup

Python 3.11과 CUDA 12.1 환경을 기준으로 개발했습니다.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

GPU 없이도 테스트와 짧은 smoke 학습은 가능하지만, 장기 PPO 학습에는
CUDA GPU 사용을 권장합니다.

## Tests

전체 테스트 실행:

```bash
.venv/bin/python -m pytest -q
```

공통 paired evaluator, 규칙 엔진, tactical feature, reward, action mask,
subprocess vector environment를 포함한 전체 테스트가 통과합니다.

## Training

### Short PPO Example

```bash
.venv/bin/python scripts/train_ppo.py \
  --total-timesteps 100000 \
  --seed 0 \
  --opponent project_rf_rule \
  --observation-mode tactical \
  --reward-mode terminal \
  --n-envs 2 \
  --vec-env dummy \
  --device cpu \
  --run-dir runs/ppo_quickstart
```

### Reproduce the 30M Experiment

실행될 명령을 먼저 확인합니다.

```bash
scripts/run_step14_30m_training.sh --dry-run
```

실제 학습과 공식 평가를 순차 실행합니다.

```bash
scripts/run_step14_30m_training.sh
```

스크립트는 seed 0, 1, 2의 30M 학습을 순서대로 실행한 뒤, 각 모델을
RF Agent 상대 5000판으로 평가합니다. 조기 종료는 사용하지 않으며 모든
seed를 30M까지 학습합니다.

결과는 다음 위치에 저장됩니다.

```text
runs/ppo_step14_30m_subproc
logs/ppo_step14_30m_subproc
```

학습 중 CPU worker 상태는 `htop`, GPU 상태는 `nvidia-smi`로 확인할 수
있습니다.

### Reproduce Common Rule 40M Training

실행 명령을 먼저 확인합니다.

```bash
scripts/run_common_rule_40m_training.sh --dry-run
```

실제 40M 학습과 공통 평가를 순차 실행합니다.

```bash
scripts/run_common_rule_40m_training.sh
```

기본 설정은 `common_rule_based`, `tactical + terminal`, seed `0, 1, 2`,
seed별 40M, `n_envs=12`, `SubprocVecEnv`, CUDA입니다.

## Evaluation

저장된 PPO 모델을 공통 paired 조건으로 평가하려면 다음 명령을 사용합니다.

```bash
.venv/bin/python scripts/evaluate_common_rule.py \
  --model-path runs/<run-name>/model.zip \
  --training-seed 0 \
  --device cuda \
  --output runs/<run-name>/eval_common_rule_paired_5000.json
```

기본 base seed는 임시로 `100000~102499`를 사용합니다. 팀이 실제 공통 seed
목록을 확정하면 JSON 배열 파일을 `--seed-file`로 전달해야 합니다. 결과에는
seed 목록 SHA-256, 전체·선공·후공 승률, Wilson 95% 신뢰구간, 평균
turn/decision 수, illegal action, evaluation error와 실행 시간이 기록됩니다.

## Documentation

- [RF PPO 팀 회의 보고서](docs/RF_PPO_TEAM_MEETING_REPORT.md)
- [RF 대응 PPO 개발 계획 및 진행 기록](docs/RF_RULE_BASED_PPO_PLAN.md)
- [윷놀이 규칙](docs/RULES.md)
- [강화학습 설계](docs/RL_DESIGN.md)
- [보드 설계](docs/BOARD_DESIGN.md)
