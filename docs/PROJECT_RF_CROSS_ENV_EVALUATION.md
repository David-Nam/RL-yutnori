# project-RF Agent 공통 환경 교차 평가

## 1. 목적

팀원의 `project-RF-` 저장소에서 원래 설정으로 학습한 agent를
`RL-yutnori`의 공통 평가 환경에 연결했다. 목적은 학습 환경은 유지하되,
최종 성능은 우리 PPO와 동일한 Rule-based Agent, seed, 선후공 조건으로
비교하는 것이다.

이 평가는 같은 환경에서 다시 학습한 결과가 아니라, 서로 다른 학습 환경
사이의 `cross-environment transfer` 평가다.

## 2. project-RF 학습

실행 설정:

```text
script: train/train_ppo.py
mode: --train-capture-agents
capture samples: 8000
capture imitation epochs: 8
evaluation games: 1000
CPU thread setting: 12
actual training device: CPU
```

실행 명령:

```bash
CUDA_VISIBLE_DEVICES="" \
OMP_NUM_THREADS=12 \
MKL_NUM_THREADS=12 \
OPENBLAS_NUM_THREADS=12 \
NUMEXPR_NUM_THREADS=12 \
.venv/bin/python -u train/train_ppo.py \
  --train-capture-agents \
  --out-dir results/ppo_training \
  --eval-games 1000 \
  --capture-samples 8000 \
  --capture-imitation-epochs 8
```

candidate imitation 결과:

| Candidate | LR | Gamma | GAE lambda | StrategicValue 상대 승률 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.0001 | 0.99 | 0.95 | 25.2% |
| 1 | 0.00005 | 0.995 | 0.98 | 24.5% |
| 2 | 0.0001 | 0.995 | 0.95 | 24.5% |
| 3 | 0.00005 | 0.99 | 0.98 | 24.0% |

candidate 0이 `ppo_capture_imitation.pt`로 선택됐다. 이를 이어서 mixed
opponent fine-tuning을 수행한 `ppo_tactical.pt`의 단계별 StrategicValue
상대 승률은 `26.1%`, `25.7%`, `23.9%`였다.

생성된 주요 checkpoint:

```text
project-RF-/results/ppo_training/ppo_capture_imitation.pt
project-RF-/results/ppo_training/ppo_tactical.pt
```

## 3. 평가 Adapter

구현 파일:

```text
yutnori/agents/project_rf_checkpoint.py
scripts/evaluate_project_rf_common.py
```

adapter는 checkpoint의 252차원 policy network 가중치를 수정하지 않고
로드한다. 공통 `GameState`를 project-RF 입력 형식으로 변환한 뒤,
project-RF action logit을 공통 action ID로 다시 변환한다.

action encoding:

```text
project-RF: action = yut_index * 4 + piece_id
RL-yutnori: action = piece_id * 5 + yut_index
```

board position은 대부분 직접 대응한다. project-RF에 존재하지 않는
`A3`, `A4`는 남은 완주 거리를 보존하도록 각각 project-RF 위치 `12`,
`13`으로 투영한다. 따라서 이 평가는 완전히 동일한 상태 공간 변환이
아니며, 결과 해석에 이 domain 차이를 포함해야 한다.

## 4. 미래 RNG 준수

project-RF의 원래 `CaptureAwarePPOAgent` tactical prior는 복제 환경에서
후보 행동을 실행한다. 턴이 바뀌면 복제된 실제 RNG 상태로 다음 윷을
던지고, 그 결과를 상대 반격 점수에 사용한다.

이는 공통 가이드의 다음 조건과 충돌한다.

```text
행동을 시뮬레이션하는 agent는 미래 윷 결과를 볼 수 없어야 한다.
시뮬레이션용 환경 복사본에 실제 평가 환경의 RNG 상태를 전달해서는 안 된다.
```

초기 adapter도 이 동작을 그대로 재현해 5000판에서 `68.50%`를 기록했지만,
이 결과는 공통 평가 결과로 인정하지 않는다.

수정된 adapter는:

- 평가 환경 RNG 객체와 상태를 복사하지 않는다.
- 후보 행동의 즉시 이동, 잡기, 완주만 계산한다.
- 상대 반격은 실제 다음 윷이 아니라 project-RF에 고정된 윷 확률의
  single-roll 기대값으로 계산한다.
- simulation sampler가 호출되면 즉시 실패하도록 막는다.

따라서 아래 최종 결과는 공통 가이드의 정보 제한을 만족한다.

## 5. 공통 평가 조건

```text
opponent: common_rule_based
base seeds: 100000~102499
games per base seed: model 선공 1판 + 후공 1판
total games: 5000
deterministic policy: true
tactical weight: 2.5
seed SHA-256: ca2043aa9201169d58d9aea993ac1d30af5f6c1202387b4ece834a36218370a1
```

평가 명령:

```bash
.venv/bin/python scripts/evaluate_project_rf_common.py \
  --model-path /home/david-nam/work-space/project-RF-/results/ppo_training/ppo_capture_imitation.pt \
  --output runs/project_rf_common_eval/ppo_capture_imitation_common_paired_5000.json
```

## 6. 최종 결과

| Model | 유형 | Wins | 전체 | 선공 | 후공 | 95% CI | Passed |
| --- | --- | ---: | ---: | ---: | ---: | --- | :---: |
| ppo_capture_imitation | RL + Rule Hybrid | 2973 | **59.46%** | 60.20% | 58.72% | 58.09~60.81% | false |
| ppo_tactical | RL + Rule Hybrid | 2770 | 55.40% | 57.40% | 53.40% | 54.02~56.77% | false |

두 모델 모두:

```text
completed games: 5000 / 5000
illegal actions: 0
evaluation errors: 0
```

`ppo_capture_imitation`은 3000승 기준에 27승 부족해 엄격한 60% 기준을
통과하지 못했다. 다만 신뢰구간은 60%를 포함하고 있어 목표에 매우
근접한 단일 checkpoint로 볼 수 있다.

같은 checkpoint에서 tactical prior를 끈 network-only 100판 smoke 승률은
`13%`였다. 이 모델의 성능은 신경망 단독보다 inference-time tactical
prior에 크게 의존하므로 `Pure RL`이 아니라 `RL + Rule Hybrid`로
분류해야 한다.

## 7. 우리 PPO와 비교

동일한 5000판 평가에서 현재 주요 결과:

| Agent | 유형 | 전체 승률 |
| --- | --- | ---: |
| RL-yutnori seed 1 40M | Pure PPO | **60.46%** |
| RL-yutnori seed 2 40M | Pure PPO | 60.40% |
| project-RF ppo_capture_imitation | RL + Rule Hybrid | 59.46% |
| RL-yutnori seed 0 40M | Pure PPO | 58.42% |
| project-RF ppo_tactical | RL + Rule Hybrid | 55.40% |

현재 최종 결론은 다음과 같다.

- 우리 seed 1 pure PPO가 전체 최고이며 60% 기준을 통과했다.
- 팀원 모델 중에는 `ppo_capture_imitation`이 가장 강하지만 59.46%로
  기준에 근소하게 미달했다.
- 학습 환경과 board rule이 다른 모델도 adapter를 통해 공통 환경에서
  illegal action 없이 평가할 수 있음을 확인했다.
- project-RF 모델의 성능은 tactical prior 의존성이 매우 크므로 보고서에서
  순수 PPO 성능으로 해석하면 안 된다.

## 8. 한계

- project-RF checkpoint는 단일 학습 seed 결과다.
- 두 학습 환경의 지름길 처리 규칙이 완전히 같지 않다.
- `A3`, `A4`는 직접 대응 위치가 없어 거리 기반으로 투영했다.
- compliant tactical prior는 미래 RNG를 제거하기 위해 원본 prior의
  상대 반격 계산을 기대값 방식으로 바꿨다.
- 따라서 이 결과는 동일 checkpoint의 공통 환경 적응 성능이지,
  project-RF 원래 환경 결과의 재현값은 아니다.
