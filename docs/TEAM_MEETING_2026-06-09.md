# 2026-06-09 윷놀이 강화학습 팀 미팅 정리

## 1. 미팅 목적

현재까지 진행한 `RL-yutnori` PPO 실험과 팀원의 `project-RF-` agent
학습·평가 결과를 같은 기준에서 정리하고, 최종 보고서에 사용할 모델과
핵심 결론을 확정한다.

프로젝트의 공동 목표는 다음과 같다.

```text
고정된 Rule-based Agent와 1:1로 대전
선공과 후공을 동일하게 반영
총 5000판 기준 승률 60% 이상 달성
성능 개선 과정을 논리적으로 실험하고 보고
```

## 2. 한눈에 보는 현재 결론

현재 최고 결과는 우리 `RL-yutnori`의 pure PPO다.

| Agent | 유형 | 전체 승률 | 선공 | 후공 | 결과 |
| --- | --- | ---: | ---: | ---: | :---: |
| RL-yutnori 40M seed 1 | Pure PPO | **60.46%** | 60.76% | 60.16% | 통과 |
| RL-yutnori 40M seed 2 | Pure PPO | **60.40%** | 61.32% | 59.48% | 통과 |
| project-RF ppo_capture_imitation | RL + Rule Hybrid | 59.46% | 60.20% | 58.72% | 미달 |
| RL-yutnori 40M seed 0 | Pure PPO | 58.42% | 59.00% | 57.84% | 미달 |
| project-RF ppo_tactical | RL + Rule Hybrid | 55.40% | 57.40% | 53.40% | 미달 |

따라서 제출 후보는 다음과 같이 정리할 수 있다.

```text
주 모델: RL-yutnori 40M seed 1
백업 모델: RL-yutnori 40M seed 2
주 모델 유형: Pure PPO
최종 공통 평가 승률: 3023 / 5000 = 60.46%
```

다만 seed 1의 통과 여유는 23승, seed 2는 20승으로 크지 않다. 세 seed의
pooled 승률은 `59.76%`이므로, “모든 학습 seed가 안정적으로 60%를
넘었다”가 아니라 “공통 기준을 통과하는 pure PPO checkpoint를
확보했다”라고 표현하는 것이 정확하다.

## 3. 공통 평가 기준

최종 비교에는 `COMMON_RULE_BASED_EVALUATION_GUIDE.md`의 paired-seed
방식을 사용했다.

```text
base seed: 2500개
각 base seed에서 모델 선공 1판
같은 base seed에서 모델 후공 1판
총 게임 수: 5000판
policy: deterministic
pass threshold: 60%
```

현재 사용한 임시 seed 범위는 `100000~102499`다.

```text
seed SHA-256:
ca2043aa9201169d58d9aea993ac1d30af5f6c1202387b4ece834a36218370a1
```

공통 Rule-based Agent는 legal action마다 다음 점수를 계산한다.

```text
완주:                       +100
상대 말 잡기:                +50
대기 중인 말 출발:            +5
업힌 말 이동: +4 * (stack_size - 1)
완주 거리:    -0.5 * distance_to_finish
```

점수가 같은 action이 여러 개면 가장 작은 action ID를 선택한다.

## 4. 우리 PPO 설계

### 4.1 알고리즘

우리 모델은 `sb3-contrib`의 `MaskablePPO`를 사용한다. legal action mask를
적용해 불가능한 행동은 policy 선택에서 제외한다.

```text
말 수: 플레이어당 4개
윷 결과: 도, 개, 걸, 윷, 모
action 수: 4 * 5 = 20
action 의미: 어떤 말에 현재 보유한 윷 결과를 적용할지 선택
```

윷을 던지는 행위 자체는 agent action이 아니다. 윷/모로 생성된 결과는
pool에 저장되며, agent가 사용할 결과와 움직일 말을 선택한다.

### 4.2 Tactical Observation

초기 base observation은 말 위치, 상태, stack, 윷 결과 pool을 제공했다.
하지만 PPO가 각 action의 잡기·완주·이동 결과를 신경망 내부에서 모두
추론해야 했다.

이를 개선하기 위해 모든 action에 다음 전술 feature를 추가했다.

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
rule-based action score
```

이 observation은 rule-based 지식을 action을 강제로 바꾸는 규칙이 아니라,
PPO가 현재 action의 의미를 쉽게 학습하도록 제공하는 입력 feature다.

### 4.3 Reward 비교

두 reward 후보를 비교했다.

```text
terminal:
  승리 +1
  패배 -1
  그 외 0

rf_shaped:
  learner capture +0.08
  learner finish +0.15
  shortcut +0.02
  opponent capture -0.08
  opponent finish -0.15
```

`rf_shaped`에는 실제로 잡힌 결과에 대한 penalty가 포함돼 있다. 다만 아직
잡히지 않았지만 다음 턴에 잡히기 쉬운 위치에 놓이는 위험 자체를 직접
penalty로 주지는 않는다.

최종적으로는 `terminal` reward가 Rule-based Agent 상대 전체 승률에서 더
좋았다. capture shaping은 잡기 성향은 강화했지만, 완주와 장기 운영까지
포함한 최종 승률을 더 높이지는 못했다.

## 5. 실험 진행 과정

### 5.1 3M 후보 선별

네 가지 observation/reward 조합을 seed 3개로 비교했다.

| Observation | Reward | 평균 승률 |
| --- | --- | ---: |
| base | terminal | 37.4% |
| base | rf_shaped | 32.4% |
| tactical | terminal | **53.3%** |
| tactical | rf_shaped | 52.3% |

이 실험의 핵심 결론은 다음과 같다.

```text
가장 큰 개선 요인: tactical observation
장기 학습 후보: tactical + terminal
reward shaping 효과: 최종 승률 기준으로 제한적
```

### 5.2 기존 Rule-based Agent 상대 10M

`tactical + terminal`을 seed별 10M timesteps로 학습했다.

| Seed | 5000판 승률 |
| ---: | ---: |
| 0 | 56.92% |
| 1 | 58.38% |
| 2 | 58.56% |
| 평균 | **57.95%** |

3M 평균 `53.3%`에서 10M 평균 `57.95%`로 올라, 학습량 증가에 따른 개선
가능성을 확인했다.

### 5.3 기존 Rule-based Agent 상대 30M

12개 CPU core를 활용하기 위해 `SubprocVecEnv`, `n_envs=12`를 사용했다.
A100 GPU에서 seed별 30M fresh training을 수행했다.

| Seed | 5000판 승률 |
| ---: | ---: |
| 0 | **60.00%** |
| 1 | 58.88% |
| 2 | 59.66% |
| 평균 | **59.51%** |

기존 평가 기준에서는 seed 0이 정확히 60%로 통과했다. 처리량도
`DummyVecEnv` 대비 약 41.8% 증가했다.

### 5.4 공통 평가 가이드 적용

공통 평가에서는 다음 두 조건이 달라졌다.

- base seed마다 선공과 후공을 한 번씩 정확히 pairing
- Rule-based score 동점 시 가장 작은 action ID 선택

기존 30M checkpoint를 공통 방식으로 다시 평가한 결과:

| Seed | 전체 | 선공 | 후공 |
| ---: | ---: | ---: | ---: |
| 0 | 57.34% | 58.80% | 55.88% |
| 1 | 55.94% | 57.60% | 54.28% |
| 2 | 56.20% | 57.56% | 54.84% |
| pooled | **56.49%** | 57.99% | 55.00% |

기존 평균 `59.51%`보다 `3.02%p` 낮아졌다.

원인을 분리한 결과:

```text
pairing 및 seed 방식 변화 영향: 약 -0.73%p
Rule-based tie-break 변화 영향: 약 -2.29%p
```

말은 규칙상 대칭이지만 observation은 말 ID별 슬롯을 갖는다. 상대의
tie-break 정책이 바뀌면 상대 말 ID의 배치 패턴도 달라지며, 기존 PPO가
이 패턴에 일부 적응했다는 것을 확인했다.

### 5.5 공통 Rule-based Agent 상대 40M 재학습

평가 상대와 학습 상대를 일치시키기 위해 `common_rule_based`를 직접
opponent로 사용했다.

```text
observation: tactical
reward: terminal
timesteps: seed별 40M
n_envs: 12
vector env: SubprocVecEnv
device: NVIDIA A100
```

최종 공통 paired 5000판 결과:

| Seed | Wins | 전체 | 선공 | 후공 | 결과 |
| ---: | ---: | ---: | ---: | ---: | :---: |
| 0 | 2921 | 58.42% | 59.00% | 57.84% | 미달 |
| 1 | 3023 | **60.46%** | 60.76% | 60.16% | 통과 |
| 2 | 3020 | **60.40%** | 61.32% | 59.48% | 통과 |

집계:

```text
pooled wins: 8964 / 15000
pooled win rate: 59.76%
seed 표준편차: 0.95%p
선공 pooled: 60.36%
후공 pooled: 59.16%
통과 seed: 2 / 3
illegal actions: 0
evaluation errors: 0
```

30M common 평가 대비 개선:

```text
30M common: 56.49%
40M common: 59.76%
개선 폭: +3.27%p
```

특히 후공 승률이 `55.00%`에서 `59.16%`로 `4.16%p` 개선됐다. 공통
opponent로 직접 재학습한 것이 전체 개선의 가장 중요한 요인이다.

## 6. 학습 중 해결한 주요 문제

10M 장기 학습 중 다음 오류가 발생했다.

```text
RuntimeError: step called while learner is not the current player
```

opponent 선공 episode에서 아주 긴 윷/모 보너스 연속으로 learner가 한 번도
행동하기 전에 게임이 끝날 수 있었다. reset이 이 terminal state를
반환하면서 다음 PPO step에서 오류가 발생했다.

수정:

- opponent opening에서 terminal이 된 reset은 폐기
- 새 game을 재샘플링
- `skipped_terminal_resets` 기록
- deterministic regression test 추가

이후 30M과 40M 학습은 정상 완료됐다.

## 7. 팀원 project-RF Agent 학습

팀원 프로젝트의 원래 설정을 유지해 capture-aware 후보를 학습했다.
현재 project-RF 학습 스크립트는 model과 tensor를 CUDA로 이동하지 않으므로
실제 학습은 CPU에서 진행됐다.

```text
mode: --train-capture-agents
capture samples: 8000
capture imitation epochs: 8
evaluation games: 1000
CPU threads: 12
```

### 7.1 ppo_capture_imitation

`ppo_capture_imitation`은 다음 요소를 결합한 agent다.

```text
Masked PPO policy network
StrategicValue teacher imitation/distillation
capture/danger/finish 상태 oversampling
inference-time tactical prior
```

checkpoint는 252차원 state와 20개 action logit을 사용하는 PPO network다.
하지만 원래 project-RF tournament에서는 `CaptureAwarePPOAgent`로 로드해
잡기, 완주, 위험 회피, 상대 반격 점수를 policy logit에 추가한다.

따라서 이 모델은 `Pure PPO`가 아니라 `RL + Rule Hybrid`로 분류해야 한다.

imitation candidate 결과:

| Candidate | StrategicValue 상대 승률 |
| ---: | ---: |
| 0 | **25.2%** |
| 1 | 24.5% |
| 2 | 24.5% |
| 3 | 24.0% |

candidate 0이 `ppo_capture_imitation.pt`로 선택됐다.

### 7.2 ppo_tactical

`ppo_capture_imitation` 가중치를 시작점으로 mixed opponent fine-tuning을
진행한 모델이다.

| Stage | StrategicValue 상대 승률 |
| --- | ---: |
| mixed stage 1 | 26.1% |
| mixed stage 2 | 25.7% |
| mixed stage 3 | 23.9% |

마지막 stage에서 오히려 성능이 하락해, 더 많은 fine-tuning이 항상 더 좋은
정책으로 이어지지는 않는다는 점을 보여준다.

## 8. project-RF 평가 Adapter

project-RF와 RL-yutnori는 action encoding과 일부 board route가 다르기
때문에 checkpoint를 바로 평가할 수 없다.

action mapping:

```text
project-RF: action = yut_index * 4 + piece_id
RL-yutnori: action = piece_id * 5 + yut_index
```

adapter는 다음을 수행한다.

- 공통 `GameState`를 project-RF 252차원 state로 변환
- project-RF action logit과 local legal action을 양방향 변환
- project-RF에 없는 `A3`, `A4`를 남은 완주 거리 기준 위치로 투영
- 동일한 common paired-seed evaluator에서 state-based agent 실행

### 8.1 미래 RNG 문제

project-RF 원본 tactical prior는 후보 행동을 복제 환경에서 실행한다.
턴이 바뀌면 복제된 실제 RNG 상태로 다음 윷을 던지고, 그 결과를 상대
반격 점수에 사용한다.

공통 가이드에서는 다음을 금지한다.

```text
다음에 나올 윷 결과 사용
평가 환경의 RNG 객체 또는 내부 RNG 상태 사용
복제 환경에 실제 평가 RNG 상태 전달
```

원본 prior를 그대로 재현한 초기 adapter는 5000판에서 `68.50%`를
기록했지만, 미래 정보를 사용하므로 이 결과는 공식 결과에서 제외했다.

최종 compliant adapter는:

- 평가 RNG를 복사하지 않음
- 미래 윷을 샘플링하지 않음
- 즉시 이동, 잡기, 완주 결과만 simulation
- project-RF의 고정 윷 확률로 single-roll expected counterplay 계산

## 9. project-RF 공통 평가 결과

우리 PPO와 동일한 base seed와 common Rule-based Agent를 사용했다.

| Model | 유형 | Wins | 전체 | 선공 | 후공 | 95% CI | 결과 |
| --- | --- | ---: | ---: | ---: | ---: | --- | :---: |
| ppo_capture_imitation | RL + Rule Hybrid | 2973 | **59.46%** | 60.20% | 58.72% | 58.09~60.81% | 미달 |
| ppo_tactical | RL + Rule Hybrid | 2770 | 55.40% | 57.40% | 53.40% | 54.02~56.77% | 미달 |

두 모델 모두:

```text
completed games: 5000 / 5000
illegal actions: 0
evaluation errors: 0
```

`ppo_capture_imitation`은 3000승 기준에 27승 부족했다. 신뢰구간은 60%를
포함하므로 매우 근접한 결과지만, 고정된 pass 판정에서는 실패다.

같은 `ppo_capture_imitation` checkpoint에서 tactical prior를 끈
network-only 100판 smoke 승률은 `13%`였다. 이는 높은 성능의 대부분이
policy network 단독이 아니라 tactical prior에서 나온다는 강한 증거다.

## 10. 두 프로젝트 비교 해석

### 10.1 Rule-based 지식의 효율성

이번 단순화된 윷놀이에서는 잡기, 완주, 지름길, 위험 회피처럼 강한 단기
전술이 명확하다. 이런 전술은 hand-crafted rule로 빠르게 표현할 수 있어
rule-based agent가 높은 sample efficiency를 보였다.

project-RF hybrid 모델도 tactical prior를 제거하면 성능이 크게 떨어졌다.
따라서 이번 환경에서 rule-based knowledge가 매우 강한 inductive bias로
작동했다고 볼 수 있다.

### 10.2 Pure PPO의 의미

우리 PPO는 action override 없이 observation과 최종 승패 reward를 통해
학습했다. 공통 상대에 대해 seed별 40M을 학습한 뒤 2개 checkpoint가 60%
기준을 넘었다.

즉, rule-based 방식이 더 빠르고 효율적이었지만 충분한 학습량과 적절한
state 설계를 적용하면 pure PPO도 Rule-based Agent를 근소하게 넘을 수
있음을 확인했다.

### 10.3 뒷도 등 규칙 확장 시 예상

뒷도와 같은 규칙을 추가해도 초기에는 rule-based가 빠르게 강한 성능을
낼 가능성이 높다. 그러나 다음과 같은 예외와 장기 판단이 증가한다.

- 뒷도로 잡기
- 위험한 말을 뒤로 이동해 회피
- 시작점과 도착점 부근의 특수 처리
- 업힌 말의 후진
- 지름길과 후진의 결합
- 상대의 다음 반격 가능성

규칙이 복잡해질수록 hand-crafted rule의 분기와 유지 비용도 증가한다.
따라서 확장된 환경에서는 pure rule-based보다 rule-guided RL 또는 hybrid
방식의 장점이 커질 가능성이 있다. 다만 RL도 state/action 설계와 희소
상황 sampling이 부족하면 자동으로 좋아지지는 않는다.

## 11. 핵심 실험 결론

1. `tactical observation`이 가장 큰 PPO 성능 개선 요인이었다.
2. `terminal reward`가 최종 승률 목표에 더 잘 맞았다.
3. capture shaping은 잡기 성향을 강화했지만 전체 승률을 높이지 못했다.
4. Rule-based tie-break처럼 작은 정책 차이도 PPO 입력 분포에 큰 영향을
   줄 수 있다.
5. 학습 상대와 평가 상대를 일치시킨 40M 재학습이 `+3.27%p` 개선을 만들었다.
6. 우리 pure PPO seed 1과 seed 2가 공통 5000판 기준 60%를 통과했다.
7. project-RF 최고 hybrid 모델은 59.46%로 매우 근접했지만 통과하지 못했다.
8. project-RF 모델은 policy network보다 tactical prior 의존도가 컸다.
9. 모든 최종 평가에서 illegal action과 evaluation error는 0건이었다.

## 12. 현재 한계

- 40M seed 0은 58.42%로 통과하지 못했다.
- 세 seed pooled 승률은 59.76%로 60%보다 0.24%p 낮다.
- 통과한 seed 1과 seed 2도 margin이 크지 않다.
- 현재 공통 seed 범위는 팀이 최종 확정한 비공개 seed 파일이 아니다.
- project-RF는 우리와 학습 환경 및 일부 board route가 다르다.
- project-RF의 `A3`, `A4`는 정확히 대응할 수 없어 거리 기반 투영을 사용했다.
- project-RF 최종 비교는 동일 평가 환경이지만 동일 학습 환경 비교는 아니다.
- project-RF checkpoint는 단일 학습 seed 결과다.

## 13. 오늘 미팅에서 결정할 내용

### 13.1 최종 모델

권장안:

```text
주 모델: RL-yutnori common_rule_based seed 1 40M
백업 모델: RL-yutnori common_rule_based seed 2 40M
```

### 13.2 보고서 표현

권장 결론:

```text
공통 Rule-based Agent 상대 paired 5000판 평가에서
pure PPO checkpoint가 60.46% 승률을 기록해 목표를 달성했다.
다만 3개 학습 seed 평균은 59.76%이므로 seed 안정성에는 한계가 있다.
```

project-RF 결과는 다음처럼 표현한다.

```text
팀원 프로젝트의 capture-aware PPO hybrid를 동일 평가 환경에 연결했으며,
최고 59.46%를 기록했다. 해당 모델은 inference-time tactical prior에
크게 의존하므로 pure PPO가 아닌 RL + Rule Hybrid로 분류했다.
```

### 13.3 추가 실험 여부

제출일은 2026년 6월 12일 금요일이다. 현재는 목표를 통과한 모델과 비교
실험이 모두 확보됐으므로 새로운 장기 학습의 우선순위는 낮다.

남은 시간의 권장 사용 순서:

1. 최종 보고서와 발표자료 작성
2. 수치와 모델 경로 재확인
3. 공통 평가 조건과 한계 명시
4. 시간이 남을 때만 32M/36M/40M checkpoint 선별 평가

새로운 40M 이상 장기 학습은 성능 향상을 보장하지 않고 보고서 작성 시간을
줄일 수 있으므로 현재 시점에서는 권장하지 않는다.

## 14. 검증 및 산출물

전체 regression:

```text
149 passed
```

주요 파일:

```text
우리 최종 후보:
bests/ppo_common_rule_40m_subproc/common_rule_based_seed_1_40m_nenv12_tactical/model.zip

우리 백업 후보:
bests/ppo_common_rule_40m_subproc/common_rule_based_seed_2_40m_nenv12_tactical/model.zip

project-RF checkpoint:
/home/david-nam/work-space/project-RF-/results/ppo_training/ppo_capture_imitation.pt
/home/david-nam/work-space/project-RF-/results/ppo_training/ppo_tactical.pt

project-RF adapter:
yutnori/agents/project_rf_checkpoint.py

project-RF 공통 평가 script:
scripts/evaluate_project_rf_common.py
```

관련 문서:

```text
docs/COMMON_RULE_BASED_EVALUATION_GUIDE.md
docs/RF_RULE_BASED_PPO_PLAN.md
docs/RF_PPO_TEAM_MEETING_REPORT.md
docs/PROJECT_RF_CROSS_ENV_EVALUATION.md
```

## 15. 1분 발표 요약

우리 팀은 Rule-based Agent를 분석해 동일한 평가 환경을 만들고,
MaskablePPO의 observation과 reward를 비교했다. 3M 후보 실험에서는
잡기·완주·거리 정보를 action별로 제공하는 tactical observation이 가장
큰 효과를 보였으며, reward shaping보다 최종 승패만 사용하는 terminal
reward가 더 좋은 결과를 냈다.

10M과 30M으로 학습량을 늘린 뒤 공통 평가 가이드를 적용했고, tie-break
정책 차이 때문에 기존 모델 성능이 하락하는 것을 확인했다. 이에 공통
Rule-based Agent를 직접 상대해 seed별 40M을 재학습했다. 최종적으로 seed
1은 60.46%, seed 2는 60.40%를 기록해 pure PPO로 목표를 달성했다.

팀원 project-RF의 capture-aware PPO도 같은 평가 환경에 adapter로 연결했다.
미래 RNG를 사용하지 않도록 tactical prior를 수정한 공통 평가에서 최고
59.46%를 기록했다. 이 모델은 network-only 성능이 낮고 tactical prior
의존도가 높아 RL + Rule Hybrid로 분류했다.

최종적으로 rule-based 지식은 윷놀이에서 매우 효율적인 inductive bias였고,
우리 실험은 충분한 학습량과 tactical observation을 사용하면 pure PPO도
고정 Rule-based Agent를 근소하게 넘어설 수 있음을 보여준다.
