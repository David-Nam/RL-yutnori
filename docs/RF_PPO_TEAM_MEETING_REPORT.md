# RF Rule-based Agent 대응 PPO 실험 보고

작성일: 2026-06-06
최종 업데이트: 2026-06-07

## 1. 목표

이번 실험의 목표는 `project-RF-` repo에 구현된 Rule-based Agent를
고정 opponent로 두고, 우리 PPO 기반 모델이 1:1 대전에서 승률 `60%`
이상을 달성할 수 있는지 확인하는 것이다.

최종 평가 조건은 다음과 같이 두었다.

```text
opponent: project_rf_rule
대전 방식: 1:1
선/후공: 랜덤
평가 판수: 5000판
pass threshold: 60%
```

## 2. PPO 후보 구성

현재 학습 모델은 `MaskablePPO` 기반이다. action mask를 사용해 illegal
action을 정책 선택에서 제외하고, learner는 `YutnoriEnv`의 단일 player
관점에서 학습한다.

이번 후보는 observation과 reward를 축으로 나눴다.

### 2.1 Observation 후보

#### base observation

기본 observation은 다음 정보를 포함한다.

- learner 말 위치와 상태
- learner stack 정보
- opponent 말 위치와 상태
- opponent stack 정보
- 현재 윷 결과 pool

장점은 단순하고 기존 PPO 실험과의 연속성이 좋다는 점이다. 단점은 각
legal action을 했을 때 capture, finish, 거리 변화가 어떻게 되는지 모델이
직접 추론해야 한다는 점이다.

#### tactical observation

`tactical` observation은 base observation 뒤에 모든 action에 대한
전술 feature를 붙인다.

각 action row는 다음 feature를 가진다.

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

이 후보의 의도는 PPO가 보드 상태만 보고 action 결과를 추론하게 두는 대신,
각 action의 전술적 의미를 직접 관찰하게 만드는 것이다. 특히 이번 목표
opponent가 rule-based heuristic을 사용하므로, capture/finish/distance 같은
명시적 action feature가 효과적일 가능성이 높다고 봤다.

### 2.2 Reward 후보

#### terminal reward

가장 기본적인 sparse reward다.

```text
learner 승리: +1
learner 패배: -1
non-terminal step: 0
```

장점은 최종 목표인 승패와 완전히 일치한다는 점이다. 단점은 학습 신호가
늦게 들어오기 때문에 긴 credit assignment가 어렵다는 점이다.

#### rf_shaped reward

Rule-based Agent의 전술 선호를 반영하기 위해 만든 shaped reward다.

```text
learner capture: +0.08 * captured_count
learner finish: +0.15 * finished_count
learner shortcut: +0.02

opponent capture: -0.08 * captured_count
opponent finish: -0.15 * finished_count
```

이 reward는 실제로 잡거나 잡히는 event에는 반응한다. 다만 상대가 다음
턴에 쉽게 잡을 수 있는 위치에 내 말을 노출하는 위험도 자체는 아직
직접 penalty로 반영하지 않는다. 즉, 결과 기반 capture/captured reward는
있지만, 미래 capture-risk를 예측해 피하는 reward는 아직 없다.

## 3. 사전 학습의 의미

여기서 사전 학습은 최종 모델을 바로 고르는 단계가 아니라, 여러 PPO 구성
중 어떤 방향이 장기 학습에 투자할 만한지 고르는 후보 선별 실험이다.

실험은 Step 13에서 진행했다.

```text
opponent: project_rf_rule
seeds: 0, 1, 2
timesteps: 3M
eval games: RF target 상대 1000판
device: cuda
GPU: NVIDIA A100-SXM4-40GB
```

비교한 후보는 네 가지다.

```text
base + terminal
base + rf_shaped
tactical + terminal
tactical + rf_shaped
```

## 4. 사전 학습 결과

RF target 1000판 평가 결과는 다음과 같다.

| observation | reward | seed 0 | seed 1 | seed 2 | mean | min | max |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base | terminal | 0.397 | 0.372 | 0.353 | 0.374 | 0.353 | 0.397 |
| base | rf_shaped | 0.283 | 0.323 | 0.365 | 0.324 | 0.283 | 0.365 |
| tactical | terminal | 0.525 | 0.545 | 0.528 | 0.533 | 0.525 | 0.545 |
| tactical | rf_shaped | 0.512 | 0.519 | 0.539 | 0.523 | 0.512 | 0.539 |

이 결과에서 가장 중요한 결론은 observation 쪽이다.

`tactical` observation은 base observation보다 RF target 상대 승률을 크게
끌어올렸다.

```text
base + terminal 평균: 0.374
tactical + terminal 평균: 0.533
```

반면 `rf_shaped` reward는 기대와 다르게 RF target 상대 승률을 올리지
못했다.

```text
base:     terminal 0.374 -> rf_shaped 0.324
tactical: terminal 0.533 -> rf_shaped 0.523
```

다만 `tactical + rf_shaped`는 `capture_first` 상대 성능이 더 좋았다.
이는 shaped reward가 capture 성향은 강화했지만, RF Agent를 이기는 전체
운영 능력을 개선했다고 보기는 어렵다는 뜻이다.

사전 학습 결론은 다음과 같다.

```text
1순위 장기 학습 후보: tactical + terminal
2순위 비교 후보: tactical + rf_shaped
base 계열 후보: 장기 학습 우선순위에서 제외
```

## 5. 장기 학습 진행 방식

Step 14에서는 사전 학습에서 가장 좋은 후보였던 `tactical + terminal`을
장기 학습했다.

실행 설정은 다음과 같다.

```text
observation: tactical
reward: terminal
opponent: project_rf_rule
seeds: 0, 1, 2
timesteps: 10M
n_envs: 16
device: cuda
GPU: NVIDIA A100-SXM4-40GB
checkpoint_freq: 1M
official eval: RF target 상대 5000판
```

실행 경로:

```text
runs/ppo_step14_long_training_v2
logs/ppo_step14_long_training_v2
```

각 seed마다 10M timesteps 학습 후, `evaluate_rf_target.py`를 사용해
RF Agent 상대 공식 5000판 평가를 수행했다.

## 6. 학습 중 발견한 이슈

최초 Step 14 실행 중 다음 오류가 발생했다.

```text
RuntimeError: step called while learner is not the current player
```

원인은 env reset 경계 조건이었다. opponent가 선공인 episode에서 reset 중
opponent turn을 내부에서 자동 진행하는데, 아주 긴 윷/모 보너스 연속이
나오면 learner가 한 번도 행동하기 전에 opponent가 게임을 끝낼 수 있었다.
기존 구현은 이 terminal 상태를 그대로 반환할 수 있었고, PPO가 다음
step을 호출하면서 learner turn이 아니라는 오류가 발생했다.

수정 내용은 다음과 같다.

- `YutnoriEnv.reset()`이 opponent opening 중 terminal이 된 game을 버리고
  새 game을 재샘플링하도록 수정
- reset info에 `skipped_terminal_resets` 기록 추가
- deterministic regression test 추가
- 전체 regression test 통과

검증 결과:

```text
tests/test_yutnori_env.py: 21 passed
related tests: 27 passed
full regression: 128 passed
```

수정 후 Step 14는 `*_v2` 경로에서 다시 실행했다.

## 7. 장기 학습 결과

공식 RF target 5000판 평가 결과는 다음과 같다.

| seed | wins | losses | win_rate | passed | illegal | starting player |
| ---: | ---: | ---: | ---: | :---: | ---: | --- |
| 0 | 2846 | 2154 | 0.5692 | false | 0 | 2509 / 2491 |
| 1 | 2919 | 2081 | 0.5838 | false | 0 | 2509 / 2491 |
| 2 | 2928 | 2072 | 0.5856 | false | 0 | 2510 / 2490 |

집계 결과:

```text
mean win_rate: 0.5795
min win_rate: 0.5692
max win_rate: 0.5856
population stdev: 0.0073
total official games: 15000
total wins: 8693
illegal_action_count sum: 0
all passed: false
```

학습 중 episode 기준 승률과 random 상대 학습 후 평가도 같이 확인했다.

| seed | trained timesteps | train episode win_rate | random after eval |
| ---: | ---: | ---: | ---: |
| 0 | 10,027,008 | 0.5110 | 0.820 |
| 1 | 10,027,008 | 0.5106 | 0.850 |
| 2 | 10,027,008 | 0.5123 | 0.800 |

각 run은 약 1시간 20분에서 1시간 28분 정도 소요되었고, seed마다 1M 단위
checkpoint 10개가 생성되었다.

## 8. 결과 해석

10M 장기 학습은 목표 `60%`에는 도달하지 못했다. 따라서 현재 상태만 보면
공식 pass는 실패다.

하지만 실험적으로는 의미 있는 개선이 있었다.

```text
3M tactical + terminal 평균: 0.533
10M tactical + terminal 평균: 0.5795
개선 폭: +0.0468
```

seed별 결과가 모두 `56.9%~58.6%`에 모여 있어, 특정 seed 하나만 우연히
좋았던 결과로 보기는 어렵다. 특히 seed 1과 seed 2는 `58%` 중반까지
올라왔다.

통계적으로도 15000판 pooled win rate는 `0.5795`다. 5000판 평가 하나의
표준 오차는 대략 `0.7%p` 수준이고, 15000판 pooled 기준 표준 오차는 약
`0.4%p` 수준이다. 따라서 현재 모델이 이미 안정적으로 60%를 넘는다고
말하기는 어렵다. 다만 10M까지 학습량을 늘렸을 때 상승 추세가 확인됐기
때문에 pure PPO의 가능성을 완전히 접기에는 이르다.

핵심 해석은 다음과 같다.

- PPO가 RF Agent의 규칙성을 어느 정도 학습하고 있다.
- 성능 향상의 핵심은 `rf_shaped` reward보다 `tactical` observation이다.
- `terminal` reward가 RF target 상대 전체 승률에는 더 안정적이었다.
- capture reward는 capture 성향을 키우지만, 전체 승률 최적화에는 부족했다.
- 현재 모델은 60% 목표 바로 아래까지 접근했으므로, 30M 확장 실험의 가치가
  있다.

## 9. 다음 단계 제안

현재 결론은 다음과 같다.

```text
Pure PPO 최고 후보: tactical + terminal
10M 공식 평균 승률: 57.95%
목표 60%: 미달
다음 우선순위: tactical + terminal 30M fresh run
```

팀 논의 후 다음 액션은 `30M` fresh run으로 확장했다.

```bash
scripts/run_step14_30m_training.sh
```

기존 10M 학습은 `DummyVecEnv`를 사용했기 때문에 `n_envs=16`이어도 env가
한 process에서 순차 실행됐다. 30M 실행에서는 CPU 12 core를 활용하기 위해
`SubprocVecEnv`와 `n_envs=12`를 사용한다. 각 env가 별도 process에서
실행되며, BLAS/OpenMP thread는 process당 1개로 제한한다.

따라서 10M과 30M은 timesteps만 바뀐 완전한 단일 변수 비교는 아니다.
이번 설정은 동일한 PPO 후보를 유지하면서 CPU 병렬 활용과 최종 성능 탐색을
우선한 실행 구성이다.

30M 결과 판단 기준은 다음과 같이 둔다.

- 평균 승률이 `60%` 이상이고 seed별 결과가 안정적이면 pure PPO 후보를
  최종 후보로 채택한다.
- 평균 승률이 `58~60%`에 머물면 checkpoint별 평가 또는 seed 추가 평가를
  통해 분산을 확인한다.
- 평균 승률이 `60%`를 넘지 못하고 plateau가 확인되면 Step 15 hybrid
  policy로 넘어간다.

Step 15 후보는 학습된 PPO action을 기본으로 사용하되, 명확한 전술 상황만
override하는 방식이다.

```text
priority: finish > capture > PPO
```

이 방향은 PPO가 학습한 전체 운영 전략을 유지하면서, 사람이 보아도 명확한
finish/capture 실수를 줄이는 목적이다.

## 10. 회의에서 공유할 핵심 요약

- 3M 후보 선별에서는 `tactical + terminal`이 가장 좋았다.
- 10M 장기 학습 결과 RF Agent 상대 공식 5000판 평균 승률은 `57.95%`였다.
- 30M 확장 결과 평균은 `59.51%`이고 seed 0은 `60.00%`로 통과했다.
- 모든 seed가 10M보다 좋아졌지만, 3개 중 1개만 통과해 안정적 달성은 아니다.
- illegal action은 모든 공식 평가에서 `0`이었다.
- 후반 checkpoint를 선별 평가한 뒤 pure PPO의 최종 가능성을 판단한다.
- checkpoint에서도 안정적인 60% 후보가 없으면 hybrid policy로 넘어간다.

## 11. 30M 확장 학습

10M 결과가 목표 바로 아래까지 올라왔기 때문에 같은
`tactical + terminal` 후보를 seed별 30M timesteps로 fresh training했다.

```text
observation: tactical
reward: terminal
opponent: project_rf_rule
seeds: 0, 1, 2
timesteps: 30M
n_envs: 12
vector env: SubprocVecEnv
device: cuda
GPU: NVIDIA A100-SXM4-40GB
checkpoint: 3M 간격
official eval: seed별 5000판
```

12개 CPU core를 실제 환경 실행에 활용하도록 각 환경을 별도 process로
분리했다. 각 process의 BLAS/OpenMP thread는 1개로 제한했다.

각 seed의 실제 trained timesteps는 `30,007,296`이고, 약 114만 episode를
완료했다. 학습 시간은 seed당 약 2시간 59분에서 3시간 1분이었다.

## 12. 30M 공식 평가 결과

| seed | wins | losses | win_rate | passed | illegal | starting player |
| ---: | ---: | ---: | ---: | :---: | ---: | --- |
| 0 | 3000 | 2000 | 0.6000 | true | 0 | 2509 / 2491 |
| 1 | 2944 | 2056 | 0.5888 | false | 0 | 2509 / 2491 |
| 2 | 2983 | 2017 | 0.5966 | false | 0 | 2510 / 2490 |

```text
mean/pooled win rate: 0.5951
min / max: 0.5888 / 0.6000
population stdev: 0.0047
total wins / games: 8927 / 15000
passed seeds: 1 / 3
illegal actions: 0
```

10M과 비교하면 다음과 같다.

| seed | 10M | 30M | 변화 |
| ---: | ---: | ---: | ---: |
| 0 | 0.5692 | 0.6000 | +0.0308 |
| 1 | 0.5838 | 0.5888 | +0.0050 |
| 2 | 0.5856 | 0.5966 | +0.0110 |
| mean | 0.5795 | 0.5951 | +0.0156 |

모든 seed가 개선됐고 seed 간 표준편차도 `0.73%p`에서 `0.47%p`로 줄었다.
따라서 30M 확장은 평균 성능뿐 아니라 seed 안정성도 개선했다.

## 13. 통계 및 학습 추세 해석

30M pooled 15000판의 Wilson 95% confidence interval은 약
`58.73%~60.30%`다. seed 0의 5000판 관측 승률은 기준과 정확히 같은
`60.00%`이지만, 해당 95% 구간은 약 `58.63%~61.35%`다.

평가 harness의 판정 규칙에서는 seed 0이 통과한 것이 맞다. 다만 통계적으로
“일반적인 실제 승률이 안정적으로 60%보다 높다”고 말하기에는 경계에 있다.
전체 평균도 기준보다 `0.49%p` 낮다.

학습 episode를 3M 구간으로 나누면 세 seed 모두 약 `44%`에서 시작해 마지막
구간에는 `57.4~58.2%`까지 상승했다. 후반에도 상승은 이어졌지만 개선 폭은
둔화됐다. 특히 seed 0은 24~27M 구간 승률이 27~30M보다 소폭 높았다.

이 사실은 30M 최종 checkpoint가 반드시 각 seed의 최적 정책은 아니라는
가능성을 보여준다. PPO는 학습이 진행될수록 단조롭게 좋아지는 알고리즘이
아니므로, 이미 저장한 21M, 24M, 27M checkpoint를 비교할 가치가 있다.

## 14. CPU 병렬화 효과

학습 처리량은 다음과 같이 바뀌었다.

```text
10M DummyVecEnv 평균: 약 1,963 timesteps/s
30M SubprocVecEnv 평균: 약 2,784 timesteps/s
처리량 증가: 약 41.8%
```

10M 상당 시간으로 환산하면 약 85.2분에서 59.9분으로 줄어든다. A100을
유지하면서 CPU 환경 simulation을 병렬화한 효과가 확인됐다.

다만 10M은 `n_envs=16 + DummyVecEnv`, 30M은
`n_envs=12 + SubprocVecEnv`다. rollout batch 크기도 달라졌으므로
30M의 성능 향상을 timesteps 증가 하나로만 해석할 수는 없다.

## 15. 현재 결론과 다음 판단

이번 결과의 결론은 다음과 같다.

```text
Pure PPO 최고 단일 결과: 60.00% (seed 0)
Pure PPO 3-seed 평균: 59.51%
목표 통과 seed: 1 / 3
판단: 최소 목표 달성 후보는 확보, 안정적인 최종 후보는 미확정
```

즉, 개별 모델 5000판 승률을 기준으로 하는 프로젝트의 최소 목표는 seed 0
모델이 달성했다. 다만 통과 여유가 정확히 0승이고 다른 두 training seed는
미달이므로, hybrid policy로 바로 넘어가기 전에 다음 순서로 pure PPO를
한 번 더 검증한다.

1. seed별 21M, 24M, 27M, 30M checkpoint를 별도 evaluation seed로 선별한다.
2. seed별 상위 checkpoint를 공식 조건 5000판으로 재검증한다.
3. 안정적인 60% 후보가 없으면 `finish > capture > PPO` hybrid를 구현한다.

이 순서는 추가 학습 없이 이미 생성한 checkpoint를 활용한다. 학습 후반의
정책 변동 때문에 놓쳤을 수 있는 최적 모델을 먼저 찾은 뒤, 그래도 부족할
때만 정책 구조를 변경한다는 점에서 실험 흐름도 명확하다.

## 16. 공통 평가 가이드 적용

팀 공통 가이드가 추가되면서 기존 평가와 다음 차이가 생겼다.

- 2,500개 base seed마다 모델 선공과 후공을 한 번씩 실행
- 선공 2,500판, 후공 2,500판을 정확히 보장
- Rule-based score 동점 시 가장 작은 action ID 선택
- full game runner를 사용해 learner 첫 행동 전 종료도 정상 게임으로 집계
- 전체/선공/후공 승률, 95% CI, illegal, error, 실행 시간을 함께 보고

기존 `ProjectRFRuleBasedAgent`는 동점에서 큰 action ID를 선택했다. 과거
실험을 재현하기 위해 기존 agent는 보존하고, 공통 평가와 재학습에는
별도의 `CommonRuleBasedAgent`를 사용한다.

평가 script는 `scripts/evaluate_common_rule.py`다. 현재 가이드 문서에는
실제 공통 seed 목록이 없으므로 이번 검증은 임시로
`100000~102499`를 사용했다. 결과 JSON에 seed 목록 SHA-256
`ca2043aa...370a1`을 기록했다. 팀이 seed 목록을 확정하면 `--seed-file`로
같은 JSON 배열을 전달해야 최종 공식 비교가 된다.

## 17. 기존 30M 모델의 공통 평가

| seed | 전체 | 선공 | 후공 | 95% CI | illegal/error |
| ---: | ---: | ---: | ---: | --- | --- |
| 0 | 0.5734 | 0.5880 | 0.5588 | 0.5596~0.5870 | 0 / 0 |
| 1 | 0.5594 | 0.5760 | 0.5428 | 0.5456~0.5731 | 0 / 0 |
| 2 | 0.5620 | 0.5756 | 0.5484 | 0.5482~0.5757 | 0 / 0 |

```text
3-seed pooled: 8474 / 15000 = 0.5649
seed 표준편차: 0.0061
선공 pooled: 0.5799
후공 pooled: 0.5500
pooled 95% CI: 0.5570~0.5728
```

기존 평가 평균 `0.5951`과 비교하면 공통 결과는 `3.02%p` 낮다.

원인을 분리하기 위해 같은 paired seed에서 상대만 기존 큰-ID 동점 agent로
바꿔 진단 평가했다.

```text
기존 큰-ID agent + paired seed: 0.5878
공통 작은-ID agent + paired seed: 0.5649
```

평가 pairing과 seed 배정 변화는 약 `-0.73%p`, opponent 동점 정책 변화는
추가로 약 `-2.29%p` 영향을 줬다. 따라서 주된 하락은 평가 통계 방식이
아니라 실제 상대 정책 변경에서 발생했다.

## 18. 재학습 필요성

규칙상 같은 상태의 말은 대칭이지만 PPO observation은 네 말의 상태를
말 ID별 슬롯으로 전달한다. 기존 상대는 동점에서 큰 action ID를 선택했고,
공통 상대는 작은 action ID를 선택한다. 이 차이는 상대 말의 ID별 배치
패턴을 바꾸며, 모델이 기존 패턴에 적응했다면 정책 입력 분포도 달라진다.

따라서 기존 checkpoint만 고르는 것으로는 충분하지 않다. 먼저 공통 상대를
직접 사용해 같은 `tactical + terminal` PPO를 다시 학습하는 것이 가장
작고 논리적인 변경이다. observation이나 reward까지 동시에 바꾸면 어떤
요인이 개선을 만들었는지 분리하기 어려우므로 이번 야간 학습에서는
그대로 유지한다.

## 19. 12시간 야간 학습 계획

```text
opponent: common_rule_based
observation: tactical
reward: terminal
seeds: 0, 1, 2
timesteps: seed별 40M
n_envs: 12
vector env: SubprocVecEnv
device: NVIDIA A100
checkpoint: 4M 간격
final eval: 공통 paired 5000판
```

30M 학습이 seed당 약 3시간이었으므로 40M은 seed당 약 4시간, 세 seed는
약 12시간으로 예상한다. 평가와 저장 시간을 포함하면 12시간을 조금 넘을
수 있다.

실행 명령:

```bash
scripts/run_common_rule_40m_training.sh --dry-run
scripts/run_common_rule_40m_training.sh
```

스크립트는 세 seed를 순차 학습하고 각 최종 모델을 공통 paired 5000판으로
평가한다. 조기 종료는 사용하지 않는다. 40M 결과가 `58~60%`이면 저장된
후반 checkpoint를 공통 evaluator로 비교하고, 58% 미만이면 hybrid와
말 ID permutation에 강한 observation 설계를 다음 후보로 검토한다.

## 20. 40M 공통 상대 재학습 결과

40M 재학습은 정상 완료됐다. 세 seed 모두 같은 git commit
`52520ae8`에서 실행됐고, A100 GPU와 12개 subprocess env를 사용했다.

```text
opponent: common_rule_based
observation: tactical
reward: terminal
timesteps: seed별 40,009,728
checkpoint: 4M 간격 10개
evaluation: 공통 paired 5000판
seed source: range:100000:2500
seed sha256: ca2043aa...370a1
```

공통 paired 5000판 평가 결과는 다음과 같다.

| seed | wins | losses | overall | first | second | passed | 95% CI |
| ---: | ---: | ---: | ---: | ---: | ---: | :---: | --- |
| 0 | 2921 | 2079 | 0.5842 | 0.5900 | 0.5784 | false | 0.5705~0.5978 |
| 1 | 3023 | 1977 | 0.6046 | 0.6076 | 0.6016 | true | 0.5910~0.6181 |
| 2 | 3020 | 1980 | 0.6040 | 0.6132 | 0.5948 | true | 0.5904~0.6175 |

집계:

```text
3-seed pooled: 8964 / 15000 = 0.5976
seed 표준편차: 0.0095
선공 pooled: 0.6036
후공 pooled: 0.5916
pooled 95% CI: 0.5897~0.6054
통과 seed: 2 / 3
illegal actions: 0
evaluation errors: 0
```

기존 30M common 평가와 비교하면 개선 폭이 크다.

```text
30M common 평균: 0.5649
40M common 평균: 0.5976
개선 폭: +0.0327
평균 추가 승수: seed당 +163.3승 / 5000판
```

특히 후공 성능이 `0.5500 -> 0.5916`으로 올라간 것이 중요하다. 기존 모델은
후공에서 크게 약했는데, 공통 상대를 직접 학습하면서 이 약점이 상당히
줄었다.

## 21. 현재 결론

이번 결과로 공통 기준을 통과하는 pure PPO 후보를 확보했다.

```text
최고 후보: seed 1 final model
승률: 3023 / 5000 = 0.6046
모델 유형: Pure RL
```

seed 2도 `0.6040`으로 거의 같은 수준이다. 따라서 제출 후보만 놓고 보면
hybrid 없이도 목표를 달성했다.

다만 안정성 관점에서는 아직 과제가 있다.

- seed 0은 `0.5842`로 실패했다.
- 3-seed 평균은 `0.5976`으로 60%보다 `0.24%p` 낮다.
- pooled 95% CI는 `0.5897~0.6054`로 60%를 포함한다.
- seed 1과 seed 2의 통과 margin은 각각 23승, 20승으로 크지는 않다.

따라서 지금 결론은 “pure PPO 성공 후보 확보”이지, “충분한 margin의 최종
후보 확정”은 아니다.

## 22. 다음 액션

새로운 40M+ 장기 학습을 바로 더 돌리기보다는, 이미 저장된 checkpoint를
선별 평가하는 것이 우선이다.

권장 순서:

1. seed별 `32M`, `36M`, `40M` checkpoint를 공통 evaluator로 선별 평가한다.
2. selection seed에서는 1000~2000판으로 빠르게 비교한다.
3. 상위 후보만 공통 5000판으로 재검증한다.
4. 더 나은 checkpoint가 없으면 seed 1 final model을 pure PPO 제출 후보로 둔다.
5. checkpoint 선별 후에도 margin이 작으면 hybrid 또는 말 ID permutation에
   강한 observation 개선을 검토한다.

이 흐름은 이미 목표를 넘은 pure PPO 후보를 보존하면서, 추가 학습 없이
통과 margin을 넓힐 수 있는지 먼저 확인하는 전략이다.
