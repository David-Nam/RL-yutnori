# 윷놀이 PPO 프로젝트 발표자료 구성안

## 1. 발표의 핵심 메시지

이번 발표는 다음 한 문장으로 요약할 수 있다.

> 윷놀이의 잡기·완주 정보를 action별 feature로 제공한 PPO를 고정된
> Rule-based Agent와 학습시킨 결과, 공통 5000판 평가에서 최고 60.46%의
> 승률로 목표를 달성했다.

발표에서는 단순히 최종 승률만 보여주기보다 다음 흐름이 드러나야 한다.

```text
문제 정의
→ 상대 Rule-based Agent 분석
→ PPO가 학습하기 어려운 지점 확인
→ state와 reward 후보 설계
→ 짧은 실험으로 후보 선별
→ 장기 학습
→ 공통 평가 기준 적용
→ 평가 상대에 맞춘 재학습
→ 60% 목표 달성
```

권장 발표 분량은 본문 11~13장, 부록 3~4장이다.

---

## Slide 1. 프로젝트 목표

### 제목

```text
Rule-based Agent를 이기는 윷놀이 PPO Agent
```

### 화면에 넣을 내용

- 2인 윷놀이 환경
- 고정 Rule-based Agent 상대
- 선공·후공을 모두 반영한 5000판 평가
- 목표 승률: 60% 이상

### 핵심 수치

```text
최종 최고 승률: 60.46%
결과: 목표 달성
모델 유형: Pure PPO
```

### 발표 설명

이 프로젝트의 목적은 단순히 윷놀이를 플레이하는 agent를 만드는 것이
아니다. 상대의 규칙을 분석하고, PPO의 state와 reward를 설계하며,
실험적으로 성능을 개선해 고정된 Rule-based Agent를 5000판 기준으로
이기는 것이 목표다.

---

## Slide 2. 게임 및 평가 환경

### 제목

```text
학습 환경과 최종 평가 조건
```

### 화면에 넣을 내용

```text
플레이어: 2명
말: 플레이어당 4개
윷 결과: 도, 개, 걸, 윷, 모
구현 규칙: 잡기, 업기, 지름길, 윷/모 추가 던지기
제외 규칙: 뒷도
```

### Action

```text
action = 사용할 말 × 보유한 윷 결과
action size = 4 × 5 = 20
```

윷 던지기는 agent의 action이 아니다. 윷 결과를 pool에 모은 뒤 어떤
결과를 어떤 말에 사용할지 선택한다.

### 최종 평가

```text
base seed 2500개
각 seed에서 모델 선공 1판 + 후공 1판
총 5000판
deterministic policy
60% 이상 통과
```

### 추천 그림

- 윷판 이미지 또는 경로도
- `현재 상태 → legal actions → 말/윷 선택 → 다음 상태` 흐름도

---

## Slide 3. Opponent는 어떻게 선정했는가

### 제목

```text
왜 Rule-based Agent를 학습 상대로 선택했는가
```

### 선정 배경

프로젝트의 공동 목표가 팀원이 구현한 고정 Rule-based Agent를 이기는
것이었기 때문에, 해당 로직을 우리 환경에 포팅해 학습과 평가 기준으로
사용했다.

### Rule-based Agent의 판단 기준

| 행동 요소 | 점수 |
| --- | ---: |
| 완주 | +100 |
| 상대 말 잡기 | +50 |
| 새 말 출발 | +5 |
| 업힌 말 이동 | +4 × 추가 말 수 |
| 완주까지 거리 | -0.5 × 거리 |

### 왜 강한 opponent인가

- 잡을 수 있으면 잡는 단기 전술이 강하다.
- 완주를 가장 높은 우선순위로 둔다.
- 업힌 말을 효율적으로 이동한다.
- 결정론적이어서 학습 전후 비교가 쉽다.
- 사람이 이해할 수 있어 PPO의 실패 행동을 분석하기 좋다.

### 발표 설명

Random Agent는 학습 파이프라인을 검증하기에는 적합하지만 최종 전략을
평가하기에는 너무 약하다. 반면 Rule-based Agent는 윷놀이의 핵심 전술을
명시적으로 사용하며 동작이 고정되어 있어, 목표 opponent이자 일관된
성능 기준으로 적합했다.

---

## Slide 4. 초기 PPO의 문제점

### 제목

```text
PPO는 무엇을 학습하기 어려웠는가
```

### 초기 상태 표현

기본 observation은 다음 정보만 제공했다.

- 내 말과 상대 말의 위치
- 말의 상태와 stack
- 현재 보유한 윷 결과

### 문제

PPO는 각 legal action에 대해 다음 내용을 신경망 내부에서 직접 추론해야
했다.

```text
이 행동으로 상대 말을 잡는가?
몇 개의 말을 잡는가?
완주할 수 있는가?
업힌 말이 함께 움직이는가?
이동 후 완주까지 얼마나 남는가?
```

Rule-based Agent는 이 정보를 코드로 즉시 계산하지만, PPO는 보드 위치와
action 관계를 많은 시행착오를 통해 배워야 했다.

### 핵심 가설

> PPO에게 정답 행동을 강제하지 않고, 각 행동의 전술적 의미를 observation으로
> 제공하면 더 빠르고 안정적으로 전략을 학습할 수 있다.

---

## Slide 5. 최종 모델 설계

### 제목

```text
MaskablePPO + Tactical Observation
```

### 모델

```text
Algorithm: MaskablePPO
Policy: MLP
Action mask: legal action만 선택
Observation: base state + action별 tactical feature
Reward: terminal reward
```

### Tactical Action Feature

각 20개 action에 다음 feature를 제공했다.

| Feature | 의미 |
| --- | --- |
| legal | 현재 선택 가능한 action인지 |
| capture | 잡기 여부 |
| captured_count | 잡는 상대 말 수 |
| finish | 완주 여부 |
| finished_count | 완주하는 내 말 수 |
| moved_count | 함께 이동하는 말 수 |
| waiting_move | 새 말을 출발시키는지 |
| stack_size | 이동 stack 크기 |
| distance_after | 이동 후 완주 거리 |
| rf_score | Rule-based 점수 |

### 중요한 구분

이 feature는 행동을 강제로 결정하는 rule이 아니다. PPO는 feature를
관찰한 뒤 최종 action을 스스로 선택한다. 따라서 최종 모델은 rule
override가 없는 pure PPO다.

### 추천 그림

```text
Board State
   ├─ Base Observation
   └─ 20 Actions × 10 Tactical Features
              ↓
         MaskablePPO
              ↓
        Legal Action 선택
```

---

## Slide 6. Reward 설계와 선택

### 제목

```text
잡기 보상을 더 주면 더 강해질까
```

### 비교한 Reward

#### Terminal Reward

```text
승리: +1
패배: -1
그 외: 0
```

#### RF-shaped Reward

```text
내 잡기: +
내 완주: +
지름길: +
상대 잡기: -
상대 완주: -
```

### 3M 후보 실험 결과

| Observation | Reward | 평균 승률 |
| --- | --- | ---: |
| base | terminal | 37.4% |
| base | rf_shaped | 32.4% |
| tactical | terminal | **53.3%** |
| tactical | rf_shaped | 52.3% |

### 결론

- 가장 큰 개선 요인은 reward shaping이 아니라 tactical observation이었다.
- 잡기 중심 reward는 잡기 성향을 강화했지만 최종 승률은 높이지 못했다.
- 완주와 장기 운영까지 포함한 목표에는 terminal reward가 더 잘 맞았다.

### 그래프 추천

네 후보의 평균 승률을 비교하는 막대그래프를 사용한다.  
`tactical + terminal` 막대만 강조한다.

---

## Slide 7. 학습 전략

### 제목

```text
짧은 후보 선별 후 장기 학습
```

### 단계별 학습

```text
3M: observation/reward 후보 선별
10M: tactical + terminal 가능성 확인
30M: 장기 학습 및 CPU 병렬화
40M: 공통 Rule-based Agent 상대 재학습
```

### 장기 학습 설정

```text
training seeds: 0, 1, 2
observation: tactical
reward: terminal
n_envs: 12
vector env: SubprocVecEnv
GPU: NVIDIA A100
최종 학습량: seed별 40M timesteps
```

### CPU/GPU 역할

- GPU: PPO policy/value network 학습
- CPU 12 core: 12개 환경 simulation 병렬 실행
- `SubprocVecEnv` 적용 후 처리량 약 41.8% 증가

### 발표 설명

처음부터 긴 학습을 실행하지 않고 3M 실험으로 후보를 줄였다. 이후
성능 상승 가능성이 확인된 하나의 조합에만 10M, 30M, 40M 계산 자원을
집중했다.

---

## Slide 8. 학습량에 따른 성능 변화

### 제목

```text
학습량을 늘리며 60%에 접근
```

### 기존 Rule-based 기준 결과

| 학습 단계 | 평균 승률 |
| --- | ---: |
| 3M | 53.3% |
| 10M | 57.95% |
| 30M | 59.51% |

### 그래프 추천

가로축을 학습 timesteps, 세로축을 평균 승률로 한 line chart:

```text
3M  → 10M  → 30M
53.3 → 57.95 → 59.51
```

60% 위치에 점선을 표시한다.

### 해석

- 학습량 증가에 따라 일관된 상승 추세가 있었다.
- 30M seed 0은 기존 평가에서 정확히 60%를 기록했다.
- 하지만 평균은 60%에 미달해 평가 조건을 다시 점검할 필요가 있었다.

---

## Slide 9. 공통 평가 적용과 성능 하락

### 제목

```text
평가 기준을 통일하자 기존 성능이 하락했다
```

### 변경된 평가 조건

- 선공·후공을 seed마다 정확히 pairing
- Rule-based 동점 행동은 가장 작은 action ID 선택

### 30M 결과 비교

```text
기존 평가 평균: 59.51%
공통 평가 평균: 56.49%
차이: -3.02%p
```

### 원인 분석

```text
seed pairing 변화: 약 -0.73%p
opponent tie-break 변화: 약 -2.29%p
```

### 핵심 발견

규칙상 말은 대칭이지만 PPO observation은 말 ID별 슬롯을 가진다.
Rule-based Agent가 동점에서 어떤 말 ID를 선택하는지가 상대 말의 배치
패턴을 바꾸고, PPO는 그 패턴에도 일부 적응했다.

### 발표 메시지

이 결과는 단순한 평가 실패가 아니라, 학습 상대와 평가 상대의 작은 정책
차이도 RL 모델의 성능에 영향을 줄 수 있다는 중요한 실험 결과다.

---

## Slide 10. 공통 상대 재학습 결과

### 제목

```text
평가 상대와 학습 상대를 일치시켜 목표 달성
```

### 40M 공통 평가 결과

| Seed | Wins | 전체 | 선공 | 후공 | 결과 |
| ---: | ---: | ---: | ---: | ---: | :---: |
| 0 | 2921 | 58.42% | 59.00% | 57.84% | 미달 |
| 1 | 3023 | **60.46%** | 60.76% | 60.16% | 통과 |
| 2 | 3020 | **60.40%** | 61.32% | 59.48% | 통과 |

### 개선

```text
30M common 평균: 56.49%
40M common 평균: 59.76%
개선: +3.27%p

후공 승률:
55.00% → 59.16%
개선: +4.16%p
```

### 핵심 결론

- 3개 seed 중 2개가 60% 기준을 통과했다.
- 최고 모델은 seed 1의 60.46%다.
- hybrid action override 없이 pure PPO로 목표를 달성했다.
- 공통 opponent를 직접 학습한 것이 특히 후공 성능을 개선했다.

### 그래프 추천

seed별 선공/후공 grouped bar chart를 사용한다.  
60% 기준선을 함께 표시한다.

---

## Slide 11. 팀원 project-RF 모델

### 제목

```text
팀원 모델은 어떤 방식으로 학습했는가
```

### ppo_capture_imitation

```text
Masked PPO network
+ StrategicValue teacher imitation
+ teacher score distillation
+ 중요 전술 상태 oversampling
+ inference-time tactical prior
```

oversampling 대상:

- 잡기 가능한 상태
- 잡힐 위험이 있는 상태
- 완주 가능한 상태
- 지름길 진입 가능한 상태

### 분류

policy network만 사용하는 pure PPO가 아니다. 평가 시 잡기, 완주, 위험
회피, 상대 반격 점수를 logit에 추가하므로 `RL + Rule Hybrid`다.

### 학습 결과

| Candidate | StrategicValue 상대 승률 |
| ---: | ---: |
| 0 | **25.2%** |
| 1 | 24.5% |
| 2 | 24.5% |
| 3 | 24.0% |

candidate 0이 최종 `ppo_capture_imitation.pt`로 선택됐다.

---

## Slide 12. 팀원 모델의 동일 환경 평가

### 제목

```text
서로 다른 모델을 같은 공통 환경에서 비교
```

### Adapter가 필요한 이유

```text
project-RF action:
yut_index × 4 + piece_id

RL-yutnori action:
piece_id × 5 + yut_index
```

두 프로젝트의 observation과 일부 지름길 route도 달라 state/action 변환
adapter를 구현했다.

### 공정성 문제와 수정

project-RF의 원본 tactical prior는 복제 환경의 실제 다음 윷 결과를 볼 수
있었다. 미래 정보 사용을 금지한 공통 평가 가이드에 맞게:

- 실제 평가 RNG 복사 제거
- 미래 윷 sampling 제거
- 고정된 윷 확률의 expected counterplay 사용

### 최종 5000판 결과

| Model | 유형 | 전체 | 선공 | 후공 | 결과 |
| --- | --- | ---: | ---: | ---: | :---: |
| ppo_capture_imitation | RL + Rule Hybrid | **59.46%** | 60.20% | 58.72% | 미달 |
| ppo_tactical | RL + Rule Hybrid | 55.40% | 57.40% | 53.40% | 미달 |

`ppo_capture_imitation`은 2973승으로 통과 기준 3000승에 27승 부족했다.

### 추가 분석

같은 checkpoint에서 tactical prior를 끈 network-only 100판 승률은
13%였다. 팀원 모델의 성능은 neural policy보다 tactical prior에 크게
의존했다.

---

## Slide 13. 최종 비교 및 결론

### 제목

```text
최종 결과와 배운 점
```

### 최종 순위

| Agent | 유형 | 공통 승률 |
| --- | --- | ---: |
| RL-yutnori seed 1 | Pure PPO | **60.46%** |
| RL-yutnori seed 2 | Pure PPO | 60.40% |
| project-RF ppo_capture_imitation | RL + Rule Hybrid | 59.46% |
| RL-yutnori seed 0 | Pure PPO | 58.42% |
| project-RF ppo_tactical | RL + Rule Hybrid | 55.40% |

### 핵심 결론

1. 윷놀이에서는 잡기·완주 같은 rule-based 지식이 매우 효율적이다.
2. PPO에는 reward shaping보다 action의 의미를 제공하는 state 설계가 더
   효과적이었다.
3. 학습 상대와 평가 상대의 정책을 일치시키는 것이 중요했다.
4. 충분한 학습량과 tactical observation으로 pure PPO도 Rule-based Agent를
   근소하게 넘어설 수 있었다.
5. 최고 모델은 공통 5000판에서 60.46%를 기록해 목표를 달성했다.

### 한계

- 통과 margin이 크지 않다.
- 3-seed 평균은 59.76%다.
- 뒷도를 포함한 전체 윷놀이 규칙은 구현하지 않았다.
- 팀원 모델 비교는 같은 평가 환경이지만 같은 학습 환경 비교는 아니다.

---

## Slide 14. 향후 개선 방향

### 제목

```text
다음에는 무엇을 개선할 수 있는가
```

### 단기 개선

- 저장된 32M, 36M, 40M checkpoint 비교
- 말 ID permutation에 강한 대칭 observation
- 여러 training seed 추가
- 평가 seed를 별도 비공개 목록으로 고정

### 규칙 확장

- 뒷도
- 더 정확한 실제 윷 확률 및 규칙
- 다양한 지름길 선택
- 상대 policy가 바뀌는 환경

### 모델 확장

- curriculum opponent
- self-play 및 snapshot pool
- 위험 회피를 위한 별도 value feature
- pure PPO와 rule-guided RL 비교

### 발표 설명

규칙이 단순할 때는 rule-based agent가 매우 효율적이다. 뒷도처럼 예외와
장기 판단이 증가하면 rule 작성 비용도 커지므로, 복잡한 환경에서는
rule-guided RL이나 self-play의 장점이 커질 수 있다.

---

## 발표용 그래프 데이터

### 그래프 1. 후보 설계 비교

| Candidate | Win Rate |
| --- | ---: |
| Base + Terminal | 0.374 |
| Base + RF-shaped | 0.324 |
| Tactical + Terminal | 0.533 |
| Tactical + RF-shaped | 0.523 |

### 그래프 2. 학습량별 성능

| Timesteps | Win Rate |
| ---: | ---: |
| 3M | 0.533 |
| 10M | 0.5795 |
| 30M | 0.5951 |

30M까지는 기존 Rule-based 평가 결과다. 공통 평가 적용 전후 결과가
섞이지 않도록 그래프 제목이나 주석에 반드시 표시한다.

### 그래프 3. 공통 평가 개선

| Experiment | Pooled Win Rate |
| --- | ---: |
| 30M, 기존 opponent로 학습 | 0.5649 |
| 40M, common opponent로 학습 | 0.5976 |

### 그래프 4. 최종 모델 비교

| Model | Win Rate |
| --- | ---: |
| Our PPO seed 1 | 0.6046 |
| Our PPO seed 2 | 0.6040 |
| project-RF capture imitation | 0.5946 |
| Our PPO seed 0 | 0.5842 |
| project-RF tactical | 0.5540 |

---

## 발표 시 주의할 표현

### 권장 표현

```text
공통 평가 기준을 통과하는 pure PPO checkpoint를 확보했다.
```

```text
3개 학습 seed 중 2개가 60% 기준을 통과했다.
```

```text
팀원 모델은 PPO network와 tactical prior를 결합한 hybrid 모델이다.
```

```text
rule-based 지식은 이번 단순화된 윷놀이에서 강한 inductive bias로 작동했다.
```

### 피해야 할 표현

```text
PPO가 항상 Rule-based Agent보다 강하다.
```

이유: seed 0과 3-seed 평균은 60% 미만이다.

```text
project-RF의 pure PPO가 59.46%를 기록했다.
```

이유: tactical prior를 포함한 hybrid 결과다.

```text
원본 project-RF agent가 68.50%를 기록했다.
```

이유: 실제 다음 윷 결과를 참조한 비공식 진단값이므로 최종 결과로 사용할
수 없다.

```text
reward shaping이 성능을 개선했다.
```

이유: 이번 실험에서 terminal reward가 더 좋은 최종 승률을 기록했다.

---

## 예상 질문과 답변

### Q1. 왜 Rule-based Agent를 opponent로 선택했는가?

프로젝트의 공식 목표 상대이며, 잡기와 완주 같은 윷놀이 핵심 전략을
명시적으로 사용한다. 또한 동작이 고정되어 있어 학습 전후 성능을
일관되게 비교할 수 있다.

### Q2. Tactical feature를 사용하면 pure RL이 아닌가?

feature는 현재 상태에서 계산 가능한 정보를 observation으로 제공할 뿐
action을 강제로 변경하지 않는다. 최종 행동은 PPO policy가 선택하므로
우리 모델은 pure PPO로 분류했다.

### Q3. 잡기 reward가 왜 성능을 높이지 못했는가?

잡기는 중요하지만 최종 목표는 게임 승리다. 과도한 중간 보상은 완주나
장기 운영보다 잡기 자체를 선호하게 만들 수 있다. terminal reward가
최종 승패와 가장 직접적으로 정렬됐다.

### Q4. 왜 3개 seed 평균은 60%가 아닌데 성공이라고 하는가?

프로젝트 판정 기준은 단일 최종 checkpoint의 5000판 승률 60%다. seed 1과
seed 2는 각각 기준을 통과했다. 다만 seed 안정성이 완전히 확보됐다고
주장하지 않고, 3-seed 평균이 59.76%라는 한계를 함께 보고한다.

### Q5. project-RF 모델의 59.46%가 순수한 PPO 성능인가?

아니다. policy network에 잡기·완주·위험 회피를 반영하는 tactical prior를
더한 hybrid 결과다. network-only smoke 성능이 낮아 prior 의존성이 크다.

### Q6. 뒷도를 추가하면 PPO가 더 유리해지는가?

초기에는 여전히 rule-based agent가 효율적일 가능성이 높다. 하지만 규칙과
예외가 많아질수록 hand-crafted rule의 복잡도가 증가하므로, 장기적으로는
rule-guided RL이나 self-play가 더 유리해질 가능성이 있다.

---

## 30초 결론

```text
저희는 Rule-based Agent의 행동 기준을 분석하고, PPO가 각 action의 잡기,
완주, 이동 결과를 쉽게 이해할 수 있도록 tactical observation을
설계했습니다. 짧은 후보 실험에서는 reward shaping보다 state 설계가 더
효과적이었고, common opponent를 상대로 seed별 40M을 재학습한 결과
최고 60.46%의 승률로 목표를 달성했습니다. 팀원 hybrid 모델도 동일한
평가 환경에서 비교했으며 59.46%를 기록했습니다. 이를 통해 이번
윷놀이 환경에서는 rule-based 지식이 매우 효율적이지만, 충분한 학습량과
적절한 observation을 사용하면 pure PPO도 이를 넘어설 수 있음을
확인했습니다.
```
