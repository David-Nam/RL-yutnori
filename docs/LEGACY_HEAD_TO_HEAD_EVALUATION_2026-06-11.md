# RL 40M PPO vs project-RF PPO 직접 대전 보고서

## 평가 대상

- Model A: `common_rule_based_seed_1_40m_nenv12_tactical/model.zip`
  - 유형: Pure PPO
  - 학습량: 40M timesteps
  - observation: legacy tactical 253차원
- Model B: `ppo_capture_imitation.pt`
  - 유형: RL + Rule Hybrid
  - project-RF compliant tactical prior weight: 2.5

## 평가 조건

- ruleset: `legacy_no_backdo_v1`
- action size: 20
- deterministic inference
- base seeds: `200000~202499`
- paired seeds: 2,500
- seed마다 Model A와 Model B의 선공을 교환
- 총 경기: 5,000판
- 미래 윷 결과와 평가 RNG 상태 참조 금지

윷 결과 확률:

| 결과 | 확률 |
| --- | ---: |
| 도 | 0.1536 |
| 개 | 0.3456 |
| 걸 | 0.3456 |
| 윷 | 0.1296 |
| 모 | 0.0256 |

## 결과

| Model | 승 | 승률 | 선공 | 후공 | Wilson 95% CI |
| --- | ---: | ---: | ---: | ---: | --- |
| RL 40M seed 1 | 2,434 | 48.68% | 49.84% | 47.52% | 47.30%~50.07% |
| project-RF PPO capture imitation | 2,566 | **51.32%** | 52.48% | 50.16% | 49.93%~52.70% |

승률 차이는 project-RF 기준 `+2.64%p`, 승수 차이는 `132승`이다.

Paired seed 결과:

| Pair 결과 | Seed 수 |
| --- | ---: |
| RL 40M 2승 | 582 |
| 1승 1패 | 1,270 |
| project-RF 2승 | 648 |

추가 지표:

| 지표 | RL 40M | project-RF |
| --- | ---: | ---: |
| 평균 잡기 수 | 2.2128 | 4.4872 |
| 평균 완주 말 수 | 2.7708 | 3.0944 |

전체 평균 turn 수는 `36.2224`, 평균 decision 수는 `48.7972`였다.
전체 선공 승률은 `51.16%`였다.

## 무결성

- completed games: 5,000 / 5,000
- illegal actions: 양쪽 모두 0
- evaluation errors: 0
- smoke와 최종 실행의 중복 seed 20개 결과 일치
- 40M PPO의 legacy adapter는 과거 pre-backdo evaluator 결과를 정확히 재현

## 해석

이 평가에서는 project-RF hybrid가 RL 40M seed 1보다 2.64%p 높은 승률을
기록했다. 다만 project-RF 승률의 Wilson 95% 신뢰구간이 50%를 근소하게
포함하고, paired-seed 기준 RL 승률 구간도 약 47.31%~50.05%이므로
통계적으로 확정적인 우위라고 단정할 수는 없다.

따라서 결론은 다음과 같다.

```text
project-RF PPO capture imitation이 근소 우세했지만,
5,000판 기준으로 두 모델의 실력이 명확히 다르다고 판정할 정도의
통계적 margin은 확보되지 않았다.
```

project-RF 모델은 Pure PPO가 아니라 inference-time tactical prior를 포함한
hybrid다. 평균 잡기 수가 RL 40M보다 약 두 배 높아, 직접 대전 우위는
capture 중심 tactical prior의 영향으로 해석할 수 있다.

## 결과 파일

```text
runs/legacy_head_to_head_40m_vs_project_rf/summary.json
runs/legacy_head_to_head_40m_vs_project_rf/games.csv
runs/legacy_head_to_head_40m_vs_project_rf/report.md
```
