# 전체 뒷도 규칙 및 50M 재학습 상세 구현 계획

작성일: 2026-06-10

## 1. 목표

현재 뒷도를 제외한 윷놀이 환경을 다음 규칙으로 확장한다.

- `BACK_DO`를 윷 결과와 observation에 추가한다.
- `WAITING`과 `FINISHED`를 제외한 모든 보드 위 말은 뒷도를 사용할 수 있다.
- 뒷도는 현재 말이 따르는 경로를 기준으로 한 칸 역방향 이동한다.
- 첫 칸 `O1`에서 뒷도를 사용하면 골인 직전의 `HOME` 칸으로 이동한다.
- `HOME`에서도 뒷도를 사용할 수 있다.
- 뒷도로 상대 말 또는 stack이 있는 칸에 도착하면 일반 이동과 동일하게 잡는다.
- 뒷도로 자기 말이 있는 칸에 도착하면 일반 이동과 동일하게 업는다.
- 뒷도로 실제 잡기가 발생하면 보너스 던지기 1회를 얻는다.
- 변경된 action/observation 공간으로 MaskablePPO를 처음부터 50M timesteps
  재학습한다.

이번 변경은 기존 20-action 환경을 24-action 환경으로 바꾸므로 기존 40M
PPO checkpoint를 이어서 학습하지 않는다.

## 2. 확정할 이동 의미

### 2.1 기본 이동

- `BACK_DO`의 이동량은 `-1`이다.
- 양수 결과는 기존과 같이 `DO=1`, `GAE=2`, `GEOL=3`, `YUT=4`,
  `MO=5`다.
- `WAITING` 말은 뒷도로 진입할 수 없다.
- `FINISHED` 말은 기존과 같이 어떤 결과로도 이동할 수 없다.
- 뒷도는 윷/모와 같은 자동 보너스 결과가 아니다.

### 2.2 첫 칸과 HOME

- `O1 --BACK_DO--> HOME`
- 이 이동의 `HOME`은 `ON_BOARD` 상태인 골인 직전 칸이다.
- `HOME --양수 이동--> FINISHED`
- `HOME --BACK_DO--> 해당 말의 귀환 경로상 직전 칸`
  - 외곽 또는 C1-C3 경로로 온 말: `O19`
  - 중앙-B3 경로로 온 말: `B4`
- `O1`에서 특수 이동으로 `HOME`에 도착한 말은 외곽 경로 문맥을 사용하므로
  다시 뒷도를 사용하면 `O19`로 이동한다.

### 2.3 분기점 역방향 규칙

뒷도는 물리적으로 인접한 임의 칸을 선택하는 action이 아니다. 말이 현재
따르는 논리 경로의 직전 칸으로 이동한다.

| 현재 논리 위치 | 뒷도 목적지 |
| --- | --- |
| 외곽 `C1` | `O4` |
| C1 지름길 `A1` | `C1` |
| C1 경유 `CENTER` | `A2` |
| C2 경유 `CENTER` | `B2` |
| 중앙 귀환 경로 `B3` | `CENTER` |
| C1-C3 경로 `C3` | `A4` |
| 외곽 경로 `C3` | `O14` |
| 외곽/C1-C3 경로 `HOME` | `O19` |
| C1/C2 중앙 귀환 경로 `HOME` | `B4` |

뒷도로 분기점에 도착했을 때는 새로운 지름길을 선택하지 않고 현재 논리
경로를 유지한다. 양수 이동으로 분기점에 정확히 도착했을 때만 기존 자동
지름길 규칙을 적용한다.

### 2.4 stack의 경로 문맥

- 같은 physical cell의 자기 말은 논리 경로가 달라도 stack으로 합쳐진다.
- 합쳐진 모든 말은 도착한 이동 말/stack의 논리 위치와 경로 문맥을 따른다.
- 이후 뒷도 역시 합쳐진 stack 전체가 해당 경로를 따라 이동한다.
- 상대 stack을 뒷도로 잡으면 stack 전체를 `WAITING`으로 되돌린다.

## 3. 윷 결과와 확률

기존 forward action의 project-RF mapping 순서를 유지하기 위해 `BACK_DO`를
마지막에 추가한다.

```text
DO, GAE, GEOL, YUT, MO, BACK_DO
```

제안 확률은 기존 도 확률 `0.1536`을 뒷도와 일반 도로 분리한다.

| 결과 | 이동량 | 확률 |
| --- | ---: | ---: |
| BACK_DO | -1 | 0.0384 |
| DO | +1 | 0.1152 |
| GAE | +2 | 0.3456 |
| GEOL | +3 | 0.3456 |
| YUT | +4 | 0.1296 |
| MO | +5 | 0.0256 |

합계는 `1.0`이어야 한다. `BONUS_RESULTS`는 계속 `YUT`, `MO`만 포함한다.

변경 파일:

- `yutnori/core/yut.py`
- `yutnori/core/__init__.py`
- `tests/test_yut.py`

## 4. 보드 위치 모델 재설계

### 4.1 현재 구조의 문제

현재 `Position`은 `route`, `index`, `physical_cell`을 저장하지만 정확히
분기점에 도착하면 outgoing route로 교체한다. 이 때문에 다음 정보가
사라진다.

- `CENTER`에 C1 방향으로 왔는지 C2 방향으로 왔는지
- `HOME`에 외곽/O19 방향으로 왔는지 중앙/B4 방향으로 왔는지

양수 이동만 있을 때는 큰 문제가 아니지만 모든 칸에서 뒷도를 허용하면
같은 physical cell에서 서로 다른 역방향 목적지를 구분할 수 없다.

### 4.2 논리 track 보존

구현에서는 기존 route 조각을 유지하면서 다음 문맥을 추가로 보존한다.

```text
route: OUTER | C1_DIAGONAL | C2_DIAGONAL | CENTER_TO_HOME
entry_route: C1_DIAGONAL | C2_DIAGONAL | None
```

`Position`은 다음 불변식을 가진다.

- `ON_BOARD`이면 `route`, `index`, `physical_cell`이 모두 존재한다.
- `physical_cell == ROUTES[route][index]`다.
- `CENTER_TO_HOME`이면 C1/C2를 구분하는 `entry_route`가 존재한다.
- 보드 위 `HOME`은 해당 route의 마지막 index와 entry 문맥을 유지한다.
- `WAITING`과 `FINISHED`는 route/index/physical_cell을 갖지 않는다.

이 조합으로 observation의 logical track ID를 계산한다.

```text
OUTER
C1_VIA_C3
C1_VIA_CENTER_HOME
C2_VIA_CENTER_HOME
```

역방향으로 C1/C2 이전 외곽 구간까지 돌아오면 route를 `OUTER`로 정규화한다.

### 4.3 Board API

`Board.move(position, steps)`는 다음 값만 허용한다.

- `steps == -1`
- `steps in 1..5`

`0`과 `-2` 이하는 오류로 처리한다. 내부 구현은 다음처럼 분리한다.

```text
move()
  -> _move_backward_one()
  -> _move_forward()
  -> _switch_forward_track_on_exact_landing()
  -> _normalize_backward_track()
```

`MoveResult`에는 기존 필드를 유지하고 분석용 `moved_backward`를 추가한다.

변경 파일:

- `yutnori/core/board.py`
- `tests/test_board.py`
- 위치를 직접 생성하는 기존 테스트와 adapter 테스트

## 5. GameState와 action 공간

### 5.1 action 인코딩

action은 기존 형식을 유지하되 결과 수를 6개로 늘린다.

```text
action = piece_id * 6 + yut_type_id
ACTION_SIZE = 4 * 6 = 24
```

`encode_action`과 `decode_action`은 enum 이름 기반 테스트로 검증한다.
숫자로 하드코딩된 기존 20-action 테스트와 문서는 모두 수정한다.

### 5.2 뒷도 legal action

`BACK_DO` action이 legal이려면 다음 조건을 모두 만족해야 한다.

- 현재 pool에 `BACK_DO`가 한 개 이상 있다.
- 선택한 말이 `ON_BOARD`다.
- 선택한 말이 stack 대표 말이다.
- 게임이 종료되지 않았고 현재 player의 action이다.

모든 `ON_BOARD` 위치에는 역방향 목적지가 있으므로 별도의 칸별 금지는 두지
않는다.

### 5.3 뒷도 잡기와 업기

`apply_action`은 이동 방향과 무관하게 기존 destination 처리 순서를 사용한다.

1. 이동할 자기 stack을 결정한다.
2. `Board.move`로 목적지를 계산한다.
3. 이동 stack 전체를 목적지에 둔다.
4. 목적지의 상대 stack 전체를 잡아 `WAITING`으로 보낸다.
5. 목적지의 자기 말과 자동으로 합친다.
6. 실제 잡기가 발생했고 사용 결과가 `YUT/MO`가 아니면 보너스 던지기를
   수행한다.

따라서 `BACK_DO` 잡기도 일반 도/개/걸 잡기처럼 보너스 던지기 1회를 얻는다.

### 5.4 뒷도만 나와 움직일 말이 없는 턴

게임 시작 시 모든 말이 `WAITING`이므로 `BACK_DO`만 나온 player는 legal
action이 없다. 이를 env wrapper가 아니라 rule engine에서 처리한다.

새로운 turn-advance helper는 playable decision state가 나올 때까지 다음을
반복한다.

1. 현재 player가 윷을 던져 pool을 만든다.
2. legal action이 있으면 해당 player의 decision state를 반환한다.
3. legal action이 없으면 pool을 비우고 auto-pass event를 기록한다.
4. 상대 player로 바꾸고 새 턴을 시작한다.

무한 custom sampler를 검출하기 위한 높은 safety bound를 두되 일반 게임의
episode 길이 제한으로 사용하지 않는다.

auto-pass 기록에는 다음을 포함한다.

- player
- rolled results
- pool counts
- reason=`NO_LEGAL_ACTION`

`reset`, `apply_action`, opponent 자동 진행, 평가 runner가 모두 같은 helper를
사용해야 한다.

변경 파일:

- `yutnori/core/game.py`
- `yutnori/env/yutnori_env.py`
- `yutnori/training/common_evaluation.py`
- `yutnori/eval/tournament.py`
- `tests/test_game_state.py`

## 6. Observation 설계

### 6.1 base observation

pool에 `BACK_DO` count만 추가하면 분기점 경로 문맥이 가려져 같은
observation에서 뒷도 목적지가 달라지는 비-Markov 상태가 생긴다. 따라서 각
말의 logical track도 observation에 포함한다.

player별 구성:

| 항목 | 크기 |
| --- | ---: |
| physical position | 4 |
| status | 4 |
| logical track id | 4 |
| stack matrix | 16 |
| 합계 | 28 |

전체 base observation:

```text
내 상태 28 + 상대 상태 28 + pool counts 6 = 62
```

track id는 다음처럼 고정한다.

```text
0 = NONE (WAITING/FINISHED)
1 = OUTER
2 = C1_VIA_C3
3 = C1_VIA_CENTER_HOME
4 = C2_VIA_CENTER_HOME
```

### 6.2 tactical observation

action 수는 24개가 되고 기존 action별 feature 10개는 유지한다.

```text
TACTICAL_OBSERVATION_SIZE = 62 + 24 * 10 = 302
```

각 뒷도 action row도 양수 action과 동일하게 다음 결과를 계산한다.

- legal
- backward capture 여부와 잡는 말 수
- finish 여부
- 이동 stack 크기
- 이동 후 완주 거리
- rule-based score

`waiting_move`는 뒷도에서 항상 `0`이다. `finish`도 뒷도에서는 항상 `0`이지만
`O1 -> HOME`은 `distance_after=1`로 표현한다.

변경 파일:

- `yutnori/env/yutnori_env.py`
- `yutnori/env/__init__.py`
- `yutnori/agents/tactical_features.py`
- `tests/test_yutnori_env.py`
- `tests/test_tactical_action_features.py`
- `tests/test_training_env_factory.py`

## 7. Baseline과 평가 규칙

### 7.1 local rule-based agents

`evaluate_action`과 `project_rf_action_score`가 signed step을 사용하도록
수정한다.

- 뒷도 잡기는 기존 capture 점수를 받는다.
- `O1 -> HOME`은 완주까지 1칸 남은 상태로 평가한다.
- 일반적인 뒷도는 완주 거리를 늘리므로 잡기나 특별한 이득이 없으면 낮은
  점수를 받는다.
- CommonRuleBasedAgent의 tie-break 규칙은 유지한다.

### 7.2 기존 checkpoint 호환성

기존 PPO 모델:

- action logits 20개
- tactical observation 253개

새 모델:

- action logits 24개
- tactical observation 302개

shape가 모두 다르므로 기존 40M 모델은 load 후 이어 학습하거나 새 규칙에서
평가하지 않는다. 새 50M 학습은 fresh initialization을 사용한다.

project-RF checkpoint adapter도 source action이 20개이므로 `BACK_DO` output이
없다. 임의 logit을 만들어 공식 비교하지 않는다. 기존 adapter는 legacy
규칙용으로 명시하고, full-backdo evaluator에서 사용하면 명확한 호환성
오류를 내도록 한다. 향후 heuristic fallback을 추가할 경우 별도 hybrid
adapter 이름과 metadata를 사용한다.

### 7.3 ruleset와 protocol metadata

다음 식별자를 config와 평가 JSON에 기록한다.

```text
ruleset = full_backdo_v1
evaluation_protocol = common_rule_based_paired_full_backdo_v1
action_size = 24
observation_size = 62 or 302
```

평가 시 model의 `config.json`과 현재 env ruleset이 다르면 실행을 중단한다.
기존 결과 JSON과 best model 파일은 삭제하거나 덮어쓰지 않는다.

변경 파일:

- `yutnori/agents/baseline.py`
- `yutnori/agents/project_rf_checkpoint.py`
- `yutnori/training/model_config.py`
- `yutnori/training/common_evaluation.py`
- `scripts/evaluate_ppo.py`
- `scripts/evaluate_common_rule.py`
- 관련 adapter/evaluation 테스트

## 8. 테스트 계획

### 8.1 Yut 단위 테스트

- 6개 결과의 이동량 확인
- 확률 합 `1.0`
- `BACK_DO + DO == 0.1536`
- `BACK_DO`가 bonus 결과가 아님
- seeded sampler 재현성

### 8.2 Board 단위 테스트

- 일반 외곽 칸의 한 칸 후진
- `O1 -> HOME`
- 외곽 `HOME -> O19`
- 중앙 귀환 `HOME -> B4`
- 외곽 `C1 -> O4`
- `A1 -> C1`
- C1 경유 `CENTER -> A2`
- C2 경유 `CENTER -> B2`
- `B3 -> CENTER`
- C1-C3 경로 `C3 -> A4`
- 외곽 경로 `C3 -> O14`
- 분기 이전으로 돌아왔을 때 OUTER track 정규화
- HOME의 양수 이동은 기존처럼 FINISHED
- 모든 기존 양수 이동 회귀 테스트
- WAITING의 뒷도와 FINISHED 이동 오류
- `0`, `-2` 이동 오류

### 8.3 GameState 단위 테스트

- action encode/decode 24개 round trip
- pool에 BACK_DO가 있을 때 ON_BOARD 대표 말만 legal
- WAITING 말의 BACK_DO illegal
- stack 전체 후진
- 뒷도 도착점에서 자기 말과 업기
- 뒷도 단일 말 잡기
- 뒷도 상대 stack 잡기
- 뒷도 잡기 후 보너스 roll 1회
- 뒷도 HOME 잡기
- 뒷도 이동 후 남은 pool 유지
- BACK_DO만 나오고 모든 말이 WAITING이면 auto-pass
- 양 player가 연속 BACK_DO를 던지는 chained auto-pass
- auto-pass 이후 playable player와 pool이 정확함

### 8.4 Observation과 mask 테스트

- base shape `(62,)`
- tactical shape `(302,)`
- action mask shape `(24,)`
- pool BACK_DO count가 정해진 index에 인코딩됨
- 같은 CENTER physical cell이라도 C1/C2 track observation이 다름
- 같은 HOME physical cell이라도 O19/B4 track observation이 다름
- tactical legal column과 action mask 일치
- 뒷도 capture/stack/distance feature 정확성
- DummyVecEnv와 SubprocVecEnv shape 확인

### 8.5 통합 회귀 테스트

- mask-aware random rollout 최소 100판
- Random vs Random 최소 1000판
- CommonRuleBasedAgent 대진 smoke
- 모든 게임 illegal action 0건
- max decision bound 초과 0건
- fixed seed에서 결과 재현
- paired evaluator의 선공/후공 게임 수 일치

## 9. 관측 로그와 분석 지표

기존 event JSON에 `BACK_DO`가 자연스럽게 기록되도록 하고 다음 집계치를
summary/evaluation에 추가한다.

- `back_do_roll_count`
- `back_do_action_count`
- `back_do_capture_count`
- `back_do_captured_piece_count`
- `back_do_home_entry_count` (`O1 -> HOME`)
- `back_do_from_home_count`
- `no_legal_action_auto_pass_count`
- player별/선공별 back-do 사용률

이 지표는 새 규칙이 실제 학습 episode에 충분히 나타났는지와 rule agent가
뒷도를 어떻게 사용하는지 확인하는 데 사용한다.

## 10. 50M 학습 실행 계획

### 10.1 전용 실행 스크립트

`scripts/run_common_rule_50m_backdo_training.sh`를 추가한다.

기본 설정:

```text
ruleset: full_backdo_v1
opponent: common_rule_based
observation: tactical
reward: terminal
seeds: 0, 1, 2
total_timesteps: 50,000,000 per seed
n_envs: 32
vec_env: subproc
device: cuda
learning_rate: 3e-4
n_steps: 2048
batch_size: 2048
gamma: 0.99
gae_lambda: 0.95
ent_coef: 0.0
checkpoint_freq: 5,000,000
```

출력 경로:

```text
runs/ppo_common_rule_50m_backdo_subproc/
logs/ppo_common_rule_50m_backdo_subproc/
```

run name 예시:

```text
common_rule_based_seed_0_50m_nenv32_tactical
```

ruleset 구분은 전용 runs root와 `config.json`의 `ruleset=full_backdo_v1`로
보장한다.

rollout 크기가 `2048 * 32 = 65,536`이므로 목표 50M을 넘겨 완료되는 실제
학습량은 seed별 `50,003,968` timesteps다.

### 10.2 실행 단계

1. 전체 `pytest` 통과
2. board transition table 단위 테스트 통과
3. random/common agent 1000판 안정성 검증
4. 100K CPU 또는 GPU PPO smoke
5. seed 0의 1M throughput/episode 통계 확인
6. 50M seed 0, 1, 2 순차 fresh training
7. 각 final model paired 5000판 평가
8. 필요하면 별도 selection seed로 40M/45M/50M checkpoint 비교
9. 선택된 checkpoint를 holdout paired seed로 최종 확인

### 10.3 중단 기준

다음 조건이면 50M 본 학습을 시작하지 않는다.

- random rollout에서 illegal action 발생
- CENTER/HOME track observation 충돌
- auto-pass loop 또는 sampler 고갈 이외의 reset 실패
- SubprocVecEnv observation/action shape 불일치
- 100K smoke model 저장/재로드 실패
- common evaluator에서 evaluation error 발생

### 10.4 최종 평가

기본 final 평가는 기존과 같은 2500 base seed에 대해 모델 선공/후공을 한
번씩 실행해 총 5000판으로 구성한다. 단 protocol 이름과 ruleset hash는 새
규칙으로 분리한다.

필수 결과:

- 전체/선공/후공 승률
- 95% Wilson interval
- illegal action과 evaluation error 0건
- 평균 turn/decision 수
- 뒷도 관련 집계치
- training seed별 결과와 3-seed 평균/표준편차

## 11. 문서 갱신

구현 완료 시 다음 문서를 함께 수정한다.

- `docs/RULES.md`: 뒷도, 역방향 분기표, 잡기/업기, 확률
- `docs/BOARD_DESIGN.md`: full track과 shared physical cell 설명
- `docs/IMPLEMENTATION_PLAN.md`: action/observation 크기와 M6 규칙
- `README.md`: 50M 실행/평가 command
- 새 결과 보고서: 구현 commit, 테스트 결과, 처리량, 3-seed 평가 결과

기존 40M 결과 문서의 수치는 수정하지 않고 `legacy_no_backdo` 결과임을
명시한다.

## 12. 구현 순서

실제 코드는 다음 순서로 변경한다.

1. `YutResult`, 확률, signed step
2. full-track Position/Board와 exhaustive board 테스트
3. GameState legal action, backward capture, auto-pass
4. baseline action evaluation
5. 24-action env와 62/302 observation
6. vector env와 evaluator
7. ruleset/config 호환성 검사
8. 전체 회귀 테스트와 rollout/tournament 검증
9. 50M 전용 실행 스크립트와 dry-run 테스트
10. smoke 학습 후 50M fresh training
