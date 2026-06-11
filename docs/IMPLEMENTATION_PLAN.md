# 윷놀이 RL 상세 구현 및 검증 계획

## Summary

실행은 단계별로 진행한다. 각 단계가 끝날 때마다 **변경 요약, 테스트 결과, 예상과 달랐던 점, 다음 단계 계획**을 보고하고, 다음 단계는 사용자 확인 후 진행한다.

이후 계획 변경이 필요하면 임의로 바꾸지 않고, 변경 사유와 선택지를 제시한 뒤 승인된 내용만 문서에 반영한다.

알고리즘은 **MaskablePPO**와 **C51**로 고정한다. 환경은 Gymnasium 호환으로 구현하되, action mask를 핵심 인터페이스로 둔다.

## Change Control

- 정보가 부족하거나 룰 해석이 애매하면 구현을 멈추고 사용자에게 확인한다.
- 특히 보드 좌표계, 참먹이 처리, 지름길 경로, reward 변경, episode 제한 추가는 확인 없이 결정하지 않는다.
- 계획 변경이 필요한 경우 다음 내용을 보고한 뒤 반영한다.
  - 원인
  - 영향받는 파일/테스트
  - 선택 가능한 대안
  - 추천안
- 각 마일스톤 완료 시 `pytest` 또는 지정 smoke test 결과를 기준으로 통과/보류를 판단한다.

## Implementation Stages

### M0. 문서 및 프로젝트 골격

- `docs/IMPLEMENTATION_PLAN.md` 저장.
- `requirements.txt`, 패키지 구조, 테스트 구조 생성.
- 검증: `python -m pytest`가 빈 테스트 또는 초기 smoke test 기준으로 실행 가능해야 한다.

### M1. 보드 좌표계 명세

- `BOARD_DESIGN.md` 초안 작성.
- 위키 기반 29칸 윷판을 physical cell과 logical route로 명세.
- 자동 지름길, 중앙 공유 칸, 참먹이 정확 도착/통과 예시 포함.
- 검증: 구현 전 사용자 확인 필수. 확인 전 `Board` 구현 금지.

### M2. 윷 샘플러와 Board 구현

- 윷 확률: 도 15.36%, 개 34.56%, 걸 34.56%, 윷 12.96%, 모 2.56%.
- `Board.move(position, steps)` 구현.
- 검증:
  - 대기 말 도/개/걸/윷/모 진입.
  - 분기점 정확 도착 후 자동 지름길.
  - 분기점 통과 시 외곽 유지.
  - 참먹이 정확 도착은 미골인.
  - 참먹이 통과는 골인.
  - seeded sampler 재현성.

### M3. GameState 룰 엔진

- 말 4개, player 2명, stack, pool, 현재 player, winner 관리.
- 업기, 잡기, stack 전체 이동, stack 전체 잡힘 구현.
- 실제 잡기가 발생한 `DO/GAE/GEOL` 이동에만 잡기 보너스 적용.
- `YUT/MO` 잡기는 보너스 중복 없음.
- 검증:
  - 지나가는 칸에서 잡기/업기 미발생.
  - 도착 칸에서만 잡기/업기 발생.
  - 잡을 수 있었지만 잡지 않은 action에는 보너스 없음.
  - pool 소진 또는 legal action 없음이면 턴 전환.

### M4. Gymnasium 환경 Wrapper

- `action_space = Discrete(24)`.
- vector observation 반환.
- `action_masks()` 제공.
- learner action 이후 opponent 턴은 env 내부에서 자동 진행해 learner의 다음 decision state 반환.
- 검증:
  - `reset(seed)` 재현성.
  - action mask가 pool, finished piece, stack 대표 말 조건 반영.
  - mask-aware random rollout 100판 이상 종료.
  - Gymnasium 기본 checker는 invalid action mask를 무시할 수 있으므로 보조 검증으로만 사용.

### M5. Baseline Agent

- `RandomAgent`, `CaptureFirstAgent`, `GreedyFinishAgent` 구현.
- 검증:
  - baseline들이 illegal action을 선택하지 않음.
  - `Random vs Random` 1000판 종료.
  - 선공 분포가 대략 균등.
  - 승률, 평균 turn 수, 평균 decision step 수 출력.

### M6. PPO 학습

- `sb3-contrib`의 `MaskablePPO` 사용.
- `action_masks()`를 통해 invalid action masking 적용.
- M6 1차 구현은 baseline opponent 학습으로 시작한다.
- 세부 계획은 `docs/MILESTONE_M6_PPO_PLAN.md`를 따른다.
- frozen snapshot pool과 `PPO vs PPO`는 1차 구현 이후 별도 실험으로 남긴다.
- 검증:
  - 짧은 학습 smoke test 완료.
  - Maskable evaluation 루프에서 illegal action 0건.
  - 학습 전후 Random 상대 승률 저장.

### M7. C51 학습

- CleanRL C51 구조를 참고해 프로젝트 env에 맞게 학습 스크립트 작성.
- 출력 shape `(batch, 24, 51)`, support `[-1, 1]`.
- action 선택과 target action 계산 모두 mask 적용.
- 검증:
  - replay buffer, target network, categorical projection 단위 테스트.
  - 짧은 학습 smoke test 완료.
  - PPO와 동일한 opponent/evaluation 프로토콜 사용.

### M8. 최종 평가 및 분석

- 대진: PPO/C51 각각 vs Random, CaptureFirst, GreedyFinish, 그리고 PPO vs C51.
- 각 대진 1000판 이상, seed 5개 반복.
- 검증:
  - 승률 평균/표준편차 산출.
  - 평균 turn, decision step, 잡기, 업기, 지름길, 골인 빈도 산출.
  - 정책 분석 로그가 누락 없이 생성됨.

## APIs And Interfaces

### Gymnasium Env

- `reset(seed=None) -> (obs, info)`
- `step(action) -> (obs, reward, terminated, truncated, info)`
- `action_masks() -> np.ndarray[bool]` shape `(24,)`

### Action

- `action = piece_id * 6 + yut_type_id`
- `piece_id: 0..3`
- `yut_type_id: DO, GAE, GEOL, YUT, MO, BACK_DO`

### Reward

- learner win: `+1`
- learner lose: `-1`
- otherwise: `0`
- shaped reward는 기본 구현에 포함하지 않는다.

## Assumptions And References

- 단계별 승인 방식으로 진행한다.
- 보드 좌표계는 Codex가 초안을 작성하되, 구현 전 사용자 확인을 받는다.
- episode 길이 제한은 기본값으로 두지 않는다.
- 문제가 확인될 때만 `max_episode_steps` 추가를 제안한다.
- MaskablePPO는 `sb3-contrib` 공식 문서 기준으로 사용한다: https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html
- C51은 CleanRL 문서를 참고한다: https://docs.cleanrl.dev/rl-algorithms/c51/
- Gymnasium env API는 공식 문서 기준으로 맞춘다: https://gymnasium.farama.org/api/env/
