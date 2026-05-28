# M6 PPO 학습 세부 구현 및 검증 계획

작성일: 2026-05-27

## 1. 목적

M6의 목적은 `MaskablePPO`를 현재 윷놀이 Gymnasium 환경에 연결하고, action mask가 적용된 PPO 학습과 평가가 end-to-end로 동작하는지 검증하는 것이다.

이번 M6의 1차 범위는 **가장 단순하고 재현 가능한 baseline opponent 학습**으로 제한한다.

- 학습 알고리즘: `sb3-contrib`의 `MaskablePPO`
- 학습 환경: `YutnoriEnv`
- 기본 opponent: `RandomAgent`
- 평가 opponent: `RandomAgent`, `CaptureFirstAgent`, `GreedyFinishAgent`
- reward: terminal sparse reward 유지
  - learner win: `+1`
  - learner lose: `-1`
  - otherwise: `0`
- shaped reward, episode length limit, self-play snapshot pool은 M6 1차 구현에 포함하지 않는다.

## 2. 현재 결정 사항

### 2.1 PPO vs PPO가 아닌 baseline opponent부터 시작

초기 PPO 학습은 `PPO vs PPO`가 아니라 `PPO vs baseline opponent`로 진행한다.

이유:

- 현재 목표는 먼저 학습 파이프라인의 정확성을 검증하는 것이다.
- `PPO vs PPO` 또는 live self-play는 두 정책이 동시에 변해 학습 실패 원인 분석이 어려워진다.
- baseline opponent는 고정된 비교 기준을 제공하므로 학습 전후 승률 변화를 해석하기 쉽다.
- M5에서 baseline agent와 tournament runner가 이미 검증되었다.

### 2.2 기본 opponent

M6 1차 학습의 기본 opponent는 `RandomAgent`로 둔다.

평가는 다음 세 opponent에 대해 수행한다.

- `RandomAgent`
- `CaptureFirstAgent`
- `GreedyFinishAgent`

### 2.3 나중에 시도할 학습 방식

M6 1차 구현이 끝난 뒤 다음 학습 방식을 별도 실험으로 추가할 수 있다.

- frozen snapshot self-play
- live `PPO vs PPO`
- baseline curriculum
- mixed opponent pool
- opponent sampling probability 조정

이 방식들은 성능 비교 실험으로 남겨두며, M6 1차 구현에는 포함하지 않는다.

## 3. 작업 위치 전략

현재 MacBook Pro M4 Pro에서도 M6의 문서화, 코드 리뷰, 가벼운 smoke 검증은 충분히 가능하다.

다만 본격적인 PPO 구현과 학습은 Google A100 VM에서 이어서 진행하는 것을 권장한다.

핵심 이유:

- 윷놀이 env step은 Python rule engine 기반이므로 CPU 병렬성이 중요하다.
- A100 VM은 GPU뿐 아니라 24-core CPU 환경을 사용할 수 있어 장기 학습과 병렬 rollout 실험에 유리하다.
- 장시간 학습, 여러 seed 반복, hyperparameter 비교를 로컬 노트북보다 안정적으로 수행할 수 있다.
- 이후 C51 또는 self-play 확장에도 같은 VM 환경을 재사용할 수 있다.

따라서 이 MacBook에서는 다음까지만 진행한다.

1. M6 세부계획 문서화
2. 변경사항 커밋
3. GitHub push

그 다음 Google A100 VM의 새 세션에서 이 문서를 기준으로 구현을 이어간다.

## 4. Google A100 VM handoff 절차

VM에서 새 세션을 시작하면 다음 순서로 이어간다.

```bash
git clone <repo-url>
cd RL-project
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest
```

이미 clone된 repo라면 다음으로 시작한다.

```bash
git pull
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest
```

VM에서 `python -m pytest`가 통과해야 M6 구현을 시작한다.

## 5. M6 세부 단계

### M6.0 계획 문서화

구현 내용:

- `docs/MILESTONE_M6_PPO_PLAN.md` 작성
- M6 1차 학습 범위와 제외 범위 명시
- A100 VM handoff 기준 명시
- 추후 self-play 학습 방식 문서화

검증:

- 문서가 repo에 저장되어 새 세션에서 바로 확인 가능해야 한다.
- `docs/IMPLEMENTATION_PLAN.md`에서 이 문서를 참조할 수 있어야 한다.

### M6.1 VM 환경 검증

구현 내용:

- Python version 확인
- `torch`, `stable-baselines3`, `sb3-contrib`, `gymnasium` import 확인
- CUDA 사용 가능 여부 확인
- CPU core 수 확인
- 기본 pytest 실행

예상 확인 명령:

```bash
python - <<'PY'
import os
import torch
import gymnasium
import stable_baselines3
import sb3_contrib

print("cpu_count", os.cpu_count())
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_device_count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("cuda_device_name", torch.cuda.get_device_name(0))
print("gymnasium", gymnasium.__version__)
print("stable_baselines3", stable_baselines3.__version__)
print("sb3_contrib", sb3_contrib.__version__)
PY

python -m pytest
```

검증 기준:

- 필수 dependency import 성공
- A100 VM에서는 `torch.cuda.is_available()`가 `True`
- 전체 테스트 통과
- MacBook에서 이어서 작업하는 경우 CUDA는 필수가 아니며, `cpu` 또는 `mps` 환경으로 smoke 검증만 수행한다.

### M6.2 PPO 학습용 env factory

구현 내용:

- PPO 학습 스크립트와 평가 스크립트에서 공통으로 사용할 env factory 작성
- opponent 이름을 CLI 인자로 선택 가능하게 구성
- seed를 env, yut sampler, baseline opponent에 일관되게 전달
- vectorized env 생성 준비

필수 opponent option:

- `random`
- `capture_first`
- `greedy_finish`

예상 파일:

- `yutnori/training/__init__.py`
- `yutnori/training/env_factory.py`
- `tests/test_training_env_factory.py`

검증 로직:

- 각 opponent 이름으로 env 생성 가능 여부 확인
- 잘못된 opponent 이름은 `ValueError` 발생 확인
- 같은 seed로 `reset()`했을 때 observation과 action mask가 재현되는지 확인
- 생성된 env가 `action_masks()`를 제공하고 shape `(20,)`인 bool array를 반환하는지 확인

### M6.3 PPO 학습 스크립트

구현 내용:

- `scripts/train_ppo.py` 작성
- `MaskablePPO("MlpPolicy", env, ...)` 사용
- CLI에서 주요 설정을 받을 수 있게 구성
- 학습 결과를 run directory에 저장
- model, config, evaluation summary를 함께 저장

필수 CLI 옵션:

- `--total-timesteps`
- `--seed`
- `--opponent`
- `--n-envs`
- `--device`
- `--learning-rate`
- `--n-steps`
- `--batch-size`
- `--gamma`
- `--gae-lambda`
- `--ent-coef`
- `--run-dir`
- `--eval-episodes`

초기 기본값:

- `total_timesteps`: `10000`
- `opponent`: `random`
- `n_envs`: `1`
- `device`: `auto`
- `learning_rate`: `3e-4`
- `n_steps`: `2048`
- `batch_size`: `64`
- `gamma`: `0.99`
- `gae_lambda`: `0.95`
- `ent_coef`: `0.0`

A100 VM 장기 학습에서는 다음을 실험 후보로 둔다.

- `n_envs`: `8`, `16`, `24`
- `total_timesteps`: `500000`, `1000000`, `5000000`
- `device`: `cuda`

검증 로직:

- 아주 짧은 학습이 예외 없이 종료되는지 확인
- model zip 파일이 저장되는지 확인
- config JSON이 저장되는지 확인
- 학습 중 invalid action으로 env가 실패하지 않는지 확인
- `MaskablePPO.learn(..., use_masking=True)` 경로를 사용했는지 확인

### M6.4 PPO 평가 스크립트

구현 내용:

- `scripts/evaluate_ppo.py` 작성
- 저장된 PPO model 로드
- 매 decision마다 current action mask를 전달해 `model.predict()` 호출
- baseline opponent별 승률과 평균 metric 저장
- illegal action 발생 수를 명시적으로 집계

필수 CLI 옵션:

- `--model-path`
- `--episodes`
- `--seed`
- `--opponent`
- `--device`
- `--output`

검증 로직:

- 저장된 model을 로드할 수 있어야 한다.
- `RandomAgent`, `CaptureFirstAgent`, `GreedyFinishAgent` 상대 평가가 각각 실행되어야 한다.
- 모든 PPO action이 현재 mask에서 valid여야 한다.
- illegal action count가 `0`이어야 한다.
- 승률, 평균 turn 수, 평균 decision 수가 출력 및 저장되어야 한다.

### M6.5 로컬 또는 VM smoke test

구현 내용:

- 짧은 PPO 학습 실행
- 저장 모델 로드
- 소규모 평가 실행
- 전체 pytest 재실행

예상 smoke 명령:

```bash
python scripts/train_ppo.py \
  --total-timesteps 2000 \
  --seed 0 \
  --opponent random \
  --n-envs 1 \
  --device auto \
  --run-dir runs/ppo_smoke

python scripts/evaluate_ppo.py \
  --model-path runs/ppo_smoke/model.zip \
  --episodes 20 \
  --seed 1 \
  --opponent random \
  --device auto \
  --output runs/ppo_smoke/eval_random.json

python -m pytest
```

검증 기준:

- 학습 스크립트 종료
- 모델 저장
- 평가 스크립트 종료
- illegal action count `0`
- 전체 pytest 통과

이 smoke test는 성능 검증이 아니다. 목적은 학습 파이프라인이 올바르게 연결되었는지 확인하는 것이다.

### M6.6 A100 장기 학습 준비

구현 내용:

- 장기 학습용 run directory 규칙 정리
- seed별 실행 command 정리
- baseline별 평가 command 정리
- 산출물 저장 구조 정리
- tmux 기반 실행 절차 정리
- tqdm 기반 진행률/ETA/episode 통계 표시 정리
- 학습 중 episode 통계 기록 및 eval 로그 연결 정리
- 학습 전후/정식 평가 episode 진행률 표시 정리
- 장기 학습 중 중간 모델 보존을 위한 checkpoint 옵션 정리
- mask-aware 평가 기반 선택적 early stopping 정리

예상 파일:

- `docs/MILESTONE_M6_LONG_TRAINING_GUIDE.md`
- `scripts/train_ppo.py`

예상 디렉터리 구조:

```text
runs/
  ppo/
    random_seed_0/
      config.json
      model.zip
      eval_random.json
      eval_capture_first.json
      eval_greedy_finish.json
      checkpoints/
      tensorboard/
```

장기 학습 예시:

```bash
python scripts/train_ppo.py \
  --total-timesteps 1000000 \
  --seed 0 \
  --opponent random \
  --n-envs 16 \
  --device cuda \
  --checkpoint-freq 100000 \
  --run-dir runs/ppo/random_seed_0
```

검증 기준:

- 동일 command를 seed만 바꿔 반복 실행할 수 있어야 한다.
- run directory가 덮어쓰기 위험 없이 생성되어야 한다.
- 짧은 checkpoint smoke에서 checkpoint zip이 생성되어야 한다.
- 짧은 progress smoke에서 tqdm 진행률과 episode 통계 postfix가 표시되어야 한다.
- 짧은 eval progress smoke에서 평가 episode 진행률이 표시되어야 한다.
- 짧은 early stopping smoke에서 목표 timestep 전에 학습이 중단되어야 한다.
- 짧은 episode stats smoke에서 완료 episode 통계가 기록되고 summary에 집계되어야 한다.
- 평가 결과가 JSON 또는 CSV로 남아야 한다.
- 장기 학습 시작/모니터링/평가 방법이 문서화되어야 한다.

## 6. PPO 구현 시 주의 사항

### 6.1 action mask

`MaskablePPO` 학습과 평가에서는 invalid action mask가 핵심이다.

반드시 지켜야 할 점:

- env는 `action_masks()`를 직접 제공해야 한다.
- mask shape는 `(20,)`이어야 한다.
- valid action은 `True`, invalid action은 `False`여야 한다.
- 평가 시에도 `model.predict(obs, action_masks=mask)` 형태로 mask를 전달해야 한다.
- 일반 SB3 `EvalCallback` 또는 `evaluate_policy`가 아니라 mask-aware 평가 방식을 사용해야 한다.

### 6.2 vectorized env

`n_envs > 1`을 사용할 때도 각 env가 직접 `action_masks()`를 제공해야 한다.

초기 smoke는 `n_envs=1`로 시작한다. VM에서 성능 확인 후 `n_envs=8`, `16`, `24`를 비교한다.

### 6.3 device 선택

MacBook:

- 기본은 `device=auto` 또는 `device=cpu`
- 작은 MLP와 Python env step 중심이라 GPU/MPS 이점은 제한적일 수 있다.

Google A100 VM:

- 장기 학습은 `device=cuda`로 실행한다.
- 하지만 병목이 env step이면 GPU 사용률이 낮을 수 있으므로 `n_envs`와 CPU 사용률을 함께 본다.

### 6.4 episode 제한

현재 프로젝트 결정에 따라 episode length limit은 기본 도입하지 않는다.

다만 PPO 학습 중 게임이 비정상적으로 길어지는 문제가 관찰되면 다음 정보를 보고한 뒤 사용자 확인을 받는다.

- 문제가 발생한 seed
- 평균 decision 수
- 최대 decision 수
- 영향을 받는 테스트와 학습 결과
- `max_episode_steps` 도입 대안

승인 없이는 episode 제한을 학습 환경 기본값으로 추가하지 않는다.

### 6.5 reward 변경

M6 1차 구현에서는 shaped reward를 추가하지 않는다.

reward 변경이 필요해 보이면 다음을 먼저 보고한다.

- 변경 이유
- 예상 장점
- 정책 왜곡 가능성
- 비교 실험 설계
- 기존 terminal sparse reward와의 성능 비교 방법

승인 없이는 reward를 변경하지 않는다.

## 7. 실험 기록 규칙

각 PPO run은 다음 정보를 남긴다.

- git commit hash
- 실행 command
- seed
- opponent
- total timesteps
- hyperparameter
- device
- CPU/GPU 정보
- 학습 시작/종료 시각
- 저장 model path
- checkpoint directory
- 평가 결과

평가 결과에는 최소한 다음 metric을 포함한다.

- opponent name
- episodes
- wins
- losses
- win rate
- average turns
- average decisions
- illegal action count

M8에서 잡기, 업기, 지름길, 골인 빈도 분석을 확장할 예정이므로 M6에서는 기본 승률과 진행 metric 위주로 저장한다.

## 8. 완료 기준

M6 1차 구현은 다음을 모두 만족해야 완료로 본다.

- `scripts/train_ppo.py` 구현
- `scripts/evaluate_ppo.py` 구현
- PPO 학습 smoke test 완료
- 저장된 PPO 모델 로드 성공
- mask-aware evaluation에서 illegal action `0`
- 학습 전후 `RandomAgent` 상대 승률 저장
- `RandomAgent`, `CaptureFirstAgent`, `GreedyFinishAgent` 상대 평가 실행 가능
- `python -m pytest` 통과
- `docs/MILESTONE_M6_REPORT.md`에 상세 검증 결과 저장
- `docs/MILESTONE_M6_LONG_TRAINING_GUIDE.md`에 장기 학습 실행 절차 저장

## 9. 추후 확장 후보

### 9.1 frozen snapshot self-play

일정 timesteps마다 PPO policy snapshot을 저장하고, opponent pool에서 snapshot을 샘플링한다.

장점:

- 상대가 점진적으로 강해진다.
- live self-play보다 학습 분포가 덜 흔들린다.

주의점:

- opponent pool 관리가 필요하다.
- snapshot 저장/로드 비용이 생긴다.
- 평가 기준을 baseline과 분리해 관리해야 한다.

### 9.2 live PPO vs PPO

두 PPO policy를 동시에 업데이트한다.

장점:

- self-play에 가장 직접적으로 가깝다.

주의점:

- 두 정책이 동시에 변해 불안정성이 커진다.
- 승률 변화만으로 개선 여부를 판단하기 어렵다.
- 구현과 디버깅 난이도가 높다.

### 9.3 curriculum opponent

초기에는 `RandomAgent`, 이후 `CaptureFirstAgent`, `GreedyFinishAgent`, snapshot opponent를 섞는다.

장점:

- 쉬운 상대부터 어려운 상대로 학습 난도를 올릴 수 있다.

주의점:

- curriculum schedule이 성능에 큰 영향을 줄 수 있다.
- 비교 실험 설계가 필요하다.

## 10. 참고 문서

- MaskablePPO 공식 문서: https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html
- Stable-Baselines3 vectorized env 문서: https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html
- Gymnasium Env API: https://gymnasium.farama.org/api/env/
- 현재 프로젝트 전체 구현 계획: `docs/IMPLEMENTATION_PLAN.md`
- M5 baseline 검증 보고서: `docs/MILESTONE_M5_REPORT.md`
