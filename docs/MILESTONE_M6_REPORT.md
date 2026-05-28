# M6 PPO 학습 구현 및 검증 보고서

작성일: 2026-05-27

## 1. 단계 요약

M6에서는 `sb3-contrib`의 `MaskablePPO`를 기존 `YutnoriEnv`에 연결했다. 1차 범위는 문서 계획대로 baseline opponent 학습 파이프라인으로 제한했다.

구현 대상:

- PPO 학습용 env factory
- MaskablePPO 학습 스크립트
- 저장 모델 평가 스크립트
- mask-aware 평가 루프
- M6 factory 테스트
- 로컬 PPO smoke 학습 및 평가 산출물

이번 smoke는 학습 성능 검증이 아니라, invalid action masking이 학습과 평가 전 구간에서 올바르게 연결되는지 확인하는 목적이다.

## 2. 변경 파일

- `yutnori/training/__init__.py`
  - training helper public export 추가
- `yutnori/training/env_factory.py`
  - baseline opponent 생성
  - 단일 `YutnoriEnv` 생성
  - `DummyVecEnv` 기반 vectorized env 생성
- `yutnori/training/ppo_evaluation.py`
  - 매 decision마다 `action_masks()`를 읽어 `model.predict(..., action_masks=mask)`로 평가
  - illegal action count 집계
- `scripts/train_ppo.py`
  - `MaskablePPO("MlpPolicy", env, ...)` 학습
  - `learn(..., use_masking=True)` 사용
  - `config.json`, `model.zip`, `summary.json`, 학습 전후 Random 평가 JSON 저장
  - 장기 학습용 checkpoint 저장 옵션 추가
- `scripts/evaluate_ppo.py`
  - 저장된 `MaskablePPO` 모델 로드
  - opponent별 mask-aware 평가 JSON 저장
- `tests/test_training_env_factory.py`
  - opponent option, seed 재현성, vector env action mask 검증
- `docs/MILESTONE_M6_REPORT.md`
  - 구현 및 검증 결과 기록
- `docs/MILESTONE_M6_LONG_TRAINING_GUIDE.md`
  - tmux 기반 장기 학습 시작 절차
  - run directory, log, checkpoint, 평가 command 규칙 정리
- `docs/MILESTONE_M6_PPO_PLAN.md`
  - M6.6 장기 학습 준비 항목과 완료 기준 보강
- `requirements.txt`
  - A100 VM의 NVIDIA driver 535/CUDA 12.2와 호환되도록 PyTorch `2.5.1+cu121` 고정
  - PyTorch CUDA 12.1 wheel index 추가

## 3. 구현 내용

### 3.1 Env Factory

지원 opponent:

- `random`
- `capture_first`
- `greedy_finish`

`make_yutnori_env()`는 opponent 이름과 seed를 받아 `YutnoriEnv`를 생성한다. `RandomAgent`에는 seed를 전달하고, deterministic baseline은 동일 인터페이스로 생성한다.

`make_yutnori_vec_env()`는 `DummyVecEnv`를 사용한다. 각 child env는 직접 `action_masks()`를 제공하므로 `sb3-contrib`의 mask utility와 호환된다.

### 3.2 PPO 학습 스크립트

필수 CLI 옵션을 구현했다.

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

추가 옵션:

- `--overwrite`: 기존 run directory가 비어 있지 않을 때 명시적으로 덮어쓰기
- `--tensorboard`: TensorBoard logging 사용
- `--verbose`: SB3 verbosity

초기 구현에서는 TensorBoard log path를 항상 넘겼으나, 현재 `requirements.txt`에 `tensorboard`가 없어서 학습 시작 전에 실패했다. M6 필수 검증은 TensorBoard가 아니라 PPO 학습/저장/평가이므로, TensorBoard는 `--tensorboard`를 지정할 때만 켜지도록 수정했다.

### 3.3 PPO 평가 스크립트

`scripts/evaluate_ppo.py`는 저장된 `model.zip`을 로드하고, 매 learner decision마다 현재 mask를 읽어 다음처럼 예측한다.

```python
model.predict(obs, deterministic=True, action_masks=mask)
```

예측된 action이 현재 mask에서 `False`이면 illegal action count를 증가시키고 즉시 실패시킨다. 정상 평가에서는 `illegal_action_count`가 `0`이어야 한다.

`--max-decisions`는 평가 harness의 안전장치다. env에 episode length limit을 추가하거나 reward를 바꾸지 않고, 비정상 장기 게임이 관찰되면 평가를 실패시키기만 한다.

### 3.4 장기 학습 준비

장기 학습은 사용자가 별도 `tmux` 세션에서 백그라운드로 실행하기로 했다. 이에 맞춰 다음을 준비했다.

- `scripts/train_ppo.py`에 `--checkpoint-freq` 추가
- `scripts/train_ppo.py`에 `--checkpoint-dir` 추가
- checkpoint 저장 경로와 save frequency를 `config.json`, `summary.json`에 기록
- `docs/MILESTONE_M6_LONG_TRAINING_GUIDE.md` 작성

`--checkpoint-freq`는 env timestep 기준이다. `n_envs > 1`에서는 SB3 callback 호출 횟수와 env timestep 수가 다르므로, 스크립트 내부에서 `checkpoint_freq / n_envs` 기준으로 callback save frequency를 계산한다.

장기 학습 run directory는 기본적으로 덮어쓰지 않는다. `--overwrite`를 명시하지 않으면 비어 있지 않은 run directory에서 실패하므로, 기존 실험 산출물을 실수로 덮어쓰는 일을 막는다.

## 4. 검증 환경

처음 시스템 Python을 확인했을 때 `python` 명령은 없었다.

```bash
python
```

결과:

- `zsh:1: command not found: python`

이 repo에는 `.venv`가 있으므로 이후 검증은 모두 다음 interpreter로 진행했다.

```bash
.venv/bin/python
```

환경 확인 결과:

- Python: `3.11.2`
- CPU core: `12`
- torch: `2.12.0+cu130`
- gymnasium: `1.2.3`
- stable-baselines3: `2.8.0`
- sb3-contrib: `2.8.0`
- CUDA available: `False`
- CUDA device count: `0`

CUDA 확인 중 PyTorch가 로컬 NVIDIA driver가 오래되어 CUDA를 초기화할 수 없다는 warning을 출력했다. 따라서 A100 VM 기준의 `torch.cuda.is_available() == True`는 이 로컬 환경에서 검증하지 않았다. 이번 검증은 `device=cpu` smoke로 진행했다.

### 4.1 A100 CUDA 환경 재검증

추가 확인일: 2026-05-27

사용자가 확인한 GPU 환경:

- NVIDIA-SMI: `535.261.03`
- Driver Version: `535.261.03`
- CUDA Version: `12.2`
- GPU: `NVIDIA A100-SXM4-40GB`

기존 `.venv`에는 `torch 2.12.0+cu130`이 설치되어 있었다. 이 조합은 CUDA 13.0 runtime wheel이므로 driver 535/CUDA 12.2 환경과 맞지 않았다.

수정:

```text
--extra-index-url https://download.pytorch.org/whl/cu121
torch==2.5.1+cu121
```

검증 명령:

```bash
.venv/bin/python -m pip install -r requirements.txt
```

결과:

- `torch-2.5.1+cu121` 설치 성공
- CUDA 12.1 runtime dependency 설치 성공

샌드박스 내부에서는 `/dev/nvidia*` 장치가 보이지 않아 `nvidia-smi`와 `torch.cuda.is_available()`가 실패했다. 승인된 샌드박스 외부 실행에서는 다음처럼 확인됐다.

```text
torch 2.5.1+cu121
torch_cuda_runtime 12.1
cuda_available True
cuda_device_count 1
cuda_device_name NVIDIA A100-SXM4-40GB
cuda_tensor_sum 1073741824.0
```

## 5. 검증 방식 및 결과

### 5.1 Training env factory 단위 테스트

명령:

```bash
.venv/bin/python -m pytest tests/test_training_env_factory.py -q
```

검증 내용:

- 잘못된 opponent 이름은 `ValueError` 발생
- `random`, `capture_first`, `greedy_finish`로 env 생성 가능
- reset observation이 observation space에 포함됨
- `action_masks()`가 bool array이고 shape이 `(20,)`
- 같은 seed로 생성한 env 두 개가 같은 observation, mask, initial rolls, opponent events를 반환
- `DummyVecEnv`에서 `get_action_masks(vec_env)`가 shape `(2, 20)` bool mask를 반환

결과:

- `6 passed in 2.55s`

### 5.2 전체 pytest 회귀 테스트

명령:

```bash
.venv/bin/python -m pytest -q
```

검증 내용:

- M2 board/yut 테스트
- M3 game state 테스트
- M4 env 테스트
- M5 baseline/tournament 테스트
- M6 training env factory 테스트

결과:

- `51 passed in 5.20s`
- 최종 재실행에서도 `51 passed in 5.14s`

### 5.3 PPO smoke 학습

처음 실행한 명령:

```bash
.venv/bin/python scripts/train_ppo.py \
  --total-timesteps 256 \
  --seed 0 \
  --opponent random \
  --n-envs 1 \
  --device cpu \
  --n-steps 64 \
  --batch-size 32 \
  --run-dir runs/ppo_smoke_m6_local \
  --eval-episodes 5
```

첫 결과:

- 실패
- 원인: `tensorboard_log`를 항상 설정해서 `tensorboard` 미설치 환경에서 SB3가 `ImportError` 발생
- 조치: `--tensorboard` 옵션을 추가하고 기본값에서는 TensorBoard logging을 끔

수정 후 실행한 명령:

```bash
.venv/bin/python scripts/train_ppo.py \
  --total-timesteps 256 \
  --seed 0 \
  --opponent random \
  --n-envs 1 \
  --device cpu \
  --n-steps 64 \
  --batch-size 32 \
  --run-dir runs/ppo_smoke_m6_local \
  --eval-episodes 5 \
  --overwrite
```

검증 내용:

- `MaskablePPO` 생성
- `learn(total_timesteps=256, use_masking=True)` 완료
- `model.zip` 저장
- `config.json` 저장
- `summary.json` 저장
- 학습 전 Random 상대 평가 저장
- 학습 후 Random 상대 평가 저장
- 평가 중 illegal action count 집계

결과:

- 학습 스크립트 정상 종료
- 모델 저장 경로: `runs/ppo_smoke_m6_local/model.zip`
- 학습 전 Random 평가:
  - episodes: `5`
  - wins: `2`
  - losses: `3`
  - win_rate: `0.4`
  - average_turns: `35.6`
  - average_decisions: `51.2`
  - illegal_action_count: `0`
- 학습 후 Random 평가:
  - episodes: `5`
  - wins: `2`
  - losses: `3`
  - win_rate: `0.4`
  - average_turns: `33.4`
  - average_decisions: `44.2`
  - illegal_action_count: `0`

판수가 5판이고 total timesteps도 256이므로 위 승률은 성능 개선 근거로 해석하지 않는다. 여기서는 학습 전후 평가 산출물이 저장되고 mask-aware evaluation이 illegal action 없이 동작했는지만 확인했다.

### 5.4 저장 모델 로드 및 opponent별 평가

RandomAgent 상대:

```bash
.venv/bin/python scripts/evaluate_ppo.py \
  --model-path runs/ppo_smoke_m6_local/model.zip \
  --episodes 5 \
  --seed 1 \
  --opponent random \
  --device cpu \
  --output runs/ppo_smoke_m6_local/eval_random.json
```

결과:

- wins: `1`
- losses: `4`
- win_rate: `0.2`
- average_turns: `28.8`
- average_decisions: `39.2`
- illegal_action_count: `0`

CaptureFirstAgent 상대:

```bash
.venv/bin/python scripts/evaluate_ppo.py \
  --model-path runs/ppo_smoke_m6_local/model.zip \
  --episodes 5 \
  --seed 2 \
  --opponent capture_first \
  --device cpu \
  --output runs/ppo_smoke_m6_local/eval_capture_first.json
```

결과:

- wins: `1`
- losses: `4`
- win_rate: `0.2`
- average_turns: `39.2`
- average_decisions: `53.6`
- illegal_action_count: `0`

GreedyFinishAgent 상대:

```bash
.venv/bin/python scripts/evaluate_ppo.py \
  --model-path runs/ppo_smoke_m6_local/model.zip \
  --episodes 5 \
  --seed 3 \
  --opponent greedy_finish \
  --device cpu \
  --output runs/ppo_smoke_m6_local/eval_greedy_finish.json
```

결과:

- wins: `2`
- losses: `3`
- win_rate: `0.4`
- average_turns: `38.4`
- average_decisions: `47.4`
- illegal_action_count: `0`

### 5.5 Compile 검증

명령:

```bash
.venv/bin/python -m compileall yutnori tests scripts
```

검증 내용:

- 새 training package
- 새 PPO scripts
- 기존 tests

결과:

- 컴파일 오류 없음

### 5.6 CUDA PPO smoke 재검증

명령:

```bash
.venv/bin/python scripts/train_ppo.py \
  --total-timesteps 128 \
  --seed 42 \
  --opponent random \
  --n-envs 1 \
  --device cuda \
  --n-steps 64 \
  --batch-size 32 \
  --run-dir runs/ppo_smoke_m6_cuda \
  --eval-episodes 3 \
  --overwrite
```

검증 내용:

- `device=cuda`로 `MaskablePPO` 생성
- `learn(..., use_masking=True)` 완료
- CUDA 환경에서 모델 저장
- 학습 전후 Random 상대 mask-aware 평가

결과:

- 학습 스크립트 정상 종료
- 모델 저장 경로: `runs/ppo_smoke_m6_cuda/model.zip`
- 학습 전 Random 평가:
  - episodes: `3`
  - wins: `3`
  - losses: `0`
  - win_rate: `1.0`
  - average_turns: `36.0`
  - average_decisions: `46.333333333333336`
  - illegal_action_count: `0`
- 학습 후 Random 평가:
  - episodes: `3`
  - wins: `3`
  - losses: `0`
  - win_rate: `1.0`
  - average_turns: `27.666666666666668`
  - average_decisions: `39.333333333333336`
  - illegal_action_count: `0`

저장 모델 CUDA 로드 및 평가 명령:

```bash
.venv/bin/python scripts/evaluate_ppo.py \
  --model-path runs/ppo_smoke_m6_cuda/model.zip \
  --episodes 3 \
  --seed 43 \
  --opponent random \
  --device cuda \
  --output runs/ppo_smoke_m6_cuda/eval_random_cuda.json
```

결과:

- wins: `2`
- losses: `1`
- win_rate: `0.6666666666666666`
- average_turns: `42.0`
- average_decisions: `50.333333333333336`
- illegal_action_count: `0`

### 5.7 장기 학습 checkpoint 준비 검증

추가 확인일: 2026-05-28

검증 목적:

- 장기 학습 중 최종 `model.zip` 저장 전에 중간 checkpoint가 생성되는지 확인한다.
- checkpoint 설정이 `config.json`과 `summary.json`에 남는지 확인한다.
- 기존 run directory 보호 로직이 유지되는지 확인한다.

검증 명령:

```bash
.venv/bin/python scripts/train_ppo.py \
  --total-timesteps 128 \
  --seed 7 \
  --opponent random \
  --n-envs 1 \
  --device cpu \
  --n-steps 64 \
  --batch-size 32 \
  --checkpoint-freq 64 \
  --run-dir runs/ppo_checkpoint_smoke \
  --eval-episodes 0 \
  --overwrite
```

검증 내용:

- `--checkpoint-freq 64`가 유효한 CLI option인지 확인
- `learn(..., callback=checkpoint_callback, use_masking=True)` 경로가 예외 없이 끝나는지 확인
- `runs/ppo_checkpoint_smoke/checkpoints/` 아래 checkpoint zip이 생성되는지 확인
- 최종 `model.zip`과 `summary.json`도 생성되는지 확인
- 같은 run directory를 `--overwrite` 없이 다시 사용하면 실패하는지 확인

결과:

- 학습 스크립트 정상 종료
- checkpoint zip 생성 확인
  - `runs/ppo_checkpoint_smoke/checkpoints/ppo_yutnori_64_steps.zip`
  - `runs/ppo_checkpoint_smoke/checkpoints/ppo_yutnori_128_steps.zip`
- 최종 `model.zip` 생성 확인
- `summary.json` 생성 확인
- 같은 run directory 재사용 시 `FileExistsError` 발생 확인

run directory 보호 검증 결과:

```text
FileExistsError: run directory is not empty: runs/ppo_checkpoint_smoke.
Use --overwrite or choose a new --run-dir.
```

이 검증은 checkpoint 저장 연결성 확인용이다. `total_timesteps=128`이므로 학습 성능을 판단하지 않는다.

### 5.8 장기 학습 가이드 검증

추가 확인일: 2026-05-28

검증 대상:

- `docs/MILESTONE_M6_LONG_TRAINING_GUIDE.md`

검증 내용:

- 장기 학습 전 환경 점검 command가 포함되어 있는지 확인
- `tmux` 세션 생성, detach, attach 방법이 포함되어 있는지 확인
- stdout/stderr log를 run directory 밖의 `logs/ppo/`에 저장하도록 안내하는지 확인
- seed별 run directory naming 규칙이 있는지 확인
- `--checkpoint-freq`를 포함한 장기 학습 command가 있는지 확인
- Random/CaptureFirst/GreedyFinish 평가 command가 모두 있는지 확인
- `illegal_action_count == 0` 확인 기준이 명시되어 있는지 확인
- 실패 대응 기준이 명시되어 있는지 확인

결과:

- 위 항목을 모두 문서에 반영했다.
- 장기 학습 자체는 사용자가 별도 `tmux` 세션에서 실행할 예정이므로, 이 보고서에서는 장기 실험 결과를 기록하지 않았다.

## 6. 산출물

로컬 smoke 산출물은 `runs/ppo_smoke_m6_local/`에 생성됐다. `runs/`는 `.gitignore` 대상이므로 실험 산출물은 git에 포함하지 않는다.

생성 파일:

- `config.json`
- `eval_before_random.json`
- `eval_after_random.json`
- `model.zip`
- `summary.json`
- `eval_random.json`
- `eval_capture_first.json`
- `eval_greedy_finish.json`

추가 CUDA smoke 산출물:

- `runs/ppo_smoke_m6_cuda/config.json`
- `runs/ppo_smoke_m6_cuda/eval_before_random.json`
- `runs/ppo_smoke_m6_cuda/eval_after_random.json`
- `runs/ppo_smoke_m6_cuda/model.zip`
- `runs/ppo_smoke_m6_cuda/summary.json`
- `runs/ppo_smoke_m6_cuda/eval_random_cuda.json`

추가 checkpoint smoke 산출물:

- `runs/ppo_checkpoint_smoke/config.json`
- `runs/ppo_checkpoint_smoke/model.zip`
- `runs/ppo_checkpoint_smoke/summary.json`
- `runs/ppo_checkpoint_smoke/checkpoints/<checkpoint>.zip`

## 7. 완료 기준 점검

- `scripts/train_ppo.py` 구현: 완료
- `scripts/evaluate_ppo.py` 구현: 완료
- PPO 학습 smoke test 완료: 완료
- 저장된 PPO 모델 로드 성공: 완료
- mask-aware evaluation에서 illegal action `0`: 완료
- 학습 전후 `RandomAgent` 상대 승률 저장: 완료
- `RandomAgent`, `CaptureFirstAgent`, `GreedyFinishAgent` 상대 평가 실행 가능: 완료
- `python -m pytest`에 해당하는 `.venv/bin/python -m pytest` 통과: 완료
- `docs/MILESTONE_M6_REPORT.md` 상세 검증 결과 저장: 완료
- `docs/MILESTONE_M6_LONG_TRAINING_GUIDE.md` 장기 학습 실행 절차 저장: 완료
- 장기 학습용 checkpoint 옵션 구현 및 smoke 검증: 완료

## 8. 보류 및 후속 확인

- A100 VM의 승인된 샌드박스 외부 실행에서는 CUDA 사용 가능을 확인했다. 다만 Codex 기본 샌드박스 내부에는 `/dev/nvidia*`가 보이지 않으므로 GPU 검증 명령은 승인 실행이 필요했다.
- 이번 smoke는 `total_timesteps=256`, `episodes=5`이므로 성능 비교 의미는 없다.
- 장기 학습 실행 절차와 checkpoint 준비는 완료했다. 실제 `n_envs=8`, `16`, `24` 및 `total_timesteps=500000+` 실험은 사용자가 별도 `tmux` 세션에서 실행한다.
- reward 변경, shaped reward, episode length limit, self-play snapshot pool은 추가하지 않았다.
