# M6 PPO 장기 학습 실행 가이드

작성일: 2026-05-28

## 1. 목적

이 문서는 M6 1차 PPO 구현을 바탕으로 A100 VM에서 장기 학습을 안전하게 시작하기 위한 실행 가이드다.

장기 학습 자체는 사용자가 별도 `tmux` 세션에서 백그라운드로 실행한다. 이 문서의 범위는 다음까지다.

- 실행 전 환경 점검
- run directory와 log 규칙
- seed별 학습 command
- tqdm 진행률과 ETA 확인 방법
- 학습 중 episode 통계 확인 방법
- checkpoint 저장 설정
- 선택적 early stopping 설정
- baseline opponent별 평가 command
- 모니터링과 실패 대응 기준

이번 단계에서도 reward 변경, episode length limit 추가, self-play snapshot pool은 하지 않는다.

## 2. 사전 점검

장기 학습을 시작하기 전에 repo root에서 다음을 확인한다.

```bash
git status --short
source .venv/bin/activate
python -m pip check
python -m pytest
nvidia-smi
```

CUDA/PyTorch 확인:

```bash
python - <<'PY'
import os
import torch
import gymnasium
import stable_baselines3
import sb3_contrib

print("cpu_count", os.cpu_count())
print("torch", torch.__version__)
print("torch_cuda_runtime", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("cuda_device_count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("cuda_device_name", torch.cuda.get_device_name(0))
    x = torch.ones((1024, 1024), device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    print("cuda_tensor_sum", float(y.sum().item()))
print("gymnasium", gymnasium.__version__)
print("stable_baselines3", stable_baselines3.__version__)
print("sb3_contrib", sb3_contrib.__version__)
PY
```

통과 기준:

- `python -m pip check`가 broken requirement를 보고하지 않는다.
- `python -m pytest`가 통과한다.
- `nvidia-smi`에서 A100이 보인다.
- `torch.cuda.is_available()`가 `True`다.
- CUDA tensor 연산이 예외 없이 끝난다.

Codex 기본 샌드박스에서는 `/dev/nvidia*` 장치가 보이지 않을 수 있다. 그런 경우 GPU 확인은 사용자 터미널 또는 승인된 실행 환경에서 확인한다.

## 3. Run Directory 규칙

장기 학습 산출물은 다음 규칙으로 저장한다.

```text
runs/
  ppo/
    random_seed_<seed>_<timesteps>_nenv<n_envs>/
      config.json
      model.zip
      summary.json
      eval_before_random.json
      eval_after_random.json
      eval_random_1000.json
      eval_capture_first_1000.json
      eval_greedy_finish_1000.json
      checkpoints/
```

예:

```text
runs/ppo/random_seed_0_1m_nenv16/
```

`scripts/train_ppo.py`는 run directory가 비어 있지 않으면 기본적으로 실패한다. 장기 학습에서는 `--overwrite`를 쓰지 않는다. 같은 설정을 다시 실행해야 하면 새 run directory를 만든다.

주의:

- `tee` 로그 파일을 run directory 안에 먼저 만들면 학습 스크립트가 "run directory is not empty"로 실패할 수 있다.
- 장기 학습 stdout/stderr 로그는 `logs/ppo/` 아래에 별도로 저장한다.

## 4. Tmux에서 학습 시작

첫 장기 run은 `500000` 또는 `1000000` timesteps로 시작한다. 여기서는 `1000000` timesteps 예시를 기본으로 둔다.

```bash
tmux new -s ppo_random_s0
```

tmux 세션 안에서 실행:

```bash
cd /home/david-nam/work-space/RL-yutnori
source .venv/bin/activate

RUN_NAME=random_seed_0_1m_nenv16
RUN_DIR=runs/ppo/${RUN_NAME}
LOG_DIR=logs/ppo
mkdir -p "${LOG_DIR}"

python -u scripts/train_ppo.py \
  --total-timesteps 1000000 \
  --seed 0 \
  --opponent random \
  --n-envs 16 \
  --device cuda \
  --learning-rate 3e-4 \
  --n-steps 2048 \
  --batch-size 2048 \
  --gamma 0.99 \
  --gae-lambda 0.95 \
  --ent-coef 0.0 \
  --checkpoint-freq 100000 \
  --run-dir "${RUN_DIR}" \
  --eval-episodes 100 \
  2>&1 | tee "${LOG_DIR}/${RUN_NAME}.log"
```

기본 실행에서는 `tqdm` 진행률 표시가 켜진다. 학습 중 현재 env timestep, 처리 속도, ETA와 함께 완료 episode 통계를 같은 tmux pane에서 볼 수 있다.

진행바 postfix 의미:

- `eps`: 학습 중 완료된 episode 수
- `ep_ts`: 완료 episode당 평균 learner timestep 수
- `ep/100k`: 100000 learner timestep당 완료 episode 수
- `ep_wr`: 학습 중 완료 episode 기준 learner 승률

tmux detach:

```text
Ctrl-b d
```

다시 붙기:

```bash
tmux attach -t ppo_random_s0
```

## 5. 모니터링

별도 터미널에서 확인:

```bash
tail -f logs/ppo/random_seed_0_1m_nenv16.log
watch -n 5 nvidia-smi
```

`tqdm`은 터미널에서 보기 좋은 동적 progress bar를 출력한다. `tee`로 저장한 로그에는 carriage return이 포함될 수 있다. 깔끔한 파일 로그가 더 중요하면 학습 실행 시 `--no-progress-bar`를 추가하고, 대신 SB3 로그가 필요할 때 `--verbose 1`을 같이 사용한다.

산출물 확인:

```bash
find runs/ppo/random_seed_0_1m_nenv16 -maxdepth 2 -type f | sort
python -m json.tool runs/ppo/random_seed_0_1m_nenv16/config.json >/dev/null
```

학습 중 완료된 episode 통계 확인:

```bash
tail -n 5 runs/ppo/random_seed_0_1m_nenv16/episodes.jsonl
```

`episodes.jsonl`은 완료된 게임 한 판마다 다음 정보를 남긴다.

- `timesteps`: 해당 episode가 끝난 시점의 learner timestep
- `learner_decisions`: learner가 그 episode에서 선택한 action 수
- `turn_count`: 게임 전체 turn 수
- `decision_count`: 게임 전체 decision 수
- `winner`
- `learner_win`

이 파일을 보면 `total_timesteps=1000000`이 실제로 몇 판 정도의 게임 경험에 해당하는지 추정할 수 있다. 최종 `summary.json`에도 `episode_stats`가 들어간다.

checkpoint 확인:

```bash
find runs/ppo/random_seed_0_1m_nenv16/checkpoints -type f | sort
```

`--checkpoint-freq 100000`은 약 100000 env timesteps마다 중간 모델을 저장한다. `n_envs > 1`에서는 callback 호출 횟수와 env timestep이 다르므로, 스크립트가 내부적으로 `checkpoint_freq / n_envs`에 맞춰 저장 주기를 조정한다.

## 6. 선택적 Early Stopping

기본 장기 학습 command는 `--total-timesteps`까지 끝까지 학습한다. 학습 중 주기적으로 평가하고 조건이 만족되면 일찍 멈추고 싶으면 early stopping 옵션을 추가한다.

중요한 구분:

- `--eval-episodes`는 학습 시작 전/후 빠른 평가용이다.
- `--early-stop-eval-episodes`는 학습 도중 early stopping 판단용이다.
- `scripts/evaluate_ppo.py --episodes`는 학습 완료 후 정식 평가용이다.

예시:

```bash
python -u scripts/train_ppo.py \
  --total-timesteps 1000000 \
  --seed 0 \
  --opponent random \
  --n-envs 16 \
  --device cuda \
  --learning-rate 3e-4 \
  --n-steps 2048 \
  --batch-size 2048 \
  --checkpoint-freq 100000 \
  --early-stop-eval-freq 100000 \
  --early-stop-eval-episodes 1000 \
  --early-stop-opponent random \
  --early-stop-win-rate 0.75 \
  --early-stop-min-timesteps 500000 \
  --run-dir "${RUN_DIR}" \
  --eval-episodes 100
```

의미:

- `--early-stop-eval-freq 100000`: 약 100000 env timesteps마다 평가
- `--early-stop-eval-episodes 1000`: 평가마다 1000판 실행
- `--early-stop-opponent random`: RandomAgent 상대로 평가
- `--early-stop-win-rate 0.75`: 승률이 75% 이상이면 중단
- `--early-stop-min-timesteps 500000`: 최소 500000 timesteps 전에는 중단하지 않음

수렴 정체를 기준으로 멈추고 싶으면 patience 옵션을 쓴다.

```bash
  --early-stop-eval-freq 100000 \
  --early-stop-eval-episodes 1000 \
  --early-stop-patience 5 \
  --early-stop-min-delta 0.005 \
  --early-stop-min-timesteps 500000
```

의미:

- 평가 승률이 기존 best보다 `0.5%p` 이상 좋아져야 개선으로 본다.
- 5번 연속 개선이 없으면 중단한다.
- 최소 500000 timesteps 전에는 patience 조건으로도 중단하지 않는다.

early stopping 결과는 다음 파일에 JSON Lines 형식으로 남는다.

```text
<RUN_DIR>/eval_during_training.jsonl
```

주의:

- early stopping은 수렴을 수학적으로 증명하지 않는다.
- 윷놀이는 stochastic game이므로 평가 판수가 너무 작으면 운에 의해 조기 중단될 수 있다.
- 학습 중 early stopping 평가는 `1000`판 정도로 시작하고, 최종 보고용 평가는 학습 종료 후 opponent별 `10000`판 이상을 별도로 실행한다.

## 7. 학습 완료 후 평가

학습이 정상 종료되면 `model.zip`이 있어야 한다.

```bash
test -f runs/ppo/random_seed_0_1m_nenv16/model.zip
```

세 baseline opponent에 대해 평가한다. M8 최종 평가 전까지 M6 장기 학습 평가는 opponent당 `1000`판을 기본값으로 둔다.

```bash
RUN_DIR=runs/ppo/random_seed_0_1m_nenv16

python scripts/evaluate_ppo.py \
  --model-path "${RUN_DIR}/model.zip" \
  --episodes 1000 \
  --seed 1000 \
  --opponent random \
  --device cuda \
  --output "${RUN_DIR}/eval_random_1000.json"

python scripts/evaluate_ppo.py \
  --model-path "${RUN_DIR}/model.zip" \
  --episodes 1000 \
  --seed 2000 \
  --opponent capture_first \
  --device cuda \
  --output "${RUN_DIR}/eval_capture_first_1000.json"

python scripts/evaluate_ppo.py \
  --model-path "${RUN_DIR}/model.zip" \
  --episodes 1000 \
  --seed 3000 \
  --opponent greedy_finish \
  --device cuda \
  --output "${RUN_DIR}/eval_greedy_finish_1000.json"
```

평가 통과 기준:

- 세 평가 command가 모두 정상 종료된다.
- 각 JSON의 `illegal_action_count`가 `0`이다.
- `wins + losses == episodes`다.
- `average_turns`와 `average_decisions`가 양수다.

빠른 확인:

```bash
python - <<'PY'
import json
from pathlib import Path

run_dir = Path("runs/ppo/random_seed_0_1m_nenv16")
for path in sorted(run_dir.glob("eval_*_1000.json")):
    data = json.loads(path.read_text())
    print(
        path.name,
        "win_rate=", data["win_rate"],
        "illegal=", data["illegal_action_count"],
        "avg_turns=", data["average_turns"],
        "avg_decisions=", data["average_decisions"],
    )
PY
```

## 8. Seed 반복 실행

첫 run이 정상 종료되면 seed만 바꿔 반복한다.

Seed 1:

```bash
tmux new -s ppo_random_s1
```

```bash
cd /home/david-nam/work-space/RL-yutnori
source .venv/bin/activate

RUN_NAME=random_seed_1_1m_nenv16
RUN_DIR=runs/ppo/${RUN_NAME}
LOG_DIR=logs/ppo
mkdir -p "${LOG_DIR}"

python -u scripts/train_ppo.py \
  --total-timesteps 1000000 \
  --seed 1 \
  --opponent random \
  --n-envs 16 \
  --device cuda \
  --learning-rate 3e-4 \
  --n-steps 2048 \
  --batch-size 2048 \
  --gamma 0.99 \
  --gae-lambda 0.95 \
  --ent-coef 0.0 \
  --checkpoint-freq 100000 \
  --run-dir "${RUN_DIR}" \
  --eval-episodes 100 \
  2>&1 | tee "${LOG_DIR}/${RUN_NAME}.log"
```

Seed 2도 `seed`, `RUN_NAME`, tmux session 이름만 바꿔 실행한다.

M6 장기 학습의 최소 추천 반복:

- `seed=0`
- `seed=1`
- `seed=2`

M8 최종 평가로 넘어가기 전 권장 반복:

- `seed=0..4`

## 9. 실패 대응

### 9.1 run directory가 이미 존재하는 경우

증상:

```text
FileExistsError: run directory is not empty
```

의미:

- 기존 산출물을 실수로 덮어쓰지 않도록 보호한 것이다.

대응:

- 새 `RUN_NAME`을 사용한다.
- 장기 학습에서는 `--overwrite`를 사용하지 않는다.

### 9.2 CUDA가 보이지 않는 경우

확인:

```bash
nvidia-smi
python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
PY
```

대응:

- `nvidia-smi`가 실패하면 VM/GPU 장치 접근 문제다.
- `nvidia-smi`는 성공하지만 PyTorch CUDA가 실패하면 `requirements.txt` 기준으로 재설치한다.

```bash
python -m pip install -r requirements.txt
```

### 9.3 학습이 비정상적으로 오래 걸리는 경우

확인할 것:

- `tail -f logs/ppo/<run>.log`
- `watch -n 5 nvidia-smi`
- tmux pane의 `tqdm` progress bar와 ETA
- CPU 사용률
- checkpoint 파일 생성 여부

현 단계에서는 episode length limit을 추가하지 않는다. 게임이 비정상적으로 길어지는 증거가 있으면 다음을 기록하고 별도 판단한다.

- seed
- run directory
- 평균 decision 수
- 최대 decision 수
- 마지막 checkpoint
- 실패 command와 log 일부

### 9.4 중간 checkpoint만 남은 경우

최종 `model.zip`이 없더라도 checkpoint는 평가할 수 있다.

```bash
python scripts/evaluate_ppo.py \
  --model-path runs/ppo/random_seed_0_1m_nenv16/checkpoints/<checkpoint>.zip \
  --episodes 100 \
  --seed 9000 \
  --opponent random \
  --device cuda \
  --output runs/ppo/random_seed_0_1m_nenv16/eval_checkpoint_random.json
```

M6 현재 스크립트는 checkpoint에서 자동 resume하는 CLI를 제공하지 않는다. checkpoint는 중간 정책 보존과 사후 평가용으로 사용한다.

### 9.5 early stopping이 너무 빨리 발생하는 경우

확인할 것:

- `eval_during_training.jsonl`
- `early_stop_eval_episodes`
- `early_stop_min_timesteps`
- `early_stop_win_rate`
- `early_stop_patience`

대응:

- `early_stop_eval_episodes`를 늘린다.
- `early_stop_min_timesteps`를 늘린다.
- threshold 기준이면 `early_stop_win_rate`를 높인다.
- patience 기준이면 `early_stop_patience`를 늘리거나 `early_stop_min_delta`를 줄인다.

## 10. 완료 보고 기준

장기 학습 run 하나가 끝나면 다음을 보고한다.

- 실행 command
- run directory
- git commit hash
- seed
- total timesteps
- n_envs
- device
- checkpoint 저장 여부
- early stopping 사용 여부와 중단 사유
- 학습 시작/종료 시각
- 학습 중 문제 발생 여부
- Random/CaptureFirst/GreedyFinish 평가 결과
- 각 평가의 `illegal_action_count`
- 성능 판단 유보 여부

M6 장기 학습 준비 단계의 검증은 다음으로 충분하다.

- checkpoint option이 짧은 smoke에서 실제 checkpoint zip을 만든다.
- `python -m pytest`가 계속 통과한다.
- run directory 보호가 유지된다.
- 장기 실행 command가 seed만 바꿔 반복 가능하게 문서화되어 있다.
