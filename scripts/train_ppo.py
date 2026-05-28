"""Train a MaskablePPO policy on the Yutnori environment."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gymnasium  # noqa: E402
import sb3_contrib  # noqa: E402
import stable_baselines3  # noqa: E402
import torch  # noqa: E402
from sb3_contrib import MaskablePPO  # noqa: E402
from stable_baselines3.common.callbacks import (  # noqa: E402
    BaseCallback,
    CallbackList,
    CheckpointCallback,
)
from tqdm.auto import tqdm  # noqa: E402

from yutnori.training import (  # noqa: E402
    OPPONENT_NAMES,
    evaluate_maskable_policy,
    make_yutnori_vec_env,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total-timesteps", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--opponent", choices=OPPONENT_NAMES, default="random")
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--checkpoint-freq", type=int, default=0)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--early-stop-eval-freq", type=int, default=0)
    parser.add_argument("--early-stop-eval-episodes", type=int, default=100)
    parser.add_argument("--early-stop-opponent", choices=OPPONENT_NAMES, default="random")
    parser.add_argument("--early-stop-win-rate", type=float, default=None)
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0)
    parser.add_argument("--early-stop-min-timesteps", type=int, default=0)
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument("--no-progress-bar", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _validate_args(args)
    run_dir = _prepare_run_dir(args.run_dir, overwrite=args.overwrite)

    config = _config_dict(args, run_dir)
    _write_json(run_dir / "config.json", config)
    callback = _callbacks(args, run_dir)

    vec_env = make_yutnori_vec_env(
        opponent=args.opponent,
        n_envs=args.n_envs,
        seed=args.seed,
    )
    try:
        model = MaskablePPO(
            "MlpPolicy",
            vec_env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            ent_coef=args.ent_coef,
            seed=args.seed,
            device=args.device,
            tensorboard_log=str(run_dir / "tensorboard") if args.tensorboard else None,
            verbose=args.verbose,
        )

        eval_summary: dict[str, Any] = {}
        if args.eval_episodes > 0:
            before = evaluate_maskable_policy(
                model,
                opponent="random",
                episodes=args.eval_episodes,
                seed=args.seed + 10_000,
            )
            eval_summary["before_random"] = before.to_dict()
            _write_json(run_dir / "eval_before_random.json", before.to_dict())

        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callback,
            use_masking=True,
        )
        model_path = run_dir / "model.zip"
        model.save(model_path)

        if args.eval_episodes > 0:
            after = evaluate_maskable_policy(
                model,
                opponent="random",
                episodes=args.eval_episodes,
                seed=args.seed + 20_000,
            )
            eval_summary["after_random"] = after.to_dict()
            _write_json(run_dir / "eval_after_random.json", after.to_dict())

        summary = {
            "model_path": str(model_path),
            "started_at": config["started_at"],
            "finished_at": datetime.now(UTC).isoformat(),
            "checkpoint_dir": config["checkpoint_dir"],
            "target_total_timesteps": args.total_timesteps,
            "trained_timesteps": model.num_timesteps,
            "evaluation": eval_summary,
        }
        _write_json(run_dir / "summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
    finally:
        vec_env.close()


def _validate_args(args: argparse.Namespace) -> None:
    if args.total_timesteps <= 0:
        raise ValueError("total_timesteps must be positive")
    if args.n_envs <= 0:
        raise ValueError("n_envs must be positive")
    if args.n_steps <= 1:
        raise ValueError("n_steps must be greater than 1")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if args.eval_episodes < 0:
        raise ValueError("eval_episodes must be non-negative")
    if args.checkpoint_freq < 0:
        raise ValueError("checkpoint_freq must be non-negative")
    if args.early_stop_eval_freq < 0:
        raise ValueError("early_stop_eval_freq must be non-negative")
    if args.early_stop_eval_episodes <= 0:
        raise ValueError("early_stop_eval_episodes must be positive")
    if args.early_stop_win_rate is not None and not (
        0.0 <= args.early_stop_win_rate <= 1.0
    ):
        raise ValueError("early_stop_win_rate must be in [0, 1]")
    if args.early_stop_patience < 0:
        raise ValueError("early_stop_patience must be non-negative")
    if args.early_stop_min_delta < 0.0:
        raise ValueError("early_stop_min_delta must be non-negative")
    if args.early_stop_min_timesteps < 0:
        raise ValueError("early_stop_min_timesteps must be non-negative")
    rollout_size = args.n_steps * args.n_envs
    if args.batch_size > rollout_size:
        raise ValueError("batch_size must be <= n_steps * n_envs")


def _prepare_run_dir(run_dir: Path, *, overwrite: bool) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    if any(run_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"run directory is not empty: {run_dir}. "
            "Use --overwrite or choose a new --run-dir."
        )
    return run_dir


class TqdmProgressCallback(BaseCallback):
    """Show env-timestep progress and ETA during long PPO runs."""

    def __init__(self, total_timesteps: int) -> None:
        super().__init__(verbose=0)
        self.total_timesteps = total_timesteps
        self._progress_bar: tqdm | None = None
        self._last_num_timesteps = 0

    def _on_training_start(self) -> None:
        self._last_num_timesteps = 0
        self._progress_bar = tqdm(
            total=self.total_timesteps,
            desc="PPO training",
            unit="ts",
            dynamic_ncols=True,
            leave=True,
        )

    def _on_step(self) -> bool:
        if self._progress_bar is None:
            return True

        delta = self.num_timesteps - self._last_num_timesteps
        if delta > 0:
            self._progress_bar.update(delta)
            self._last_num_timesteps = self.num_timesteps
        return True

    def _on_training_end(self) -> None:
        if self._progress_bar is None:
            return

        self._progress_bar.close()
        self._progress_bar = None


class MaskableEarlyStoppingCallback(BaseCallback):
    """Periodically evaluate with action masks and stop on configured criteria."""

    def __init__(
        self,
        *,
        eval_freq: int,
        eval_episodes: int,
        opponent: str,
        seed: int,
        min_timesteps: int,
        win_rate_threshold: float | None,
        patience: int,
        min_delta: float,
        output_path: Path,
    ) -> None:
        super().__init__(verbose=0)
        self.eval_freq = eval_freq
        self.eval_episodes = eval_episodes
        self.opponent = opponent
        self.seed = seed
        self.min_timesteps = min_timesteps
        self.win_rate_threshold = win_rate_threshold
        self.patience = patience
        self.min_delta = min_delta
        self.output_path = output_path
        self._next_eval_timestep = eval_freq
        self._eval_index = 0
        self._best_win_rate: float | None = None
        self._no_improvement_count = 0

    def _on_training_start(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text("")

    def _on_step(self) -> bool:
        if self.num_timesteps < self._next_eval_timestep:
            return True

        result = evaluate_maskable_policy(
            self.model,
            opponent=self.opponent,
            episodes=self.eval_episodes,
            seed=self.seed + self._eval_index,
        )
        self._eval_index += 1

        improved = self._is_improved(result.win_rate)
        if improved:
            self._best_win_rate = result.win_rate
            self._no_improvement_count = 0
        else:
            self._no_improvement_count += 1

        can_stop = self.num_timesteps >= self.min_timesteps
        stop_reason = self._stop_reason(result.win_rate) if can_stop else None
        payload = {
            "evaluated_at": datetime.now(UTC).isoformat(),
            "timesteps": self.num_timesteps,
            "eval_index": self._eval_index,
            "best_win_rate": self._best_win_rate,
            "improved": improved,
            "no_improvement_count": self._no_improvement_count,
            "stop_reason": stop_reason,
            "result": result.to_dict(),
        }
        with self.output_path.open("a") as file:
            file.write(json.dumps(payload, sort_keys=True) + "\n")

        message = (
            f"eval {self._eval_index}: timesteps={self.num_timesteps}, "
            f"opponent={self.opponent}, win_rate={result.win_rate:.4f}, "
            f"illegal={result.illegal_action_count}, "
            f"best={self._best_win_rate:.4f}"
        )
        if stop_reason is not None:
            message = f"{message}, stop_reason={stop_reason}"
        tqdm.write(message)

        self._next_eval_timestep += self.eval_freq
        return stop_reason is None

    def _is_improved(self, win_rate: float) -> bool:
        if self._best_win_rate is None:
            return True
        return win_rate > self._best_win_rate + self.min_delta

    def _stop_reason(self, win_rate: float) -> str | None:
        if (
            self.win_rate_threshold is not None
            and win_rate >= self.win_rate_threshold
        ):
            return f"win_rate>={self.win_rate_threshold}"
        if self.patience > 0 and self._no_improvement_count >= self.patience:
            return f"no_improvement_patience={self.patience}"
        return None


def _callbacks(
    args: argparse.Namespace,
    run_dir: Path,
) -> CallbackList | BaseCallback | None:
    callbacks: list[BaseCallback] = []
    if not args.no_progress_bar:
        callbacks.append(TqdmProgressCallback(args.total_timesteps))

    checkpoint_callback = _checkpoint_callback(args, run_dir)
    if checkpoint_callback is not None:
        callbacks.append(checkpoint_callback)

    early_stopping_callback = _early_stopping_callback(args, run_dir)
    if early_stopping_callback is not None:
        callbacks.append(early_stopping_callback)

    if not callbacks:
        return None
    if len(callbacks) == 1:
        return callbacks[0]
    return CallbackList(callbacks)


def _checkpoint_callback(
    args: argparse.Namespace,
    run_dir: Path,
) -> CheckpointCallback | None:
    if args.checkpoint_freq == 0:
        return None

    checkpoint_dir = _resolve_checkpoint_dir(args, run_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_freq_calls = _checkpoint_save_freq_calls(args)
    return CheckpointCallback(
        save_freq=save_freq_calls,
        save_path=str(checkpoint_dir),
        name_prefix="ppo_yutnori",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )


def _checkpoint_save_freq_calls(args: argparse.Namespace) -> int | None:
    if args.checkpoint_freq == 0:
        return None
    return max((args.checkpoint_freq + args.n_envs - 1) // args.n_envs, 1)


def _early_stopping_callback(
    args: argparse.Namespace,
    run_dir: Path,
) -> MaskableEarlyStoppingCallback | None:
    if args.early_stop_eval_freq == 0:
        return None

    return MaskableEarlyStoppingCallback(
        eval_freq=args.early_stop_eval_freq,
        eval_episodes=args.early_stop_eval_episodes,
        opponent=args.early_stop_opponent,
        seed=args.seed + 30_000,
        min_timesteps=args.early_stop_min_timesteps,
        win_rate_threshold=args.early_stop_win_rate,
        patience=args.early_stop_patience,
        min_delta=args.early_stop_min_delta,
        output_path=run_dir / "eval_during_training.jsonl",
    )


def _resolve_checkpoint_dir(args: argparse.Namespace, run_dir: Path) -> Path | None:
    if args.checkpoint_freq == 0:
        return None
    if args.checkpoint_dir is not None:
        return args.checkpoint_dir
    return run_dir / "checkpoints"


def _config_dict(args: argparse.Namespace, run_dir: Path) -> dict[str, Any]:
    checkpoint_dir = _resolve_checkpoint_dir(args, run_dir)
    return {
        "command": sys.argv,
        "git_commit": _git_commit(),
        "started_at": datetime.now(UTC).isoformat(),
        "seed": args.seed,
        "opponent": args.opponent,
        "total_timesteps": args.total_timesteps,
        "n_envs": args.n_envs,
        "device": args.device,
        "learning_rate": args.learning_rate,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "ent_coef": args.ent_coef,
        "eval_episodes": args.eval_episodes,
        "checkpoint_freq": args.checkpoint_freq,
        "checkpoint_dir": None if checkpoint_dir is None else str(checkpoint_dir),
        "checkpoint_save_freq_calls": _checkpoint_save_freq_calls(args),
        "early_stop_eval_freq": args.early_stop_eval_freq,
        "early_stop_eval_episodes": args.early_stop_eval_episodes,
        "early_stop_opponent": args.early_stop_opponent,
        "early_stop_win_rate": args.early_stop_win_rate,
        "early_stop_patience": args.early_stop_patience,
        "early_stop_min_delta": args.early_stop_min_delta,
        "early_stop_min_timesteps": args.early_stop_min_timesteps,
        "early_stop_eval_log": (
            None
            if args.early_stop_eval_freq == 0
            else str(run_dir / "eval_during_training.jsonl")
        ),
        "tensorboard": args.tensorboard,
        "progress_bar": not args.no_progress_bar,
        "system": _system_info(),
    }


def _system_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "gymnasium": gymnasium.__version__,
        "stable_baselines3": stable_baselines3.__version__,
        "sb3_contrib": sb3_contrib.__version__,
    }
    if torch.cuda.is_available():
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
    return info


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
