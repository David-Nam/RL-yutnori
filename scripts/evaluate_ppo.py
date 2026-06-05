"""Evaluate a saved MaskablePPO policy with action masks."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sb3_contrib import MaskablePPO  # noqa: E402

from yutnori.env import OBSERVATION_MODES  # noqa: E402
from yutnori.training import (  # noqa: E402
    OPPONENT_NAMES,
    evaluate_maskable_policy,
    resolve_model_observation_mode,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--opponent", choices=OPPONENT_NAMES, required=True)
    parser.add_argument("--observation-mode", choices=OBSERVATION_MODES, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-decisions", type=int, default=10_000)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--no-progress-bar", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes < 0:
        raise ValueError("episodes must be non-negative")
    if args.max_decisions <= 0:
        raise ValueError("max_decisions must be positive")

    observation_mode = resolve_model_observation_mode(
        args.model_path,
        args.observation_mode,
    )
    model = MaskablePPO.load(args.model_path, device=args.device)
    result = evaluate_maskable_policy(
        model,
        opponent=args.opponent,
        episodes=args.episodes,
        seed=args.seed,
        observation_mode=observation_mode,
        deterministic=not args.stochastic,
        max_decisions=args.max_decisions,
        show_progress=not args.no_progress_bar,
        progress_desc=f"Evaluate {args.opponent}",
    )
    payload: dict[str, Any] = {
        "model_path": str(args.model_path),
        "evaluated_at": datetime.now(UTC).isoformat(),
        "deterministic": not args.stochastic,
        "observation_mode": observation_mode,
        **result.to_dict(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
