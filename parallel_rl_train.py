"""
parallel_rl_train.py
--------------------
Launch multiple independent rl_trainer.py runs in parallel and keep the best model.

This is useful on multi-core CPUs (like a 7800X3D) where one trainer process
cannot fully utilize all cores.

Example
-------
python parallel_rl_train.py --runs 8 --max-parallel 8 --episodes 500 --episode-length 250
python parallel_rl_train.py --runs 12 --max-parallel 12 --episodes 500 --episode-length 250 --alpha 0.01 --epsilon-decay-episodes 400 --eval-every 10 --eval-episodes 30 --numpy-threads 1
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


ROOT = Path(__file__).resolve().parent
DEFAULT_RUNS_DIR = ROOT / "models" / "runs"
DEFAULT_BEST_OUT = ROOT / "models" / "linear_double_q_weights.npz"


@dataclass
class Job:
    run_index: int
    seed: int
    save_path: Path
    proc: subprocess.Popen


def _build_trainer_cmd(args: argparse.Namespace, seed: int, save_path: Path) -> List[str]:
    return [
        sys.executable,
        "rl_trainer.py",
        "--episodes",
        str(args.episodes),
        "--episode-length",
        str(args.episode_length),
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--max-cars",
        str(args.max_cars),
        "--spawn-rate",
        str(args.spawn_rate),
        "--gamma",
        str(args.gamma),
        "--alpha",
        str(args.alpha),
        "--epsilon-start",
        str(args.epsilon_start),
        "--epsilon-end",
        str(args.epsilon_end),
        "--epsilon-decay-episodes",
        str(args.epsilon_decay_episodes),
        "--eval-every",
        str(args.eval_every),
        "--eval-episodes",
        str(args.eval_episodes),
        "--seed",
        str(seed),
        "--save-path",
        str(save_path),
    ]


def _spawn_job(args: argparse.Namespace, run_index: int, seed: int, run_dir: Path) -> Job:
    save_path = run_dir / f"run_{run_index:02d}_seed_{seed}.npz"
    log_path = run_dir / f"run_{run_index:02d}_seed_{seed}.log"

    cmd = _build_trainer_cmd(args, seed=seed, save_path=save_path)

    env = os.environ.copy()
    if args.numpy_threads is not None:
        t = str(max(1, args.numpy_threads))
        env["OMP_NUM_THREADS"] = t
        env["MKL_NUM_THREADS"] = t
        env["OPENBLAS_NUM_THREADS"] = t
        env["NUMEXPR_NUM_THREADS"] = t

    log_file = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=env,
    )

    return Job(run_index=run_index, seed=seed, save_path=save_path, proc=proc)


def _read_meta(meta_path: Path) -> Optional[Dict[str, float]]:
    if not meta_path.exists():
        return None
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        result = data.get("result", {})
        return {
            "best_eval_reward": float(result.get("best_eval_reward", -float("inf"))),
            "best_episode": float(result.get("best_episode", -1)),
        }
    except Exception:
        return None


def _pick_best(run_dir: Path) -> Optional[Dict[str, object]]:
    best: Optional[Dict[str, object]] = None

    for meta_path in sorted(run_dir.glob("*.json")):
        info = _read_meta(meta_path)
        if info is None:
            continue

        model_path = meta_path.with_suffix(".npz")
        if not model_path.exists():
            continue

        candidate = {
            "meta_path": meta_path,
            "model_path": model_path,
            "best_eval_reward": float(info["best_eval_reward"]),
            "best_episode": float(info["best_episode"]),
        }

        if best is None or candidate["best_eval_reward"] > best["best_eval_reward"]:
            best = candidate

    return best


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parallel multi-seed RL training runner")

    parser.add_argument("--runs", type=int, default=8, help="Total independent runs to launch")
    parser.add_argument("--max-parallel", type=int, default=8, help="Max processes running at once")
    parser.add_argument("--base-seed", type=int, default=42, help="Seed for run 1; subsequent runs increment")

    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--episode-length", type=int, default=250)
    parser.add_argument("--width", type=int, default=20)
    parser.add_argument("--height", type=int, default=20)
    parser.add_argument("--max-cars", type=int, default=30)
    parser.add_argument("--spawn-rate", type=float, default=0.3)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--epsilon-start", type=float, default=0.25)
    parser.add_argument("--epsilon-end", type=float, default=0.02)
    parser.add_argument("--epsilon-decay-episodes", type=int, default=400)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=30)

    parser.add_argument(
        "--run-dir",
        type=str,
        default=str(DEFAULT_RUNS_DIR),
        help="Directory for per-run model, metadata, and logs",
    )
    parser.add_argument(
        "--best-out",
        type=str,
        default=str(DEFAULT_BEST_OUT),
        help="Path to copy the best model to",
    )
    parser.add_argument(
        "--numpy-threads",
        type=int,
        default=1,
        help="Thread cap per process for BLAS/OpenMP libs (1 is recommended for many parallel jobs)",
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    runs = max(1, args.runs)
    max_parallel = max(1, min(args.max_parallel, runs))
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    best_out = Path(args.best_out)
    if not best_out.is_absolute():
        best_out = ROOT / best_out
    best_out.parent.mkdir(parents=True, exist_ok=True)

    print("=== Parallel RL Training ===")
    print(f"runs={runs} | max_parallel={max_parallel} | episodes={args.episodes} | episode_length={args.episode_length}")
    print(f"run_dir={run_dir}")

    pending = [(i + 1, args.base_seed + i) for i in range(runs)]
    active: List[Job] = []
    completed = 0

    while pending or active:
        while pending and len(active) < max_parallel:
            run_index, seed = pending.pop(0)
            job = _spawn_job(args, run_index=run_index, seed=seed, run_dir=run_dir)
            active.append(job)
            print(f"Started run {run_index}/{runs} | seed={seed} | pid={job.proc.pid}")

        still_active: List[Job] = []
        for job in active:
            code = job.proc.poll()
            if code is None:
                still_active.append(job)
                continue

            completed += 1
            print(
                f"Finished run {job.run_index}/{runs} | seed={job.seed} | exit={code} | "
                f"model={job.save_path.name}"
            )

        active = still_active

        if active:
            time.sleep(1.0)

    best = _pick_best(run_dir)
    if best is None:
        print("No successful run metadata found. Check run logs in run_dir.")
        sys.exit(1)

    best_model = Path(best["model_path"])
    best_meta = Path(best["meta_path"])
    shutil.copy2(best_model, best_out)
    shutil.copy2(best_meta, best_out.with_suffix(".json"))

    print("\n=== Summary ===")
    print(f"Best eval reward: {best['best_eval_reward']:.3f}")
    print(f"Best episode: {int(best['best_episode'])}")
    print(f"Best source model: {best_model}")
    print(f"Promoted model: {best_out}")
    print(f"Promoted metadata: {best_out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
